import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

__all__ = ['iterative_normalization', 'IterNorm', 'IterNormRotation']

class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
        # (Same original iterative norm logic)
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().reshape(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)

            P = [None]*(ctx.T+1)
            P[0] = torch.eye(d, device=X.device, dtype=X.dtype).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(
                input=P[0].mul(eps),
                batch1=xc,
                batch2=xc.transpose(1,2),
                beta=1.0,
                alpha=(1.0/m)
            )
            rTr = (Sigma*P[0]).sum((1,2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma*rTr
            saved.append(Sigma_N)

            for k in range(ctx.T):
                P_k3 = torch.matrix_power(P[k], 3)
                P[k+1] = torch.baddbmm(
                    input=P[k].mul(1.5),
                    batch1=P_k3,
                    batch2=Sigma_N,
                    beta=1.0,
                    alpha=-0.5
                )
            saved.extend(P)

            # whiten matrix => Sigma^{-1/2}
            wm = P[ctx.T].mul_(rTr.sqrt())

            # update buffers
            running_mean.copy_(momentum*mean + (1. - momentum)*running_mean)
            running_wmat.copy_(momentum*wm + (1. - momentum)*running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat

        xn = wm.matmul(xc)
        Xn = xn.reshape(X.size(1), X.size(0), *X.size()[2:]).transpose(0,1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        saved = ctx.saved_variables
        xc = saved[0]  # [g, d, m]
        rTr = saved[1] # [g, 1, 1]
        sn = saved[2].transpose(-2, -1)  # normalized Sigma
        P  = saved[3:]
        g, d, m = xc.size()

        g_ = grad.transpose(0,1).contiguous().reshape_as(xc)
        g_wm = g_.matmul(xc.transpose(-2,-1))
        g_P = g_wm * rTr.sqrt()
        wm  = P[ctx.T]
        g_sn= 0

        for k in range(ctx.T,1,-1):
            P[k-1].transpose_(-2,-1)
            P2 = P[k-1].matmul(P[k-1])
            g_sn += P2.matmul(P[k-1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)

            g_P.baddbmm_(batch1=g_tmp, batch2=P2, beta=1.5, alpha=-0.5)
            g_P.baddbmm_(batch1=P2,    batch2=g_tmp, beta=1.0, alpha=-0.5)
            g_P.baddbmm_(batch1=P[k-1].matmul(g_tmp), batch2=P[k-1], beta=1.0, alpha=-0.5)

        g_sn += g_P

        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2,-1).matmul(wm)) * P[0]).sum((1,2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2,-1) + 2.0*g_tr)*(-0.5/m * rTr)

        g_x = torch.baddbmm(
            input=wm.matmul(g_ - g_.mean(-1,keepdim=True)),
            batch1=g_sigma, batch2=xc,
            beta=1.0, alpha=1.0
        )
        grad_input = g_x.reshape(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0,1).contiguous()
        return grad_input, None, None, None, None, None, None, None


class IterNorm(nn.Module):
    """
    A standard iterative normalization layer that doesn't do concept whitening rotation.
    """
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=4, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super().__init__()
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        if num_channels is None:
            num_channels = (num_features-1)//num_groups +1
        num_groups = num_features//num_channels
        while num_features%num_channels!=0:
            num_channels//=2
            num_groups = num_features//num_channels
        assert num_groups>0 and num_features%num_groups==0
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape=[1]*dim
        shape[1]=self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias   = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        self.register_buffer('running_wm', torch.eye(num_channels, device=torch.device('cpu'))
                             .expand(num_groups,num_channels,num_channels).clone())
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(
            X, self.running_mean, self.running_wm, self.num_channels,
            self.T, self.eps, self.momentum, self.training
        )
        if self.affine:
            return X_hat*self.weight + self.bias
        else:
            return X_hat


class IterNormRotation(nn.Module):
    """
    An Iterative Normalization layer that also does concept whitening rotation (CW).
    Now augmented to accept bounding-box arguments:
      forward(self, X, X_redact_coords=None, orig_x_dim=None)
    """
    def __init__(self, num_features, num_groups=1, num_channels=None, T=10, dim=4, eps=1e-5, momentum=0.05, affine=False,
                 mode=-1, activation_mode='pool_max', cw_lambda=0.1):
        super().__init__()
        assert dim==4, "IterNormRotation only supports 4D inputs"
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        self.mode = mode
        self.activation_mode = activation_mode
        self.cw_lambda = cw_lambda

        # A toggle to do bounding-box redaction if needed
        self.use_redaction = False

        # Force num_groups=1 for typical usage
        assert num_groups==1, "Please keep num_groups=1"
        if num_channels is None:
            num_channels = (num_features-1)//num_groups+1
        num_groups = num_features//num_channels
        while num_features%num_channels!=0:
            num_channels//=2
            num_groups = num_features//num_channels
        assert num_groups>0 and num_features%num_groups==0

        self.num_groups = num_groups
        self.num_channels = num_channels
        shape=[1]*dim
        shape[1]=self.num_features

        # optional affine
        self.weight = Parameter(torch.Tensor(*shape))
        self.bias   = Parameter(torch.Tensor(*shape))

        # Some pool ops used for 'pool_max' activation alignment
        self.maxpool   = nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=3, stride=3)

        # buffers
        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        self.register_buffer('running_wm', torch.eye(num_channels,device=torch.device('cpu'))
                             .expand(num_groups,num_channels,num_channels).clone())
        self.register_buffer('running_rot',torch.eye(num_channels,device=torch.device('cpu'))
                             .expand(num_groups,num_channels,num_channels).clone())
        self.register_buffer('sum_G', torch.zeros(num_groups,num_channels,num_channels))
        self.register_buffer('counter', torch.ones(num_channels)*0.001)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor, X_redact_coords=None, orig_x_dim=None):
        """
        Now we accept bounding-box arguments: (X_redact_coords, orig_x_dim).
        If self.use_redaction==True and X_redact_coords is not None,
        we can do partial zeroing of X, etc.
        """
        # 1) Do iterative whitening
        X_hat = iterative_normalization_py.apply(
            X, self.running_mean, self.running_wm, self.num_channels,
            self.T, self.eps, self.momentum, self.training
        )

        # 2) If you want bounding-box gating, do it here:
        if self.use_redaction and X_redact_coords is not None:
            # e.g. X_hat = your_redact_function(X_hat, X_redact_coords, orig_x_dim)
            # Example:
            # X_hat = redact(X_hat, coords=X_redact_coords, orig_x_dim=orig_x_dim)
            pass

        # 3) Accumulate gradient for concept alignment if mode>=0
        size_X = X_hat.size()
        size_R = self.running_rot.size()
        # reshape for rotation
        X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

        with torch.no_grad():
            if self.mode >=0:
                # whichever activation alignment logic
                X_test = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)

                if self.activation_mode=='mean':
                    grad = -X_hat.mean((0,3,4))
                    self.sum_G[:, self.mode, :] = self.momentum*grad + (1.-self.momentum)*self.sum_G[:,self.mode,:]
                    self.counter[self.mode]+=1

                elif self.activation_mode=='max':
                    max_values = torch.max(torch.max(X_test,3,keepdim=True)[0],4,keepdim=True)[0]
                    max_bool   = (max_values==X_test)
                    denom = max_bool.to(X_hat).sum((3,4))
                    grad = -((X_hat*max_bool.to(X_hat)).sum((3,4))/denom).mean(0)
                    self.sum_G[:, self.mode, :] = self.momentum*grad+(1.-self.momentum)*self.sum_G[:,self.mode,:]
                    self.counter[self.mode]+=1

                elif self.activation_mode=='pos_mean':
                    pos_bool = (X_test>0)
                    denom = pos_bool.to(X_hat).sum((3,4))+0.0001
                    grad = -((X_hat*pos_bool.to(X_hat)).sum((3,4))/denom).mean(0)
                    self.sum_G[:,self.mode,:] = self.momentum*grad+(1.-self.momentum)*self.sum_G[:,self.mode,:]
                    self.counter[self.mode]+=1

                elif self.activation_mode=='pool_max':
                    X_test_nchw = X_test.reshape(size_X)
                    max_val, max_idx = self.maxpool(X_test_nchw)
                    X_unpool = self.maxunpool(max_val, max_idx, output_size=size_X)
                    X_unpool = X_unpool.view(size_X[0], size_R[0], size_R[2], *size_X[2:])
                    maxpool_bool=(X_test==X_unpool)
                    denom = maxpool_bool.to(X_hat).sum((3,4))
                    grad = -((X_hat*maxpool_bool.to(X_hat)).sum((3,4))/denom).mean(0)
                    self.sum_G[:, self.mode,:]= self.momentum*grad+(1.-self.momentum)*self.sum_G[:, self.mode,:]
                    self.counter[self.mode]+=1

        # 4) multiply by rotation
        X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
        X_hat = X_hat.view(*size_X)

        if self.affine:
            return X_hat*self.weight + self.bias
        return X_hat

    def update_rotation_matrix(self):
        """
        update rotation from sum_G
        """
        size_R = self.running_rot.size()
        with torch.no_grad():
            G = self.sum_G / self.counter.reshape(-1,1)
            R = self.running_rot.clone().to(G.device)
            for _ in range(2):
                tau = 1000
                alpha=0
                beta=1e8
                c1=1e-4
                c2=0.9
                A = torch.einsum('gin,gjn->gij', G, R)-torch.einsum('gin,gjn->gij', R, G)
                I = torch.eye(size_R[2], device=G.device).expand(*size_R)
                dF_0 = -0.5*(A**2).sum()
                cnt=0
                while True:
                    Q = torch.bmm((I+0.5*tau*A).inverse(), I-0.5*tau*A)
                    Y_tau = torch.bmm(Q,R)
                    F_X = (G*R).sum()
                    F_Y_tau = (G*Y_tau).sum()
                    inv_ = (I+0.5*tau*A).inverse()
                    dF_tau= -torch.bmm(
                        torch.einsum('gni,gnj->gij', G, inv_),
                        torch.bmm(A,0.5*(R+Y_tau))
                    )[0].trace()
                    if F_Y_tau>F_X+c1*tau*dF_0+1e-18:
                        beta = tau
                        tau = (beta+alpha)/2
                    elif dF_tau+1e-18<c2*dF_0:
                        alpha = tau
                        tau = (beta+alpha)/2
                    else:
                        break
                    cnt+=1
                    if cnt>500:
                        print("----update fail---")
                        break
                Q = torch.bmm((I+0.5*tau*A).inverse(), I-0.5*tau*A)
                R = torch.bmm(Q,R)
            self.running_rot = R
            self.counter=torch.ones(size_R[-1], device=G.device)*0.001

    def reset_counters(self):
        self.counter=(torch.ones(self.num_channels)*0.001).to(self.sum_G.device)
        self.sum_G.zero_()

    def extra_repr(self):
        return ("{num_features}, T={T}, eps={eps}, momentum={momentum}, "
                "affine={affine}, activation_mode={activation_mode}".format(**self.__dict__))


if __name__=='__main__':
    # Quick test
    itn = IterNormRotation(num_features=64, T=5, activation_mode='pool_max')
    itn.use_redaction=True  # if you want bounding-box gating
    print(itn)
    x = torch.randn(8,64,14,14)
    # example: bounding box coords
    coords = torch.tensor([[2,2,10,10]]*8, dtype=torch.float32)
    y = itn(x, X_redact_coords=coords, orig_x_dim=14)
    print("Output shape:", y.shape)