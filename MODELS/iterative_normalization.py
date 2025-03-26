import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

__all__ = ['iterative_normalization_py', 'IterNorm', 'IterNormRotation']

class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
        # EXACT SAME as your existing code for whitening
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().reshape(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)

            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d, device=X.device, dtype=X.dtype).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(
                input=P[0].mul(eps),
                batch1=xc,
                batch2=xc.transpose(1, 2),
                beta=1.0,
                alpha=(1.0 / m)
            )
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
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

            wm = P[ctx.T].mul_(rTr.sqrt())
            running_mean.copy_(momentum * mean + (1. - momentum)*running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum)*running_wmat)

        else:
            xc = x - running_mean
            wm = running_wmat

        xn = wm.matmul(xc)
        Xn = xn.reshape(X.size(1), X.size(0), *X.size()[2:]).transpose(0,1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        # EXACT SAME as your existing code
        (grad,) = grad_outputs
        saved = ctx.saved_variables
        xc = saved[0]
        rTr = saved[1]
        sn = saved[2].transpose(-2, -1)
        P = saved[3:]
        g, d, m = xc.size()

        g_ = grad.transpose(0,1).contiguous().reshape_as(xc)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0

        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)

            g_P.baddbmm_(
                batch1=g_tmp, batch2=P2, beta=1.5, alpha=-0.5
            )
            g_P.baddbmm_(
                batch1=P2, batch2=g_tmp, beta=1.0, alpha=-0.5
            )
            g_P.baddbmm_(
                batch1=P[k - 1].matmul(g_tmp), batch2=P[k - 1],
                beta=1.0, alpha=-0.5
            )

        g_sn += g_P
        g_tr = (
            (-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm))
            * P[0]
        ).sum((1,2), keepdim=True) * P[0]

        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2.0*g_tr) * (-0.5/m * rTr)
        g_x = torch.baddbmm(
            input=wm.matmul(g_ - g_.mean(-1, keepdim=True)),
            batch1=g_sigma,
            batch2=xc,
            beta=1.0,
            alpha=1.0
        )

        grad_input = g_x.reshape(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0,1).contiguous()
        return grad_input, None, None, None, None, None, None, None


class IterNorm(nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=4, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        if num_channels is None:
            num_channels = (num_features - 1)//num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //=2
            num_groups = num_features // num_channels
        assert num_groups>0 and num_features % num_channels==0

        self.num_groups=num_groups
        self.num_channels=num_channels
        shape=[1]*dim
        shape[1]=self.num_features
        if self.affine:
            self.weight=Parameter(torch.Tensor(*shape))
            self.bias=Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels,1))
        self.register_buffer('running_wm', torch.eye(num_channels, device=torch.device('cpu'))
                             .expand(num_groups,num_channels,num_channels).clone())
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, X):
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
    Full QCW Implementation: single-axis (mode >=0) or subspace-based (winner-takes-all).
    Includes optional free concepts via subspace_map expansions, scaled by cw_lambda.
    """
    def __init__(self, 
                 num_features, 
                 num_groups=1, 
                 num_channels=None, 
                 T=10, 
                 dim=4, 
                 eps=1e-5, 
                 momentum=0.05, 
                 affine=False,
                 mode=-1, 
                 activation_mode='pool_max',
                 cw_lambda=1.0,
                 subspace_map=None,
                 use_free=False):
        super().__init__()
        assert dim==4, "IterNormRotation only supports 4D"
        self.T=T
        self.eps=eps
        self.momentum=momentum
        self.num_features=num_features
        self.affine=affine
        self.dim=dim
        self.mode=mode
        self.activation_mode=activation_mode
        self.cw_lambda=cw_lambda
        self.subspace_map=subspace_map if subspace_map else {}
        self.use_free=use_free
        self.active_subspace=None

        assert num_groups==1, "Please keep num_groups=1"
        if num_channels is None:
            num_channels=(num_features-1)//num_groups +1
        num_groups=num_features//num_channels
        while num_features % num_channels!=0:
            num_channels//=2
            num_groups=num_features//num_channels
        assert num_groups>0 and num_features % num_channels==0

        self.num_groups=num_groups
        self.num_channels=num_channels
        shape=[1]*dim
        shape[1]=self.num_features

        self.weight=Parameter(torch.Tensor(*shape))
        self.bias=Parameter(torch.Tensor(*shape))

        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=3,return_indices=True)
        self.maxunpool=nn.MaxUnpool2d(kernel_size=3,stride=3)

        self.register_buffer('running_mean', torch.zeros(num_groups,num_channels,1))
        self.register_buffer(
            'running_wm',
            torch.eye(num_channels,device=torch.device('cpu')).expand(num_groups,num_channels,num_channels).clone()
        )
        self.register_buffer(
            'running_rot',
            torch.eye(num_channels,device=torch.device('cpu')).expand(num_groups,num_channels,num_channels).clone()
        )
        self.register_buffer('sum_G', torch.zeros(num_groups,num_channels,num_channels))
        self.register_buffer('counter', torch.ones(num_channels)*0.001)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def set_subspace(self, subspace_name:str):
        self.active_subspace=subspace_name

    def clear_subspace(self):
        self.active_subspace=None

    def update_rotation_matrix(self):
        size_R=self.running_rot.size()
        with torch.no_grad():
            G=self.sum_G / self.counter.reshape(-1,1)
            R=self.running_rot.clone().to(G.device)
            for _ in range(2):
                tau=1000
                alpha=0
                beta=1e8
                c1=1e-4
                c2=0.9

                A=torch.einsum('gin,gjn->gij',G,R)-torch.einsum('gin,gjn->gij',R,G)
                I=torch.eye(size_R[2],device=G.device).expand(*size_R)
                dF_0=-0.5*(A**2).sum()
                cnt=0
                while True:
                    Q=torch.bmm((I+0.5*tau*A).inverse(),I-0.5*tau*A)
                    Y_tau=torch.bmm(Q,R)
                    F_X=(G*R).sum()
                    F_Y_tau=(G*Y_tau).sum()
                    inv_=(I+0.5*tau*A).inverse()
                    dF_tau=-torch.bmm(
                        torch.einsum('gni,gnj->gij',G,inv_),
                        torch.bmm(A,0.5*(R+Y_tau))
                    )[0].trace()

                    if F_Y_tau>F_X+c1*tau*dF_0+1e-18:
                        beta=tau
                        tau=(beta+alpha)/2
                    elif dF_tau+1e-18< c2*dF_0:
                        alpha=tau
                        tau=(beta+alpha)/2
                    else:
                        break
                    cnt+=1
                    if cnt>500:
                        print("update fail")
                        break
                Q=torch.bmm((I+0.5*tau*A).inverse(),I-0.5*tau*A)
                R=torch.bmm(Q,R)
            self.running_rot=R
            self.counter=torch.ones(size_R[-1],device=G.device)*0.001

    def forward(self, X:torch.Tensor):
        X_hat=iterative_normalization_py.apply(
            X,self.running_mean,self.running_wm,self.num_channels,
            self.T,self.eps,self.momentum,self.training
        )
        size_X=X_hat.size()
        size_R=self.running_rot.size()
        # reshape => [B, G, C, H, W], typically G=1
        X_hat=X_hat.view(size_X[0],size_R[0],size_R[2],*size_X[2:])

        with torch.no_grad():
            # Single-axis approach if mode>=0 and no active_subspace
            if self.mode>=0 and self.active_subspace is None:
                self._accumulate_gradient_single_axis(X_hat, self.mode)

            # subspace approach if active_subspace is set
            elif self.active_subspace is not None:
                if self.active_subspace in self.subspace_map:
                    subspace_axes=self.subspace_map[self.active_subspace]
                    self._accumulate_gradient_subspace(X_hat, subspace_axes)
                else:
                    # fallback => do nothing
                    pass

        # apply rotation
        X_hat=torch.einsum('bgchw,gdc->bgdhw',X_hat,self.running_rot)
        X_hat=X_hat.view(*size_X)
        if self.affine:
            return X_hat*self.weight+self.bias
        else:
            return X_hat

    def _reduce_activation(self,X_hat):
        # X_hat shape => [B, G, C, H, W], typically G=1 => [B,C,H,W]
        B,G,C,H,W=X_hat.shape
        X_reshaped=X_hat.reshape(B,C,H,W)
        if self.activation_mode=='mean':
            act=X_reshaped.mean(dim=(2,3))
        elif self.activation_mode=='max':
            act,_=X_reshaped.reshape(B,C,-1).max(dim=2)
        elif self.activation_mode=='pos_mean':
            pos_bool=(X_reshaped>0).float()
            sum_val=(X_reshaped*pos_bool).sum(dim=(2,3))
            denom=pos_bool.sum(dim=(2,3))+1e-6
            act=sum_val/denom
        elif self.activation_mode=='pool_max':
            mp=self.maxpool(X_reshaped)
            # mp[0] is the values => shape [B,C,H',W']
            act=mp[0].reshape(B,C,-1).mean(dim=2)
        else:
            act=X_reshaped.mean(dim=(2,3))

        return act  # shape [B,C]

    def _accumulate_gradient_single_axis(self, X_hat, axis_idx):
        act=self._reduce_activation(X_hat)
        grad=-act.mean(dim=0)  # shape [C]
        self.sum_G[:,axis_idx,:]=self.momentum*grad+(1.-self.momentum)*self.sum_G[:,axis_idx,:]
        self.counter[axis_idx]+=act.shape[0]

    def _accumulate_gradient_subspace(self, X_hat, subspace_axes):
        """
        Winner-takes-all among subspace_axes. scaled by cw_lambda
        """
        B,G,C,H,W=X_hat.shape
        act=self._reduce_activation(X_hat)  # shape [B,C]
        # gather subspace portion
        if len(subspace_axes)==0:
            return
        subspace_acts=act[:, subspace_axes] # shape [B, len(subspace_axes)]
        winners=subspace_acts.argmax(dim=1) # shape [B], index in [0..len(subspace_axes)-1]

        aggregator=torch.zeros(C,C, device=act.device)
        local_counter=torch.zeros(C, device=act.device)

        for i in range(B):
            global_axis=subspace_axes[winners[i].item()]
            # negative push => aggregator[global_axis,:] += -cw_lambda * act[i,:]
            aggregator[global_axis,:]+= -self.cw_lambda*act[i,:]
            local_counter[global_axis]+=1

        aggregator=aggregator/float(B)
        for a in subspace_axes:
            self.sum_G[0,a,:]=self.momentum*aggregator[a,:]+(1.-self.momentum)*self.sum_G[0,a,:]
            self.counter[a]+=local_counter[a].item()

    def extra_repr(self):
        return (f"{self.num_features}, num_channels={self.num_channels}, T={self.T}, eps={self.eps}, "
                f"momentum={self.momentum}, affine={self.affine}, cw_lambda={self.cw_lambda}, "
                f"subspace_map={list(self.subspace_map.keys())}")
