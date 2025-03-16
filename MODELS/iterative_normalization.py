"""
Reference:  Iterative Normalization: Beyond Standardization towards Efficient Whitening, CVPR 2019

- Paper:
- Code: https://github.com/huangleiBuaa/IterNorm
"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch

__all__ = ['iterative_normalization', 'IterNorm']

class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
        # change NxCxHxW to (G x D) x (NxHxW), i.e., g*d*m
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().reshape(ctx.g, nc, -1)  # safer than .view(...)
        _, d, m = x.size()
        saved = []
        if training:
            # calculate centered activation by subtracting mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)

            # covariance matrix
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d, device=X.device, dtype=X.dtype).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(
                input=P[0].mul(eps),
                batch1=xc,
                batch2=xc.transpose(1, 2),
                beta=1.0,
                alpha=(1.0 / m)
            )

            # reciprocal trace of Sigma
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)

            for k in range(ctx.T):
                P_k3 = torch.matrix_power(P[k], 3)
                P[k + 1] = torch.baddbmm(
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
            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat

        xn = wm.matmul(xc)
        Xn = xn.reshape(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs
        saved = ctx.saved_variables
        xc = saved[0]       # [g, d, m]
        rTr = saved[1]      # [g, 1, 1]
        sn = saved[2].transpose(-2, -1)  # [g, d, d] => normalized Sigma
        P = saved[3:]       # length T+1
        g, d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().reshape_as(xc)  # [g, d, m]
        g_wm = g_.matmul(xc.transpose(-2, -1))  # [g, d, d]
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0

        # loop from T down to 2
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)

            # fixed baddbmm_ calls to new style
            g_P.baddbmm_(
                batch1=g_tmp,
                batch2=P2,
                beta=1.5,
                alpha=-0.5
            )
            g_P.baddbmm_(
                batch1=P2,
                batch2=g_tmp,
                beta=1.0,
                alpha=-0.5
            )
            g_P.baddbmm_(
                batch1=P[k - 1].matmul(g_tmp),
                batch2=P[k - 1],
                beta=1.0,
                alpha=-0.5
            )

        g_sn += g_P

        g_tr = (
            (-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm))
            * P[0]
        ).sum((1, 2), keepdim=True) * P[0]

        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2.0 * g_tr) * (-0.5 / m * rTr)

        # we want 'g_sigma' as batch1, 'xc' as batch2
        g_x = torch.baddbmm(
            input=wm.matmul(g_ - g_.mean(-1, keepdim=True)),  # shape [g, d, m]
            batch1=g_sigma,
            batch2=xc,
            beta=1.0,
            alpha=1.0
        )

        grad_input = g_x.reshape(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None


class IterNorm(torch.nn.Module):
    def __init__(self, num_features, num_groups=1, num_channels=None, T=5, dim=4, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNorm, self).__init__()
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, \
            "num features={}, num groups={}".format(num_features, num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        self.register_buffer('running_wm', torch.eye(num_channels, device=torch.device('cpu'))
                             .expand(num_groups, num_channels, num_channels).clone())
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(
            X, self.running_mean, self.running_wm, self.num_channels,
            self.T, self.eps, self.momentum, self.training
        )
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return ('{num_features}, num_channels={num_channels}, T={T}, eps={eps}, '
                'momentum={momentum}, affine={affine}'.format(**self.__dict__))


class IterNormRotation(torch.nn.Module):
    """
    Concept Whitening Module

    The Whitening part is adapted from IterNorm. The core of CW module is learning
    an extra rotation matrix R that align target concepts with the output feature maps.
    Each subconcept should have its own distinct axis during the alignment process.
    """
    def __init__(self, num_features, num_groups=1, num_channels=None, T=10, dim=4, eps=1e-5, momentum=0.05, affine=False,
                 mode=-1, activation_mode='pool_max', *args, **kwargs):
        super(IterNormRotation, self).__init__()
        assert dim == 4, 'IterNormRotation does not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        self.mode = mode  # High-level concept mode
        self.subconcept_idx = -1  # Subconcept index to be aligned
        self.use_subconcept = True  # Whether to use subconcept alignment
        self.activation_mode = activation_mode

        assert num_groups == 1, 'Please keep num_groups = 1.'
        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, \
            "num features={}, num groups={}".format(num_features, num_groups)

        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features

        self.weight = Parameter(torch.Tensor(*shape))
        self.bias = Parameter(torch.Tensor(*shape))

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True)
        self.maxunpool = torch.nn.MaxUnpool2d(kernel_size=3, stride=3)

        # running mean
        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        self.register_buffer(
            'running_wm',
            torch.eye(num_channels, device=torch.device('cpu')).expand(num_groups, num_channels, num_channels).clone()
        )
        self.register_buffer(
            'running_rot',
            torch.eye(num_channels, device=torch.device('cpu')).expand(num_groups, num_channels, num_channels).clone()
        )
        # Store gradients for each axis (concept/subconcept)
        self.register_buffer('sum_G', torch.zeros(num_groups, num_channels, num_channels))
        # Counter for each individual axis to track update frequency
        self.register_buffer("counter", torch.ones(num_channels) * 0.001)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def update_rotation_matrix(self):
        size_R = self.running_rot.size()
        with torch.no_grad():
            G = self.sum_G / self.counter.reshape(-1, 1)
            R = self.running_rot.clone().to(G.device)
            for _ in range(2):
                tau = 1000
                alpha = 0
                beta = 1e8
                c1 = 1e-4
                c2 = 0.9

                A = torch.einsum('gin,gjn->gij', G, R) - torch.einsum('gin,gjn->gij', R, G)
                I = torch.eye(size_R[2], device=G.device).expand(*size_R)
                dF_0 = -0.5 * (A ** 2).sum()
                cnt = 0
                while True:
                    Q = torch.bmm((I + 0.5 * tau * A).inverse(), I - 0.5 * tau * A)
                    Y_tau = torch.bmm(Q, R)
                    F_X = (G * R).sum()
                    F_Y_tau = (G * Y_tau).sum()

                    inv_ = (I + 0.5 * tau * A).inverse()
                    dF_tau = -torch.bmm(
                        torch.einsum('gni,gnj->gij', G, inv_),
                        torch.bmm(A, 0.5 * (R + Y_tau))
                    )[0].trace()

                    if F_Y_tau > F_X + c1 * tau * dF_0 + 1e-18:
                        beta = tau
                        tau = (beta + alpha) / 2
                    elif dF_tau + 1e-18 < c2 * dF_0:
                        alpha = tau
                        tau = (beta + alpha) / 2
                    else:
                        break
                    cnt += 1
                    if cnt > 500:
                        print("--------------------update fail------------------------")
                        print(F_Y_tau, F_X + c1 * tau * dF_0)
                        print(dF_tau, c2 * dF_0)
                        print("-------------------------------------------------------")
                        break

                # print(tau, F_Y_tau)
                Q = torch.bmm((I + 0.5 * tau * A).inverse(), I - 0.5 * tau * A)
                R = torch.bmm(Q, R)

            self.running_rot = R
            self.counter = torch.ones(size_R[-1], device=G.device) * 0.001

    def set_subconcept(self, subconcept_idx):
        """Set the subconcept index to be aligned in the next forward pass"""
        self.subconcept_idx = subconcept_idx
        
    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(
            X, self.running_mean, self.running_wm, self.num_channels,
            self.T, self.eps, self.momentum, self.training
        )
        size_X = X_hat.size()
        size_R = self.running_rot.size()

        X_hat = X_hat.view(size_X[0], size_R[0], size_R[2], *size_X[2:])

        with torch.no_grad():
            # Determine which axis to align with - use subconcept_idx if available, otherwise use mode (high-level concept)
            target_axis = self.subconcept_idx if (self.use_subconcept and self.subconcept_idx >= 0) else self.mode
            
            if target_axis >= 0:
                if self.activation_mode == 'mean':
                    grad = -X_hat.mean((0, 3, 4))
                    self.sum_G[:, target_axis, :] = self.momentum * grad + (1. - self.momentum) * self.sum_G[:, target_axis, :]
                    self.counter[target_axis] += 1
                elif self.activation_mode == 'max':
                    X_test = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
                    max_values = torch.max(torch.max(X_test, 3, keepdim=True)[0], 4, keepdim=True)[0]
                    max_bool = (max_values == X_test)
                    denom = max_bool.to(X_hat).sum((3, 4))
                    grad = -((X_hat * max_bool.to(X_hat)).sum((3, 4)) / denom).mean(0)
                    self.sum_G[:, target_axis, :] = self.momentum * grad + (1. - self.momentum) * self.sum_G[:, target_axis, :]
                    self.counter[target_axis] += 1
                elif self.activation_mode == 'pos_mean':
                    X_test = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
                    pos_bool = (X_test > 0)
                    denom = pos_bool.to(X_hat).sum((3, 4)) + 0.0001
                    grad = -((X_hat * pos_bool.to(X_hat)).sum((3, 4)) / denom).mean(0)
                    self.sum_G[:, target_axis, :] = self.momentum * grad + (1. - self.momentum) * self.sum_G[:, target_axis, :]
                    self.counter[target_axis] += 1
                elif self.activation_mode == 'pool_max':
                    X_test = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
                    X_test_nchw = X_test.reshape(size_X)
                    maxpool_value, maxpool_indices = self.maxpool(X_test_nchw)
                    X_test_unpool = self.maxunpool(maxpool_value, maxpool_indices, output_size=size_X)
                    X_test_unpool = X_test_unpool.view(size_X[0], size_R[0], size_R[2], *size_X[2:])
                    maxpool_bool = (X_test == X_test_unpool)
                    denom = maxpool_bool.to(X_hat).sum((3, 4))
                    grad = -((X_hat * maxpool_bool.to(X_hat)).sum((3, 4)) / denom).mean(0)
                    self.sum_G[:, target_axis, :] = self.momentum * grad + (1. - self.momentum) * self.sum_G[:, target_axis, :]
                    self.counter[target_axis] += 1
                    
                # Reset subconcept_idx after use
                self.subconcept_idx = -1

        X_hat = torch.einsum('bgchw,gdc->bgdhw', X_hat, self.running_rot)
        X_hat = X_hat.view(*size_X)
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return ('{num_features}, num_channels={num_channels}, T={T}, eps={eps}, '
                'momentum={momentum}, affine={affine}'.format(**self.__dict__))


if __name__ == '__main__':
    ItN = IterNormRotation(64, num_groups=2, T=10, momentum=1, affine=False)
    print(ItN)
    ItN.train()
    x = torch.randn(16, 64, 14, 14)
    x.requires_grad_()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().reshape(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))

    y.sum().backward()
    print('x grad', x.grad.size())

    ItN.eval()
    y = ItN(x)
    z = y.transpose(0, 1).contiguous().reshape(x.size(1), -1)
    print(z.matmul(z.t()) / z.size(1))
