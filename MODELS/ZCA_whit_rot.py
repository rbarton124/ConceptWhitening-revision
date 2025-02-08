############################################
# whit_rot.py
############################################

import torch
import torch.nn as nn
from torch.nn import Parameter

__all__ = ['ZCAWhitening', 'ZCARotation']

class ZCAWhitening(nn.Module):
    """
    ZCA-based Whitening layer.
    Maintains:
      - running_mean (C)
      - running_cov  (C,C)
    Uses a momentum-based update, similar to BatchNorm.
    Optionally applies an affine transform.

    Input shape: (N, C, H, W)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_cov', torch.eye(num_features))

        # Learnable parameters (like BN)
        if self.affine:
            self.weight = Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.weight = None
            self.bias = None

    def forward(self, X: torch.Tensor):
        """
        Forward pass. If self.training, update running mean/cov.
        Then apply ZCA to whiten. If self.affine, apply scale/bias.
        """
        if X.dim() != 4:
            raise ValueError("Input must be 4D (N,C,H,W)")
        N, C, H, W = X.size()
        if C != self.num_features:
            raise ValueError(f"Expected {self.num_features} channels, got {C}.")

        # Flatten spatial dims: X_flat => shape (C, N*H*W)
        X_flat = X.permute(1, 0, 2, 3).reshape(C, -1)

        if self.training:
            # 1) batch mean
            mean = X_flat.mean(dim=1, keepdim=True)  # shape (C,1)
            X_centered = X_flat - mean
            # 2) batch covariance
            m_ = X_centered.size(1)
            cov = (X_centered @ X_centered.transpose(0, 1)) / float(m_)

            # 3) update running stats
            self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.squeeze(dim=1))
            self.running_cov.mul_(1 - self.momentum).add_(self.momentum * cov)
        else:
            # use running stats
            mean = self.running_mean.unsqueeze(1)
            X_centered = X_flat - mean
            cov = self.running_cov

        # 4) ZCA transform
        cov_reg = cov + self.eps * torch.eye(C, device=cov.device, dtype=cov.dtype)
        U, S, _ = torch.linalg.svd(cov_reg, full_matrices=False)
        zca_mat = U @ torch.diag(1.0 / torch.sqrt(S + self.eps)) @ U.transpose(-1, -2)

        # Whiten
        X_whitened = zca_mat @ X_centered
        # Reshape to (N,C,H,W)
        X_whitened = X_whitened.view(C, N, H, W).permute(1, 0, 2, 3)

        # Optional affine
        if self.affine:
            X_whitened = X_whitened * self.weight + self.bias

        return X_whitened

    def extra_repr(self):
        return (f"ZCAWhitening(num_features={self.num_features}, eps={self.eps}, "
                f"momentum={self.momentum}, affine={self.affine})")


class ZCARotation(nn.Module):
    """
    Concept Whitening module:
     1) ZCAWhitening submodule
     2) A rotation matrix (C,C)
     3) Accumulated gradient signals for concept alignment

    Similar to IterNormRotation:
     - mode = concept index or -1
     - activation_mode controls alignment approach
     - update_rotation_matrix() to do SVD-based rotation alignment
    """

    def __init__(self,
                 num_features,
                 eps=1e-5,
                 momentum=0.1,
                 activation_mode='pool_max',
                 affine=False,
                 mode=-1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.activation_mode = activation_mode
        self.affine = affine
        self.mode = mode

        # 1) ZCA submodule
        self.zca = ZCAWhitening(num_features, eps=self.eps, momentum=self.momentum, affine=False)

        # 2) Rotation matrix
        self.register_buffer('running_rot', torch.eye(num_features))

        # 3) concept grad accumulators
        # We'll assume single group => shape (C,C)
        self.register_buffer('sum_G', torch.zeros(num_features, num_features))
        self.register_buffer('counter', torch.ones(num_features) * 0.001)

        # Optional affine after rotation
        if self.affine:
            self.weight = Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = Parameter(torch.zeros(1, num_features, 1, 1))
        else:
            self.weight = None
            self.bias = None

        # For advanced alignment modes
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True)
        self.maxunpool = nn.MaxUnpool2d(kernel_size=3, stride=3)

    def forward(self, X: torch.Tensor):
        """
        1) ZCA whiten
        2) Matmul by running_rot
        3) If mode >= 0, accumulate concept alignment signals
        4) optional affine
        """
        # 1) whiten
        X_hat = self.zca(X)  # [N,C,H,W]

        N, C, H, W = X_hat.shape
        # 2) rotate => flatten => matmul => reshape
        X_reshaped = X_hat.permute(0, 2, 3, 1).reshape(-1, C)
        X_rot = X_reshaped @ self.running_rot
        X_rot = X_rot.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

        # 3) accumulate concept signals if mode >=0
        if self.mode >= 0:
            self._accumulate_concept_grad(X_hat)

        # 4) optional affine
        if self.affine:
            X_rot = X_rot * self.weight + self.bias

        return X_rot

    def _accumulate_concept_grad(self, X_hat):
        """
        Based on activation_mode, accumulate partial signals in sum_G
        for concept self.mode. This mimics your original logic for alignment.
        """
        with torch.no_grad():
            N, C, H, W = X_hat.shape
            if self.activation_mode == 'mean':
                # gradient is negative of mean => shape (C,)
                grad = -X_hat.mean(dim=(0,2,3))
                self.sum_G[:, self.mode] = self.momentum*grad + (1-self.momentum)*self.sum_G[:, self.mode]
                self.counter[self.mode] += 1

            elif self.activation_mode == 'max':
                # rotate X_hat => X_test
                X_reshaped = X_hat.permute(0, 2, 3, 1).reshape(-1, C)
                X_test = X_reshaped @ self.running_rot
                X_test = X_test.view(N, H, W, C).permute(0, 3, 1, 2)
                max_val = X_test.amax(dim=(2,3))  # [N,C]
                # accumulate partial
                # note: the full logic depends on how you originally define "max"
                grad = -(X_hat*(X_test == max_val.unsqueeze(-1).unsqueeze(-1))).sum((0,2,3)) / (N*H*W)
                self.sum_G[:, self.mode] = self.momentum*grad + (1-self.momentum)*self.sum_G[:, self.mode]
                self.counter[self.mode] += 1

            # elif self.activation_mode == 'pool_max': etc.

    def change_mode(self, mode: int):
        """Set which concept axis to accumulate into. -1 => none."""
        self.mode = mode

    def update_rotation_matrix(self):
        """
        SVD-based Procrustes on sum_G => update running_rot
        Then reset sum_G & counters.
        """
        with torch.no_grad():
            # shape (C,C)
            G = self.sum_G / self.counter.unsqueeze(1)
            U, _, V = torch.linalg.svd(G, full_matrices=False)
            R_new = U @ V.transpose(-1, -2)

            # Option A: direct assignment
            self.running_rot.copy_(R_new)

            # Option B: momentum-based blend of old R & new R
            # self.running_rot = (1 - self.momentum)*self.running_rot + self.momentum*R_new

            # Reset accumulators
            self.sum_G.zero_()
            self.counter.fill_(0.001)

    def extra_repr(self):
        return (f"ZCARotation(num_features={self.num_features}, eps={self.eps}, "
                f"momentum={self.momentum}, mode={self.mode}, "
                f"activation_mode={self.activation_mode}, affine={self.affine})")
