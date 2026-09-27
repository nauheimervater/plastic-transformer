"""
PlasticLinearProjected: Projected Delta-Rule Adapter with Subspace Invariance.
Author: Thomas Nauheimer
"""

import math
import torch
from torch import nn


class PlasticLinearProjected(nn.Module):
    """
    Projected Delta-Rule Adapter for a frozen nn.Linear layer.
    
    Guarantees:
    - Input-subspace protection via orthogonal projector P = I - Q Q^T.
    - Constant-time low-rank updates via thin QR and core SVD compression.
    - Zero interference on protected basis span(Q).
    """
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        protected_basis: torch.Tensor = None,
        learning_rate: float = 0.5,
        decay: float = 0.0,
        max_mass: float = 10.0,
        gate: float = 1.0
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError('base must be nn.Linear')
        if base.weight.dtype not in (torch.float32, torch.float64):
            raise ValueError('Research implementation requires float32 or float64')
        if not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError('Invalid rank')
        if not all(math.isfinite(v) for v in (learning_rate, decay, max_mass, gate)):
            raise ValueError('Parameters must be finite')
        if not (0 < learning_rate <= 1 and 0 <= decay < 1 and max_mass > 0 and 0 <= gate <= 1):
            raise ValueError('Invalid update parameters')
            
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        
        # Persist update configuration together with the tensor state
        make = base.weight.new_tensor
        self.register_buffer('learning_rate', make(learning_rate))
        self.register_buffer('decay', make(decay))
        self.register_buffer('max_mass', make(max_mass))
        self.register_buffer('gate', make(gate))
        self.register_buffer('U', base.weight.new_zeros(base.out_features, rank))
        self.register_buffer('V', base.weight.new_zeros(base.in_features, rank))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long, device=base.weight.device))
        
        if protected_basis is None:
            q = base.weight.new_empty(base.in_features, 0)
        else:
            basis = torch.as_tensor(protected_basis, device=base.weight.device, dtype=base.weight.dtype)
            if basis.ndim != 2 or basis.shape[0] != base.in_features or not torch.isfinite(basis).all():
                raise ValueError('Invalid protected input basis')
            u, s, _ = torch.linalg.svd(basis, full_matrices=False)
            threshold = max(basis.shape) * torch.finfo(basis.dtype).eps * (s.max() if s.numel() else 0)
            q = u[:, s > threshold]
        self.register_buffer('Q', q)
        if protected_basis is not None:
            self.set_protected_subspace(protected_basis)

    @torch.no_grad()
    def set_protected_subspace(self, basis: torch.Tensor = None):
        """
        Set or update the orthonormal basis Q for the protected subspace.
        Subsequent adaptations guarantee Delta W * Q = 0.
        """
        if basis is None or (isinstance(basis, torch.Tensor) and basis.numel() == 0):
            self.Q = self.base.weight.new_empty(self.base.in_features, 0)
            return
        basis = torch.as_tensor(basis, device=self.base.weight.device, dtype=self.base.weight.dtype)
        if basis.ndim != 2 or basis.shape[0] != self.base.in_features or not torch.isfinite(basis).all():
            raise ValueError(f"Protected basis must have shape ({self.base.in_features}, k)")
        u, s, _ = torch.linalg.svd(basis, full_matrices=False)
        threshold = max(basis.shape) * torch.finfo(basis.dtype).eps * (s.max() if s.numel() else 0)
        self.Q = u[:, s > threshold]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Read-only: inference forward pass never modifies weights
        return self.base(x) + self.gate * ((x @ self.V) @ self.U.T)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x - (x @ self.Q) @ self.Q.T

    @torch.no_grad()
    def adapt(self, x: torch.Tensor, target: torch.Tensor):
        if torch.is_grad_enabled():
            raise RuntimeError('Unexpected autograd state: adapt requires torch.no_grad()')
        if x.ndim != 2 or x.shape[1] != self.base.in_features or target.shape != (x.shape[0], self.base.out_features) or not x.shape[0]:
            raise ValueError('Expected batched inputs and matching target activations')
        if not torch.isfinite(x).all() or not torch.isfinite(target).all():
            raise ValueError('Nonfinite adaptation data')
        if self.gate.item() == 0:
            return {'updated': False, 'reason': 'gate_disabled'}
            
        before = self(x)
        error = target - before
        projected = self.project(x)
        
        scale = torch.sqrt(self.learning_rate * self.gate / x.shape[0])
        left = torch.cat((self.U * torch.sqrt(1 - self.decay), error.T * scale), dim=1)
        right = torch.cat((self.project(self.V.T).T * torch.sqrt(1 - self.decay), projected.T * scale), dim=1)
        
        # Compress the sum of low-rank updates without materializing full dense matrix
        ql, rl = torch.linalg.qr(left, mode='reduced')
        qr, rr = torch.linalg.qr(right, mode='reduced')
        u, s, vh = torch.linalg.svd(rl @ rr.T, full_matrices=False)
        s = s[:self.rank]
        s = s * torch.minimum(s.new_tensor(1), self.max_mass / s.norm().clamp_min(torch.finfo(s.dtype).tiny))
        new_u = (ql @ u[:, :self.rank]) * s.sqrt()
        new_v = self.project(((qr @ vh[:self.rank].T) * s.sqrt()).T).T
        
        if not torch.isfinite(new_u).all() or not torch.isfinite(new_v).all():
            raise ValueError('Update produced nonfinite factors')
            
        self.U.copy_(new_u)
        self.V.copy_(new_v)
        self.updates.add_(1)
        return {
            'updated': True,
            'mse_before': (error ** 2).mean().item(),
            'mse_after': ((self(x) - target) ** 2).mean().item()
        }

    @torch.no_grad()
    def diagnostics(self):
        mass = ((self.U.T @ self.U) * (self.V.T @ self.V)).sum().clamp_min(0).sqrt()
        interference = (self.U @ (self.V.T @ self.Q)).norm()
        return {
            'mass': mass.item(),
            'protected_interference': interference.item(),
            'updates': self.updates.item(),
            'protection': 'input_subspace'
        }

    @torch.no_grad()
    def reset(self):
        self.U.zero_()
        self.V.zero_()
        self.updates.zero_()
