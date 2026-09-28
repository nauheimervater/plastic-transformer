"""
PlasticLinearProjected: projected delta-rule fast weights for a frozen nn.Linear layer.
Author: Thomas Nauheimer
"""

import math
import torch
from torch import nn


class PlasticLinearProjected(nn.Module):
    """
    Frozen linear layer plus low-rank fast weights:

        y = W x + b + gate * (A_frozen + A_active) x,    A = U V^T

    Properties (exact in exact arithmetic, near machine precision in floating point):
    - Updates of the *active* factors are right-projected by P = I - Q Q^T, so every update
      satisfies Delta A_active Q = 0. Outputs for inputs inside span(Q) are therefore not
      changed by adaptation. Inputs with components outside span(Q) are changed.
    - Update cost O((d_in + d_out) m^2 + m^3) with m = rank + batch size; the dense
      d_out x d_in matrix is never materialized.

    Continual use: protecting inputs of something that was *already learned* requires
    `consolidate()` before growing Q. Consolidation freezes the active factors into a
    separate store that later updates never touch; only then can Q be expanded without
    erasing the learned mapping. `set_protected_subspace` refuses to run on nonzero
    active factors unless erasure is requested explicitly.
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

        make = base.weight.new_tensor
        self.register_buffer('learning_rate', make(learning_rate))
        self.register_buffer('decay', make(decay))
        self.register_buffer('max_mass', make(max_mass))
        self.register_buffer('gate', make(gate))
        self.register_buffer('U', base.weight.new_zeros(base.out_features, rank))
        self.register_buffer('V', base.weight.new_zeros(base.in_features, rank))
        self.register_buffer('updates', torch.zeros((), dtype=torch.long, device=base.weight.device))
        # Variable-size state: protected basis and consolidated (frozen) factors
        self.register_buffer('Q', base.weight.new_zeros(base.in_features, 0))
        self.register_buffer('U_frozen', base.weight.new_zeros(base.out_features, 0))
        self.register_buffer('V_frozen', base.weight.new_zeros(base.in_features, 0))
        if protected_basis is not None:
            self.set_protected_subspace(protected_basis)

    # ------------------------------------------------------------------ state handling

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Q and the frozen store change size over time; resize before the strict copy.
        for name in ('Q', 'U_frozen', 'V_frozen'):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key], device=self.U.device, dtype=self.U.dtype))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @property
    def has_active_state(self) -> bool:
        return bool(self.U.abs().max() > 0 and self.V.abs().max() > 0)

    @torch.no_grad()
    def set_protected_subspace(self, basis: torch.Tensor = None, erase_active: bool = False):
        """
        Set the protected input subspace span(Q) (orthonormalized internally).

        Raises if the active factors are nonzero: projecting them would erase what they learned
        on the newly protected inputs. Call `consolidate()` first, or pass erase_active=True
        to project the active factors deliberately.
        """
        if basis is None or (isinstance(basis, torch.Tensor) and basis.numel() == 0):
            self.Q = self.base.weight.new_zeros(self.base.in_features, 0)
            return
        basis = torch.as_tensor(basis, device=self.base.weight.device, dtype=self.base.weight.dtype)
        if basis.ndim != 2 or basis.shape[0] != self.base.in_features or not torch.isfinite(basis).all():
            raise ValueError(f"Protected basis must have shape ({self.base.in_features}, k)")
        if self.has_active_state and not erase_active:
            raise RuntimeError(
                'Active fast weights are nonzero. Changing Q would erase learned content on the '
                'newly protected inputs. Call consolidate() first, or pass erase_active=True.')
        u, s, _ = torch.linalg.svd(basis, full_matrices=False)
        threshold = max(basis.shape) * torch.finfo(basis.dtype).eps * (s.max() if s.numel() else 0)
        self.Q = u[:, s > threshold].contiguous()
        if self.has_active_state:
            self.V.copy_(self.project(self.V.T).T)

    @torch.no_grad()
    def consolidate(self):
        """Freeze the active factors into the consolidated store and reset the active part to zero.
        The store grows by `rank` columns per call; it is never modified by later updates."""
        if self.has_active_state:
            self.U_frozen = torch.cat([self.U_frozen, self.U], dim=1)
            self.V_frozen = torch.cat([self.V_frozen, self.V], dim=1)
        self.U.zero_()
        self.V.zero_()

    @torch.no_grad()
    def reset(self, include_frozen: bool = True):
        self.U.zero_()
        self.V.zero_()
        self.updates.zero_()
        if include_frozen:
            self.U_frozen = self.U_frozen[:, :0].contiguous()
            self.V_frozen = self.V_frozen[:, :0].contiguous()

    # ------------------------------------------------------------------ computation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Read-only: the forward pass never modifies state
        fast = (x @ self.V) @ self.U.T
        if self.V_frozen.shape[1]:
            fast = fast + (x @ self.V_frozen) @ self.U_frozen.T
        return self.base(x) + self.gate * fast

    def project(self, x: torch.Tensor) -> torch.Tensor:
        if self.Q.shape[1] == 0:
            return x
        return x - (x @ self.Q) @ self.Q.T

    @torch.no_grad()
    def adapt(self, x: torch.Tensor, target: torch.Tensor):
        """One projected delta-rule step toward target activations for inputs x (rows).

        Plain delta rule: stable only if learning_rate * lambda_max(X^T X / N) < 2. For language-model
        activations with large norms use PlasticModelWrapper.gradient_step (NLMS-normalized) or scale
        the error accordingly."""
        if torch.is_grad_enabled():
            raise RuntimeError('adapt() requires torch.no_grad()')
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

        # Compress the sum of low-rank terms without materializing the dense matrix
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
        interference = (self.U @ (self.V.T @ self.Q)).norm() if self.Q.shape[1] else self.U.new_zeros(())
        return {
            'mass': mass.item(),
            'protected_interference': interference.item(),  # active factors only
            'protected_dim': self.Q.shape[1],
            'frozen_rank': self.U_frozen.shape[1],
            'updates': self.updates.item(),
        }

    def state_numel(self, include_q: bool = False) -> int:
        n = self.U.numel() + self.V.numel() + self.U_frozen.numel() + self.V_frozen.numel()
        return n + (self.Q.numel() if include_q else 0)
