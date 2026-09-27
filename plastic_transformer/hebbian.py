"""
PlasticLinearHebbian: Exploratory online Hebbian fast-weight layer.
Author: Thomas Nauheimer (2026)

Note: For production and verifiable subspace protection, use PlasticLinearProjected.
PlasticLinearHebbian provides an unconstrained associative baseline for research comparison.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlasticLinearHebbian(nn.Module):
    """
    Exploratory Hebbian fast-weight layer.
    Implements effective weight matrix:
        W_eff(t) = W_slow + alpha * (U(t) @ V(t)^T)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        eta: float = 0.05,
        decay: float = 0.001,
        alpha: float = 1.0,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.eta = eta
        self.decay = decay
        self.alpha = alpha
        
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.register_buffer('U', torch.zeros((out_features, rank), **factory_kwargs))
        self.register_buffer('V', torch.zeros((in_features, rank), **factory_kwargs))
        
        self.last_input = None
        self.last_output = None
        self.plasticity_enabled = True

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 16,
        eta: float = 0.05,
        decay: float = 0.001,
        alpha: float = 1.0
    ):
        """Wraps an existing pre-trained nn.Linear layer preserving device and dtype."""
        plastic = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            eta=eta,
            decay=decay,
            alpha=alpha,
            bias=(linear.bias is not None),
            device=linear.weight.device,
            dtype=linear.weight.dtype
        )
        plastic.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            plastic.bias.data.copy_(linear.bias.data)
        return plastic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_slow = F.linear(x, self.weight, self.bias)
        
        if self.plasticity_enabled and (self.U.abs().sum() > 0 or self.V.abs().sum() > 0):
            x_proj = torch.matmul(x, self.V)
            y_fast = torch.matmul(x_proj, self.U.t())
            y = y_slow + self.alpha * y_fast
        else:
            y = y_slow
            
        if self.plasticity_enabled:
            self.last_input = x.detach().reshape(-1, self.in_features).mean(dim=0)
            self.last_output = y.detach().reshape(-1, self.out_features).mean(dim=0)
            
        return y

    def hebbian_step(self, surprise_signal: float = 1.0):
        if not self.plasticity_enabled or self.last_input is None or self.last_output is None:
            return
            
        x_pre = self.last_input
        y_post = self.last_output
        
        norm_x = torch.norm(x_pre) + 1e-7
        norm_y = torch.norm(y_post) + 1e-7
        u_delta = (y_post / norm_y).unsqueeze(1)
        v_delta = (x_pre / norm_x).unsqueeze(1)
        
        effective_eta = self.eta * surprise_signal
        
        self.U.mul_(1.0 - self.decay)
        self.V.mul_(1.0 - self.decay)
        
        self.U.add_(u_delta.expand(-1, self.rank) * (effective_eta / self.rank))
        self.V.add_(v_delta.expand(-1, self.rank) * (effective_eta / self.rank))

    def reset(self):
        """Resets fast weights to zero."""
        self.U.zero_()
        self.V.zero_()
        self.last_input = None
        self.last_output = None

    def reset_plasticity(self):
        """Alias for reset()."""
        self.reset()
