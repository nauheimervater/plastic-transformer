"""
PlasticLinearHebbian: Hebbian/Oja-style online synaptic adaptation for PyTorch.
Author: Thomas Nauheimer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlasticLinearHebbian(nn.Module):
    """
    PlasticLinear Layer with online Hebbian dynamics.
    
    Implements effective weight matrix:
        W_eff(t) = W_slow + alpha * (U(t) @ V(t)^T)
    
    Attributes:
        weight: Frozen pre-trained base knowledge (d_out x d_in).
        U(t): Dynamic fast-weight factor (d_out x r).
        V(t): Dynamic fast-weight factor (d_in x r).
        Adapts online during inference without global backpropagation.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        eta: float = 0.05,
        decay: float = 0.001,
        alpha: float = 1.0,
        bias: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.eta = eta        # Plasticity learning rate
        self.decay = decay    # Synaptic decay rate (forgetting curve)
        self.alpha = alpha    # Scaling factor for fast weights
        
        # 1. Slow Weights (Frozen foundational memory)
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        # Initialize slow weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # 2. Fast Weights: Low-rank decomposition A(t) = U @ V^T
        # Initialized to zero so baseline behavior matches W_slow bit-for-bit initially
        self.register_buffer('U', torch.zeros(out_features, rank))
        self.register_buffer('V', torch.zeros(in_features, rank))
        
        # Runtime activation cache for Hebbian correlation
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
        """Wraps an existing pre-trained nn.Linear layer into a PlasticLinearHebbian layer."""
        plastic = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            eta=eta,
            decay=decay,
            alpha=alpha,
            bias=(linear.bias is not None)
        )
        plastic.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            plastic.bias.data.copy_(linear.bias.data)
        return plastic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward pass: y_slow = x @ W_slow^T + bias
        y_slow = F.linear(x, self.weight, self.bias)
        
        # Fast weight associative projection: y_fast = (x @ V) @ U^T
        if self.plasticity_enabled and (self.U.abs().sum() > 0 or self.V.abs().sum() > 0):
            x_proj = torch.matmul(x, self.V)
            y_fast = torch.matmul(x_proj, self.U.t())
            y = y_slow + self.alpha * y_fast
        else:
            y = y_slow
            
        # Cache mean activations for online Hebbian update
        if self.plasticity_enabled:
            self.last_input = x.detach().reshape(-1, self.in_features).mean(dim=0)
            self.last_output = y.detach().reshape(-1, self.out_features).mean(dim=0)
            
        return y

    def hebbian_step(self, surprise_signal: float = 1.0):
        """
        Executes online synaptic update:
            Delta A = eta * (z_post (x) z_pre^T) - decay * A
        Normalized via Oja stability condition.
        """
        if not self.plasticity_enabled or self.last_input is None or self.last_output is None:
            return
            
        x_pre = self.last_input
        y_post = self.last_output
        
        # Oja normalization
        norm_x = torch.norm(x_pre) + 1e-7
        norm_y = torch.norm(y_post) + 1e-7
        u_delta = (y_post / norm_y).unsqueeze(1)
        v_delta = (x_pre / norm_x).unsqueeze(1)
        
        effective_eta = self.eta * surprise_signal
        
        # Update dynamic low-rank buffers
        self.U.mul_(1.0 - self.decay)
        self.V.mul_(1.0 - self.decay)
        
        self.U.add_(u_delta.expand(-1, self.rank) * (effective_eta / self.rank))
        self.V.add_(v_delta.expand(-1, self.rank) * (effective_eta / self.rank))

    def reset_plasticity(self):
        """Resets the fast weights back to zero (amnesia baseline)."""
        self.U.zero_()
        self.V.zero_()
        self.last_input = None
        self.last_output = None
