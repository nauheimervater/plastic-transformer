"""
PlasticModelWrapper: Easy injection of plastic synaptic layers into any PyTorch model.
Author: Thomas Nauheimer
"""

from typing import List, Optional
import torch
import torch.nn as nn
from .hebbian import PlasticLinearHebbian


class PlasticModelWrapper(nn.Module):
    """
    Wraps an existing PyTorch / HuggingFace model with plastic synaptic layers.
    Allows real-time test-time adaptation and persistent episodic memory snapshots.
    """
    def __init__(
        self,
        base_model: nn.Module,
        target_modules: Optional[List[str]] = None,
        rank: int = 16,
        eta: float = 0.05,
        decay: float = 0.001
    ):
        super().__init__()
        self.base_model = base_model
        self.plastic_layers: List[PlasticLinearHebbian] = []
        
        # Default target modules in standard Transformer architectures (Llama, Qwen, Mistral, GPT)
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "c_attn", "fc1", "fc2", "out_proj", "dense"]
            
        self._inject_plasticity(self.base_model, target_modules, rank=rank, eta=eta, decay=decay)

    def _inject_plasticity(
        self,
        module: nn.Module,
        target_modules: List[str],
        rank: int,
        eta: float,
        decay: float
    ):
        for name, child in module.named_children():
            is_target = any(target in name.lower() for target in target_modules)
            if is_target and isinstance(child, nn.Linear):
                plastic_layer = PlasticLinearHebbian.from_linear(child, rank=rank, eta=eta, decay=decay)
                setattr(module, name, plastic_layer)
                self.plastic_layers.append(plastic_layer)
            else:
                self._inject_plasticity(child, target_modules, rank, eta, decay)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def step_plasticity(self, surprise_signal: float = 1.0):
        """Triggers local Hebbian updates across all plastic layers."""
        for layer in self.plastic_layers:
            layer.hebbian_step(surprise_signal=surprise_signal)

    def reset_synapses(self):
        """Wipes fast weights (simulates standard amnesia baseline)."""
        for layer in self.plastic_layers:
            layer.reset_plasticity()

    def set_plasticity_enabled(self, enabled: bool):
        for layer in self.plastic_layers:
            layer.plasticity_enabled = enabled

    def save_synaptic_memory(self, filepath: str):
        """Saves only the lightweight plastic synapses (U, V buffers) - typically < 5 MB."""
        memory_state = {
            f"layer_{i}": {"U": layer.U.cpu(), "V": layer.V.cpu()}
            for i, layer in enumerate(self.plastic_layers)
        }
        torch.save(memory_state, filepath)

    def load_synaptic_memory(self, filepath: str):
        """Loads previously consolidated episodic memory into the model's live synapses."""
        memory_state = torch.load(filepath, weights_only=True)
        for i, layer in enumerate(self.plastic_layers):
            key = f"layer_{i}"
            if key in memory_state:
                layer.U.copy_(memory_state[key]["U"].to(layer.U.device))
                layer.V.copy_(memory_state[key]["V"].to(layer.V.device))
