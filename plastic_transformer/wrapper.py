"""
PlasticModelWrapper: Inject plastic synaptic layers into any PyTorch model.
Author: Thomas Nauheimer (2026)
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from .projected_delta import PlasticLinearProjected
from .hebbian import PlasticLinearHebbian


class PlasticModelWrapper(nn.Module):
    """
    Wraps an existing PyTorch / HuggingFace model with plastic synaptic layers.
    Defaults to the mathematically verified PlasticLinearProjected layer with
    subspace protection and QR/SVD compression.
    """
    def __init__(
        self,
        base_model: nn.Module,
        target_modules: Optional[List[str]] = None,
        rank: int = 8,
        learning_rate: float = 0.5,
        decay: float = 0.0,
        max_mass: float = 10.0,
        layer_type: str = "projected"
    ):
        super().__init__()
        self.base_model = base_model
        self.plastic_layers: List[Union[PlasticLinearProjected, PlasticLinearHebbian]] = []
        self.layer_type = layer_type
        
        # Default target modules in standard Transformer architectures (Llama, Qwen, Mistral, GPT)
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "o_proj", "mlp_gate", "mlp_down", "fc1", "fc2"]
            
        self._inject_plasticity(
            self.base_model,
            target_modules,
            rank=rank,
            learning_rate=learning_rate,
            decay=decay,
            max_mass=max_mass
        )

    def _inject_plasticity(
        self,
        module: nn.Module,
        target_modules: List[str],
        rank: int,
        learning_rate: float,
        decay: float,
        max_mass: float
    ):
        for name, child in module.named_children():
            is_target = any(target in name.lower() for target in target_modules)
            if is_target and isinstance(child, nn.Linear):
                # Ensure child weights are frozen
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                    
                if self.layer_type == "projected":
                    adapter = PlasticLinearProjected(
                        base=child,
                        rank=rank,
                        learning_rate=learning_rate,
                        decay=decay,
                        max_mass=max_mass
                    )
                else:
                    adapter = PlasticLinearHebbian.from_linear(
                        linear=child,
                        rank=rank,
                        decay=decay
                    )
                    
                setattr(module, name, adapter)
                self.plastic_layers.append(adapter)
            else:
                self._inject_plasticity(child, target_modules, rank, learning_rate, decay, max_mass)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def reset_synapses(self):
        """Resets fast weights across all plastic layers to zero."""
        for layer in self.plastic_layers:
            layer.reset()

    def set_gate(self, gate_value: float):
        """Sets gate scaling factor [0.0, 1.0] across all plastic layers."""
        for layer in self.plastic_layers:
            if hasattr(layer, 'gate'):
                layer.gate.fill_(gate_value)

    def save_synaptic_memory(self, filepath: str):
        """Saves only the dynamic fast-weight buffers (U, V) - typically < 5 MB."""
        memory_state = {
            f"layer_{i}": {"U": layer.U.cpu(), "V": layer.V.cpu()}
            for i, layer in enumerate(self.plastic_layers)
        }
        torch.save(memory_state, filepath)

    def load_synaptic_memory(self, filepath: str):
        """Loads previously consolidated episodic memory into live synapses."""
        memory_state = torch.load(filepath, weights_only=True)
        for i, layer in enumerate(self.plastic_layers):
            key = f"layer_{i}"
            if key in memory_state:
                layer.U.copy_(memory_state[key]["U"].to(layer.U.device))
                layer.V.copy_(memory_state[key]["V"].to(layer.V.device))
