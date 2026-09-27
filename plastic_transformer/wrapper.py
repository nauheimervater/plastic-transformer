"""
PlasticModelWrapper: Inject plastic synaptic layers into any PyTorch model.
Author: Thomas Nauheimer (2026)
"""

from typing import Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
from .projected_delta import PlasticLinearProjected


class PlasticModelWrapper(nn.Module):
    """
    Wraps an existing PyTorch / HuggingFace model with plastic synaptic layers.
    Uses the mathematically verified PlasticLinearProjected layer with
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
        gate: float = 1.0,
        protected_subspaces: Optional[Dict[str, torch.Tensor]] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.plastic_layers: List[PlasticLinearProjected] = []
        self.plastic_layer_map: Dict[str, PlasticLinearProjected] = {}
        
        # Standard target modules across modern LLM architectures (Llama-3, Qwen, Mistral, GPT)
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "fc1", "fc2", "mlp_gate", "mlp_down"
            ]
            
        self._inject_plasticity(
            self.base_model,
            target_modules,
            prefix="",
            rank=rank,
            learning_rate=learning_rate,
            decay=decay,
            max_mass=max_mass,
            gate=gate,
            protected_subspaces=protected_subspaces or {}
        )

    def _inject_plasticity(
        self,
        module: nn.Module,
        target_modules: List[str],
        prefix: str,
        rank: int,
        learning_rate: float,
        decay: float,
        max_mass: float,
        gate: float,
        protected_subspaces: Dict[str, torch.Tensor]
    ):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            is_target = any(target in name.lower() for target in target_modules)
            
            if is_target and isinstance(child, nn.Linear):
                # Ensure child base weights are frozen
                child.weight.requires_grad_(False)
                if child.bias is not None:
                    child.bias.requires_grad_(False)
                    
                basis = protected_subspaces.get(full_name, None)
                adapter = PlasticLinearProjected(
                    base=child,
                    rank=rank,
                    protected_basis=basis,
                    learning_rate=learning_rate,
                    decay=decay,
                    max_mass=max_mass,
                    gate=gate
                )
                    
                setattr(module, name, adapter)
                self.plastic_layers.append(adapter)
                self.plastic_layer_map[full_name] = adapter
            else:
                self._inject_plasticity(
                    child,
                    target_modules,
                    full_name,
                    rank,
                    learning_rate,
                    decay,
                    max_mass,
                    gate,
                    protected_subspaces
                )

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    @torch.no_grad()
    def calibrate_subspace(
        self,
        calibration_inputs: Union[torch.Tensor, List[Any]],
        k: int = 8,
        max_samples: int = 1024
    ) -> Dict[str, int]:
        """
        Calibrate the protected subspace span(Q) across all plastic layers by recording
        layer-wise input activations across forward passes of calibration_inputs.
        
        For each layer with accumulated inputs X in R^(N x d_in), computes the top-k
        singular vectors V_k (dominant input directions) and sets layer.set_protected_subspace(V_k).
        
        Subsequent adaptations in these layers will have zero interference (Delta W * Q = 0)
        on any input aligned with these dominant calibration directions.
        
        Returns:
            Dict mapping layer name to the effective rank of calibrated Q.
        """
        layer_inputs: Dict[str, List[torch.Tensor]] = {name: [] for name in self.plastic_layer_map}
        hooks = []
        
        for name, layer in self.plastic_layer_map.items():
            def make_hook(layer_name: str):
                def hook_fn(mod, inp):
                    x = inp[0].detach()
                    x = x.reshape(-1, x.shape[-1])
                    layer_inputs[layer_name].append(x.cpu())
                return hook_fn
            hooks.append(layer.register_forward_pre_hook(make_hook(name)))
            
        was_training = self.training
        self.eval()
        try:
            if isinstance(calibration_inputs, torch.Tensor):
                items = [calibration_inputs]
            elif isinstance(calibration_inputs, (list, tuple)):
                items = list(calibration_inputs)
            else:
                items = [calibration_inputs]
                
            for item in items:
                if isinstance(item, dict):
                    self.forward(**item)
                elif isinstance(item, (list, tuple)):
                    self.forward(*item)
                else:
                    self.forward(item)
        finally:
            for h in hooks:
                h.remove()
            self.train(was_training)
            
        results = {}
        for name, layer in self.plastic_layer_map.items():
            collected = layer_inputs[name]
            if not collected:
                results[name] = 0
                continue
            X = torch.cat(collected, dim=0)
            if X.shape[0] > max_samples:
                X = X[:max_samples]
            X = X.to(device=layer.base.weight.device, dtype=layer.base.weight.dtype)
            
            actual_k = min(k, X.shape[1], X.shape[0])
            if actual_k > 0:
                _, _, vh = torch.linalg.svd(X, full_matrices=False)
                # Leading k input directions are columns of vh[:k].T: shape (d_in, k)
                basis = vh[:actual_k].T
                layer.set_protected_subspace(basis)
                results[name] = layer.Q.shape[1]
            else:
                layer.set_protected_subspace(None)
                results[name] = 0
                
        return results

    def set_subspace(self, layer_name: str, basis: Optional[torch.Tensor]):
        """Sets the protected basis Q for a specific named layer."""
        if layer_name not in self.plastic_layer_map:
            raise KeyError(f"Layer '{layer_name}' is not an active plastic layer.")
        self.plastic_layer_map[layer_name].set_protected_subspace(basis)

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
        """Saves dynamic fast-weight buffers (U, V) and protected bases (Q)."""
        memory_state = {
            name: {
                "U": layer.U.cpu(),
                "V": layer.V.cpu(),
                "Q": layer.Q.cpu()
            }
            for name, layer in self.plastic_layer_map.items()
        }
        torch.save(memory_state, filepath)

    def load_synaptic_memory(self, filepath: str):
        """Loads previously saved synaptic memory state."""
        memory_state = torch.load(filepath, weights_only=True)
        for name, layer in self.plastic_layer_map.items():
            if name in memory_state:
                layer.U.copy_(memory_state[name]["U"].to(layer.U.device))
                layer.V.copy_(memory_state[name]["V"].to(layer.V.device))
                if "Q" in memory_state[name]:
                    layer.Q = memory_state[name]["Q"].to(layer.Q.device)
