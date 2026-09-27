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

    def _collect_activations(
        self,
        inputs: Union[torch.Tensor, List[Any]],
        max_samples: int = 1024
    ) -> Dict[str, torch.Tensor]:
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
            if isinstance(inputs, torch.Tensor):
                items = [inputs]
            elif isinstance(inputs, (list, tuple)):
                items = list(inputs)
            else:
                items = [inputs]
                
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
            
        collected_tensors = {}
        for name, layer in self.plastic_layer_map.items():
            tensors = layer_inputs[name]
            if not tensors:
                continue
            X = torch.cat(tensors, dim=0)
            if X.shape[0] > max_samples:
                # Random sampling rather than deterministic truncation
                perm = torch.randperm(X.shape[0])[:max_samples]
                X = X[perm]
            collected_tensors[name] = X.to(
                device=layer.base.weight.device, 
                dtype=layer.base.weight.dtype
            )
        return collected_tensors

    @torch.no_grad()
    def calibrate_subspace(
        self,
        calibration_inputs: Union[torch.Tensor, List[Any]],
        k: Optional[int] = 8,
        energy_threshold: Optional[float] = None,
        max_samples: int = 1024
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calibrate the protected subspace span(Q) across all plastic layers by recording
        layer-wise input activations across forward passes of calibration_inputs.
        
        Supports:
        - Fixed dimension k (top-k singular vectors).
        - Energy threshold (GPM style: cumulative variance sum(s_i^2)/sum(s^2) >= energy_threshold).
        
        Returns:
            Dict mapping layer name to calibration metrics (rank, energy_explained).
        """
        collected = self._collect_activations(calibration_inputs, max_samples=max_samples)
        results = {}
        
        for name, layer in self.plastic_layer_map.items():
            if name not in collected:
                results[name] = {"rank": 0, "energy_explained": 0.0}
                continue
            X = collected[name]
            
            _, s, vh = torch.linalg.svd(X, full_matrices=False)
            total_energy = (s ** 2).sum()
            cum_energy = torch.cumsum(s ** 2, dim=0) / total_energy.clamp_min(1e-12)
            
            if energy_threshold is not None:
                # GPM-style energy criterion
                meets_thresh = (cum_energy >= energy_threshold).nonzero()
                actual_k = meets_thresh[0].item() + 1 if len(meets_thresh) > 0 else len(s)
                if k is not None:
                    actual_k = min(actual_k, k)
            else:
                actual_k = min(k if k is not None else 8, X.shape[1], X.shape[0])
                
            actual_k = min(actual_k, vh.shape[0])
            if actual_k > 0:
                basis = vh[:actual_k].T
                layer.set_protected_subspace(basis)
                energy_exp = cum_energy[actual_k - 1].item()
                results[name] = {"rank": actual_k, "energy_explained": energy_exp}
            else:
                layer.set_protected_subspace(None)
                results[name] = {"rank": 0, "energy_explained": 0.0}
                
        return results

    @torch.no_grad()
    def expand_subspace(
        self,
        new_inputs: Union[torch.Tensor, List[Any]],
        k_max_new: int = 4,
        energy_threshold: float = 0.95,
        max_samples: int = 1024
    ) -> Dict[str, int]:
        """
        Continual Learning: Sequentially expand existing protected subspace Q with
        novel activation directions observed in new_inputs.
        
        Projects new activations onto the orthogonal complement (I - Q Q^T), extracts
        dominant residual directions via SVD, and appends them to Q.
        """
        collected = self._collect_activations(new_inputs, max_samples=max_samples)
        results = {}
        
        for name, layer in self.plastic_layer_map.items():
            if name not in collected:
                results[name] = layer.Q.shape[1]
                continue
            X = collected[name]
            
            if layer.Q.shape[1] == 0:
                # No previous basis, calibrate directly
                _, s, vh = torch.linalg.svd(X, full_matrices=False)
                k_sel = min(k_max_new, vh.shape[0])
                layer.set_protected_subspace(vh[:k_sel].T)
                results[name] = layer.Q.shape[1]
                continue
                
            # Project new activations onto orthogonal complement of existing Q
            X_perp = layer.project(X)
            residual_energy = (X_perp ** 2).sum() / (X ** 2).sum().clamp_min(1e-12)
            
            if residual_energy > 0.01: # Has significant new orthogonal energy
                _, s_perp, vh_perp = torch.linalg.svd(X_perp, full_matrices=False)
                k_new = min(k_max_new, vh_perp.shape[0])
                new_dirs = vh_perp[:k_new].T
                
                # Combine and re-orthonormalize
                combined = torch.cat([layer.Q, new_dirs], dim=1)
                layer.set_protected_subspace(combined)
                
            results[name] = layer.Q.shape[1]
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
