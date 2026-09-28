"""
PlasticModelWrapper: inject PlasticLinearProjected layers into a PyTorch model.
Author: Thomas Nauheimer (2026)
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .projected_delta import PlasticLinearProjected

_FAST_STATE = ("U", "V", "Q", "U_frozen", "V_frozen")


class PlasticModelWrapper(nn.Module):
    """
    Replaces matching nn.Linear children of `base_model` by PlasticLinearProjected layers.

    Learning uses gradient-derived layer targets (test-time training of the fast weights):

        with plastic.capture():
            loss = loss_fn(plastic(**batch))
        plastic.gradient_step(loss, lr=0.3)

    `gradient_step` sets T = Y - lr * N / mean(||x||^2) * dL/dY per layer, so each layer's
    adapt() performs a projected, NLMS-normalized gradient step on its fast weights. This needs a
    backward pass through the (frozen) network; it is not a forward-only update.

    Continual use across sessions: `expand_subspace(session_inputs)` consolidates the active
    fast weights first and then grows Q, so later sessions cannot overwrite what was learned on
    the protected inputs of this layer.
    """

    DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]

    def __init__(
        self,
        base_model: nn.Module,
        target_modules: Optional[List[str]] = None,
        rank: int = 8,
        learning_rate: float = 1.0,
        decay: float = 0.0,
        max_mass: float = 10.0,
        gate: float = 1.0,
        protected_subspaces: Optional[Dict[str, torch.Tensor]] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.plastic_layers: List[PlasticLinearProjected] = []
        self.plastic_layer_map: Dict[str, PlasticLinearProjected] = {}
        targets = [t.lower() for t in (target_modules or self.DEFAULT_TARGETS)]
        protected_subspaces = protected_subspaces or {}

        candidates = []
        for parent_name, parent in base_model.named_modules():
            for name, child in parent.named_children():
                if isinstance(child, nn.Linear) and name.lower() in targets:
                    full = f"{parent_name}.{name}" if parent_name else name
                    candidates.append((parent, name, child, full))
        for parent, name, child, full in candidates:
            adapter = PlasticLinearProjected(
                base=child, rank=rank, protected_basis=protected_subspaces.get(full),
                learning_rate=learning_rate, decay=decay, max_mass=max_mass, gate=gate)
            setattr(parent, name, adapter)
            self.plastic_layers.append(adapter)
            self.plastic_layer_map[full] = adapter
        if not self.plastic_layers:
            raise ValueError(f"No nn.Linear children named {targets} found (matching is by exact child name)")

        self._capturing = False
        self._X: Dict[str, torch.Tensor] = {}
        self._Y: Dict[str, torch.Tensor] = {}
        for full, layer in self.plastic_layer_map.items():
            layer.register_forward_pre_hook(self._make_pre_hook(full))
            layer.register_forward_hook(self._make_post_hook(full))

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    # ------------------------------------------------------------------ learning

    def _make_pre_hook(self, name):
        def hook(mod, inp):
            if self._capturing:
                self._X[name] = inp[0]
        return hook

    def _make_post_hook(self, name):
        def hook(mod, inp, out):
            if not self._capturing:
                return None
            if not out.requires_grad:
                # Frozen base: nothing upstream requires grad. Start the graph here with a
                # non-leaf tensor so downstream in-place operations remain legal.
                out = out.detach().requires_grad_(True) * 1.0
            self._Y[name] = out
            return out
        return hook

    @contextmanager
    def capture(self):
        """Record inputs and outputs of all plastic layers during the enclosed forward pass."""
        self._X, self._Y = {}, {}
        self._capturing = True
        try:
            with torch.enable_grad():
                yield self
        finally:
            self._capturing = False

    def gradient_step(self, loss: torch.Tensor, lr: float) -> Dict[str, Any]:
        """Projected NLMS gradient step on every plastic layer captured in the last forward pass."""
        if not self._Y:
            raise RuntimeError("No captured activations; run the forward pass inside `with plastic.capture():`")
        names = list(self._Y)
        grads = torch.autograd.grad(loss, [self._Y[n] for n in names], allow_unused=True)
        updated = 0
        with torch.no_grad():
            for name, g in zip(names, grads):
                if g is None:
                    continue
                layer = self.plastic_layer_map[name]
                X = self._X[name].detach().reshape(-1, layer.base.in_features)
                Y = self._Y[name].detach().reshape(-1, layer.base.out_features)
                G = g.reshape(-1, layer.base.out_features)
                mask = G.norm(dim=1) > 0
                if not mask.any():
                    continue
                X, Y, G = X[mask], Y[mask], G[mask]
                step = lr * X.shape[0] / X.pow(2).sum(dim=1).mean().clamp_min(1e-12)
                layer.adapt(X, Y - step * G)
                updated += 1
        self._X, self._Y = {}, {}
        return {"loss": loss.item(), "layers_updated": updated}

    # ------------------------------------------------------------------ subspaces

    def _collect_activations(self, inputs: Union[torch.Tensor, List[Any]], max_samples: int) -> Dict[str, torch.Tensor]:
        rows: Dict[str, List[torch.Tensor]] = {name: [] for name in self.plastic_layer_map}
        hooks = []
        for name, layer in self.plastic_layer_map.items():
            def make_hook(layer_name):
                def hook(mod, inp):
                    x = inp[0].detach()
                    rows[layer_name].append(x.reshape(-1, x.shape[-1]).cpu())
                return hook
            hooks.append(layer.register_forward_pre_hook(make_hook(name)))
        items = [inputs] if isinstance(inputs, torch.Tensor) or not isinstance(inputs, (list, tuple)) else list(inputs)
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
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
        out = {}
        for name, layer in self.plastic_layer_map.items():
            if not rows[name]:
                continue
            X = torch.cat(rows[name])
            if X.shape[0] > max_samples:
                X = X[torch.randperm(X.shape[0])[:max_samples]]
            out[name] = X.to(device=layer.base.weight.device, dtype=layer.base.weight.dtype)
        return out

    @staticmethod
    def _energy_rank(s: torch.Tensor, needed_energy: torch.Tensor) -> int:
        if needed_energy <= 0:
            return 0
        return min(int((torch.cumsum(s ** 2, 0) < needed_energy).sum().item()) + 1, s.numel())

    @torch.no_grad()
    def calibrate_subspace(
        self,
        calibration_inputs: Union[torch.Tensor, List[Any]],
        k: Optional[int] = None,
        energy_threshold: Optional[float] = None,
        max_samples: int = 4096
    ) -> Dict[str, Dict[str, Any]]:
        """
        Set Q per layer from calibration activations (uncentered SVD of layer inputs).
        Use either a fixed k or a GPM energy threshold; with both, k caps the energy rank.
        Requires zero active fast weights (fresh model, or after consolidate()).
        """
        if k is None and energy_threshold is None:
            raise ValueError("Pass k and/or energy_threshold")
        collected = self._collect_activations(calibration_inputs, max_samples)
        results = {}
        for name, layer in self.plastic_layer_map.items():
            if name not in collected:
                continue
            X = collected[name]
            _, s, vh = torch.linalg.svd(X, full_matrices=False)
            if energy_threshold is not None:
                kk = self._energy_rank(s, energy_threshold * (s ** 2).sum())
                if k is not None:
                    kk = min(kk, k)
            else:
                kk = min(k, s.numel())
            layer.set_protected_subspace(vh[:kk].T if kk else None)
            results[name] = {"rank": kk, "energy_explained": ((s[:kk] ** 2).sum() / (s ** 2).sum()).item() if kk else 0.0}
        return results

    @torch.no_grad()
    def consolidate(self):
        """Freeze all active fast weights into the per-layer consolidated stores."""
        for layer in self.plastic_layers:
            layer.consolidate()

    @torch.no_grad()
    def expand_subspace(
        self,
        new_inputs: Union[torch.Tensor, List[Any]],
        energy_threshold: float = 0.97,
        max_new: Optional[int] = None,
        max_samples: int = 4096
    ) -> Dict[str, int]:
        """
        Continual learning step (GPM energy criterion): consolidate the active fast weights,
        then add residual directions of `new_inputs` (outside span(Q)) until span(Q) holds
        `energy_threshold` of their energy. Returns the new dim(Q) per layer.
        """
        self.consolidate()
        collected = self._collect_activations(new_inputs, max_samples)
        results = {}
        for name, layer in self.plastic_layer_map.items():
            if name not in collected:
                results[name] = layer.Q.shape[1]
                continue
            X = collected[name]
            total = (X ** 2).sum()
            R = layer.project(X)
            need = energy_threshold * total - (total - (R ** 2).sum())
            _, s, vh = torch.linalg.svd(R, full_matrices=False)
            kk = self._energy_rank(s, need)
            if max_new is not None:
                kk = min(kk, max_new)
            if kk:
                layer.set_protected_subspace(torch.cat([layer.Q, vh[:kk].T], dim=1))
            results[name] = layer.Q.shape[1]
        return results

    def set_subspace(self, layer_name: str, basis: Optional[torch.Tensor], erase_active: bool = False):
        if layer_name not in self.plastic_layer_map:
            raise KeyError(f"Layer '{layer_name}' is not a plastic layer.")
        self.plastic_layer_map[layer_name].set_protected_subspace(basis, erase_active=erase_active)

    # ------------------------------------------------------------------ utilities

    def reset_synapses(self, include_frozen: bool = True):
        for layer in self.plastic_layers:
            layer.reset(include_frozen=include_frozen)

    def set_gate(self, gate_value: float):
        """gate = 0 restores the exact base model; gate = 1 enables all fast weights."""
        for layer in self.plastic_layers:
            layer.gate.fill_(gate_value)

    def state_bytes(self, include_q: bool = False) -> int:
        return sum(l.state_numel(include_q) * l.U.element_size() for l in self.plastic_layers)

    def save_synaptic_memory(self, filepath: str):
        """Save active and consolidated fast weights and protected bases (not the base model)."""
        torch.save({name: {k: getattr(layer, k).cpu() for k in _FAST_STATE}
                    for name, layer in self.plastic_layer_map.items()}, filepath)

    def load_synaptic_memory(self, filepath: str):
        state = torch.load(filepath, weights_only=True)
        with torch.no_grad():
            for name, layer in self.plastic_layer_map.items():
                if name not in state:
                    continue
                for k in _FAST_STATE:
                    if k in state[name]:
                        setattr(layer, k, state[name][k].to(device=layer.U.device, dtype=layer.U.dtype).clone())
