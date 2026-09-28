import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch import nn

from plastic_transformer import PlasticLinearProjected, PlasticModelWrapper

f64 = torch.float64


class TestPlasticLinearProjected(unittest.TestCase):
    def test_dense_reference(self):
        torch.manual_seed(9)
        layer = PlasticLinearProjected(nn.Linear(7, 5, dtype=f64), rank=5, max_mass=100.0)
        x = torch.randn(3, 7, dtype=f64)
        target = torch.randn(3, 5, dtype=f64)
        expected = 0.5 * (target - layer(x)).T @ x / 3
        layer.adapt(x, target)
        torch.testing.assert_close(layer.U @ layer.V.T, expected)

    def test_subspace_invariance(self):
        torch.manual_seed(42)
        q, _ = torch.linalg.qr(torch.randn(16, 4, dtype=f64))
        adapter = PlasticLinearProjected(nn.Linear(16, 8, dtype=f64), rank=4, protected_basis=q)
        with torch.no_grad():
            before = adapter(q.T).clone()
            for _ in range(20):
                adapter.adapt(torch.randn(5, 16, dtype=f64), torch.randn(5, 8, dtype=f64))
            diff = (adapter(q.T) - before).abs().max().item()
        self.assertLess(diff, 1e-14)

    def test_set_subspace_refuses_to_erase_learned_state(self):
        torch.manual_seed(99)
        adapter = PlasticLinearProjected(nn.Linear(16, 8, dtype=f64), rank=4)
        with torch.no_grad():
            adapter.adapt(torch.randn(5, 16, dtype=f64), torch.randn(5, 8, dtype=f64))
        q, _ = torch.linalg.qr(torch.randn(16, 3, dtype=f64))
        with self.assertRaises(RuntimeError):
            adapter.set_protected_subspace(q)
        adapter.set_protected_subspace(q, erase_active=True)  # explicit opt-in
        self.assertLess((adapter.U @ (adapter.V.T @ adapter.Q)).abs().max().item(), 1e-14)

    def test_consolidate_then_expand_preserves_learned_mapping(self):
        """Learn on x_a, consolidate, protect span(x_a), learn something else: outputs on x_a stay."""
        torch.manual_seed(7)
        adapter = PlasticLinearProjected(nn.Linear(16, 8, dtype=f64), rank=4, max_mass=100.0)
        x_a = F.normalize(torch.randn(3, 16, dtype=f64), dim=1)  # delta rule is stable for lr*||x||^2 < 2
        t_a = torch.randn(3, 8, dtype=f64)
        with torch.no_grad():
            for _ in range(200):
                adapter.adapt(x_a, t_a)
            learned = adapter(x_a).clone()
            self.assertLess(((learned - t_a) ** 2).mean().item(), 1e-3)
            adapter.consolidate()
            torch.testing.assert_close(adapter(x_a), learned)  # consolidation keeps the function
            adapter.set_protected_subspace(x_a.T)
            for _ in range(50):
                adapter.adapt(F.normalize(torch.randn(4, 16, dtype=f64), dim=1), torch.randn(4, 8, dtype=f64))
            self.assertLess((adapter(x_a) - learned).abs().max().item(), 1e-12)

    def test_without_consolidation_protection_erases(self):
        """Documents the failure mode that consolidate() exists for."""
        torch.manual_seed(7)
        adapter = PlasticLinearProjected(nn.Linear(16, 8, dtype=f64), rank=4, max_mass=100.0)
        x_a, t_a = F.normalize(torch.randn(3, 16, dtype=f64), dim=1), torch.randn(3, 8, dtype=f64)
        with torch.no_grad():
            for _ in range(200):
                adapter.adapt(x_a, t_a)
            learned_err = ((adapter(x_a) - t_a) ** 2).mean().item()
            adapter.set_protected_subspace(x_a.T, erase_active=True)
            erased_err = ((adapter(x_a) - t_a) ** 2).mean().item()
        self.assertGreater(erased_err, 100 * learned_err)

    def test_state_dict_roundtrip_with_variable_size_state(self):
        torch.manual_seed(123)
        adapter = PlasticLinearProjected(nn.Linear(8, 4, dtype=f64), rank=2)
        x = torch.randn(2, 8, dtype=f64)
        with torch.no_grad():
            adapter.adapt(x, torch.randn(2, 4, dtype=f64))
            adapter.consolidate()
            adapter.set_protected_subspace(x.T)
            adapter.adapt(torch.randn(3, 8, dtype=f64), torch.randn(3, 4, dtype=f64))
        stream = io.BytesIO()
        torch.save(adapter.state_dict(), stream)
        stream.seek(0)
        restored = PlasticLinearProjected(nn.Linear(8, 4, dtype=f64), rank=2)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        torch.testing.assert_close(adapter(x), restored(x))
        self.assertEqual(restored.Q.shape[1], 2)
        self.assertEqual(restored.U_frozen.shape[1], 2)


class TinyNet(nn.Module):
    def __init__(self, vocab=32, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d, dtype=f64)
        self.up_proj = nn.Linear(d, 32, dtype=f64)
        self.down_proj = nn.Linear(32, d, dtype=f64)
        self.head = nn.Linear(d, vocab, dtype=f64)

    def forward(self, ids):
        h = self.emb(ids)
        h = h + self.down_proj(torch.relu(self.up_proj(h)))
        return self.head(h)


def seq_loss(model, ids):
    logits = model(ids)
    return F.cross_entropy(logits[0, :-1], ids[0, 1:])


class TestPlasticModelWrapper(unittest.TestCase):
    def test_matching_is_by_exact_child_name(self):
        net = TinyNet()
        wrapper = PlasticModelWrapper(net, target_modules=["up_proj", "down_proj"], rank=4)
        self.assertEqual(set(wrapper.plastic_layer_map), {"up_proj", "down_proj"})
        with self.assertRaises(ValueError):
            PlasticModelWrapper(TinyNet(), target_modules=["proj"], rank=4)

    def test_calibration_invariance_and_gate(self):
        torch.manual_seed(42)
        net = TinyNet()
        ref = TinyNet()
        ref.load_state_dict(net.state_dict())
        wrapper = PlasticModelWrapper(net, target_modules=["up_proj", "down_proj"], rank=4)
        calib = [torch.randint(0, 32, (1, 12)) for _ in range(3)]
        wrapper.calibrate_subspace(calib, k=4)
        for layer in wrapper.plastic_layers:
            self.assertEqual(layer.Q.shape[1], 4)
        ids = torch.randint(0, 32, (1, 10))
        for _ in range(5):
            with wrapper.capture():
                loss = seq_loss(wrapper, ids)
            wrapper.gradient_step(loss, lr=0.5)
        for layer in wrapper.plastic_layers:
            with torch.no_grad():
                diff = (layer(layer.Q.T) - layer.base(layer.Q.T)).abs().max().item()
            self.assertLess(diff, 1e-12)
        wrapper.set_gate(0.0)
        with torch.no_grad():
            torch.testing.assert_close(wrapper(ids), ref(ids))

    def test_gradient_step_reduces_loss(self):
        torch.manual_seed(3)
        wrapper = PlasticModelWrapper(TinyNet(), target_modules=["up_proj", "down_proj"], rank=8, max_mass=100.0)
        ids = torch.randint(0, 32, (1, 12))
        with torch.no_grad():
            before = seq_loss(wrapper, ids).item()
        for _ in range(30):
            with wrapper.capture():
                loss = seq_loss(wrapper, ids)
            wrapper.gradient_step(loss, lr=0.3)
        with torch.no_grad():
            after = seq_loss(wrapper, ids).item()
        self.assertLess(after, before - 0.2)

    def test_expand_subspace_consolidates_and_grows(self):
        torch.manual_seed(101)
        wrapper = PlasticModelWrapper(TinyNet(), target_modules=["up_proj", "down_proj"], rank=4, max_mass=100.0)
        ids_a = torch.randint(0, 32, (1, 12))
        for _ in range(20):
            with wrapper.capture():
                loss = seq_loss(wrapper, ids_a)
            wrapper.gradient_step(loss, lr=0.3)
        dims = wrapper.expand_subspace([ids_a], energy_threshold=0.99)
        self.assertTrue(all(d > 0 for d in dims.values()))
        self.assertTrue(all(l.U_frozen.shape[1] == 4 and not l.has_active_state for l in wrapper.plastic_layers))

    def test_save_load_synaptic_memory(self):
        torch.manual_seed(5)
        wrapper = PlasticModelWrapper(TinyNet(), target_modules=["up_proj", "down_proj"], rank=4)
        ids = torch.randint(0, 32, (1, 10))
        for _ in range(3):
            with wrapper.capture():
                loss = seq_loss(wrapper, ids)
            wrapper.gradient_step(loss, lr=0.3)
        wrapper.expand_subspace([ids])
        with wrapper.capture():
            loss = seq_loss(wrapper, ids)
        wrapper.gradient_step(loss, lr=0.3)
        with torch.no_grad():
            out = wrapper(ids)
        path = os.path.join(tempfile.mkdtemp(), "mem.pt")
        wrapper.save_synaptic_memory(path)
        wrapper.reset_synapses()
        for layer in wrapper.plastic_layers:
            layer.set_protected_subspace(None)
        wrapper.load_synaptic_memory(path)
        with torch.no_grad():
            torch.testing.assert_close(wrapper(ids), out)


if __name__ == "__main__":
    unittest.main()
