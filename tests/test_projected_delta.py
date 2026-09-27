import io
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch import nn
from plastic_transformer import PlasticLinearProjected, PlasticModelWrapper


class TestPlasticLinearProjected(unittest.TestCase):
    def test_dense_reference(self):
        torch.manual_seed(9)
        layer = PlasticLinearProjected(nn.Linear(7, 5, dtype=torch.float64), rank=5, max_mass=100.0)
        x = torch.randn(3, 7, dtype=torch.float64)
        target = torch.randn(3, 5, dtype=torch.float64)
        
        expected = 0.5 * (target - layer(x)).T @ x / 3
        layer.adapt(x, target)
        torch.testing.assert_close(layer.U @ layer.V.T, expected)

    def test_subspace_invariance(self):
        torch.manual_seed(42)
        base = nn.Linear(16, 8, dtype=torch.float64)
        directions, _ = torch.linalg.qr(torch.randn(16, 4, dtype=torch.float64))
        q = directions
        
        adapter = PlasticLinearProjected(base, rank=4, protected_basis=q)
        
        # Test input inside protected subspace
        x_prot = q.T
        with torch.no_grad():
            y_prot_before = adapter(x_prot).clone()
            
        # Adapt on novel data
        x_novel = torch.randn(5, 16, dtype=torch.float64)
        target = torch.randn(5, 8, dtype=torch.float64)
        for _ in range(20):
            adapter.adapt(x_novel, target)
            
        with torch.no_grad():
            y_prot_after = adapter(x_prot)
            diff = (y_prot_after - y_prot_before).abs().max().item()
            
        # Perturbation on protected subspace must be machine epsilon level
        self.assertLess(diff, 1e-15)

    def test_state_dict_roundtrip(self):
        torch.manual_seed(123)
        base = nn.Linear(8, 4, dtype=torch.float64)
        adapter = PlasticLinearProjected(base, rank=2)
        x = torch.randn(2, 8, dtype=torch.float64)
        target = torch.randn(2, 4, dtype=torch.float64)
        adapter.adapt(x, target)
        
        stream = io.BytesIO()
        torch.save(adapter.state_dict(), stream)
        stream.seek(0)
        
        restored = PlasticLinearProjected(nn.Linear(8, 4, dtype=torch.float64), rank=2)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        
        torch.testing.assert_close(adapter(x), restored(x))


class TestPlasticModelWrapper(unittest.TestCase):
    def test_wrapper_calibration_and_invariance(self):
        torch.manual_seed(42)
        
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(16, 16, dtype=torch.float64)
                self.down_proj = nn.Linear(16, 8, dtype=torch.float64)
            def forward(self, x):
                return self.down_proj(torch.relu(self.q_proj(x)))
                
        base_net = SimpleNet()
        wrapper = PlasticModelWrapper(base_net, rank=4)
        self.assertEqual(len(wrapper.plastic_layers), 2)
        
        # Calibration inputs (e.g. baseline task)
        calib_data = [torch.randn(10, 16, dtype=torch.float64) for _ in range(3)]
        wrapper.calibrate_subspace(calib_data, k=4)
        
        for layer in wrapper.plastic_layers:
            self.assertEqual(layer.Q.shape[1], 4)
            
        # Adapt on orthogonal inputs
        for name, layer in wrapper.plastic_layer_map.items():
            P = torch.eye(layer.base.in_features, dtype=torch.float64) - layer.Q @ layer.Q.T
            novel_x = (P @ torch.randn(layer.base.in_features, 3, dtype=torch.float64)).T
            target = layer.base(novel_x) + 0.1 * torch.randn(3, layer.base.out_features, dtype=torch.float64)
            layer.adapt(novel_x, target)
            
            # Check invariance along Q
            q_in = layer.Q.T
            out_before = layer.base(q_in)
            out_after = layer(q_in)
            diff = (out_after - out_before).abs().max().item()
            self.assertLess(diff, 1e-15)
            
        # Test gate = 0.0
        wrapper.set_gate(0.0)
        test_in = torch.randn(4, 16, dtype=torch.float64)
        out_frozen = base_net(test_in)
        out_gated = wrapper(test_in)
        torch.testing.assert_close(out_frozen, out_gated)

    def test_post_adaptation_subspace_protection(self):
        torch.manual_seed(99)
        base = nn.Linear(16, 8, dtype=torch.float64)
        adapter = PlasticLinearProjected(base, rank=4)
        
        # Adapt without Q first
        x = torch.randn(5, 16, dtype=torch.float64)
        target = torch.randn(5, 8, dtype=torch.float64)
        adapter.adapt(x, target)
        
        # Now set Q post-adaptation
        new_q, _ = torch.linalg.qr(torch.randn(16, 3, dtype=torch.float64))
        adapter.set_protected_subspace(new_q)
        
        # Invariance AQ = 0 must hold immediately
        fast_out = adapter.U @ (adapter.V.T @ new_q)
        self.assertLess(fast_out.abs().max().item(), 1e-15)

    def test_expand_subspace_continual(self):
        torch.manual_seed(101)
        base = nn.Sequential(nn.Linear(12, 12, dtype=torch.float64))
        wrapper = PlasticModelWrapper(base, target_modules=["0"], rank=4)
        
        # Initial calibration
        in1 = torch.randn(10, 12, dtype=torch.float64)
        wrapper.calibrate_subspace(in1, k=2)
        initial_dim = wrapper.plastic_layers[0].Q.shape[1]
        self.assertEqual(initial_dim, 2)
        
        # Expand subspace with new task inputs
        in2 = torch.randn(10, 12, dtype=torch.float64)
        wrapper.expand_subspace(in2, k_max_new=2)
        expanded_dim = wrapper.plastic_layers[0].Q.shape[1]
        self.assertGreater(expanded_dim, initial_dim)


if __name__ == "__main__":
    unittest.main()


