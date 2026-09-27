import io
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch import nn
from plastic_transformer import PlasticLinearProjected


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


if __name__ == "__main__":
    unittest.main()
