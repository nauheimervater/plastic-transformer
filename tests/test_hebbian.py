import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from plastic_transformer import PlasticLinearHebbian, PlasticModelWrapper


class TestPlasticLinearHebbian(unittest.TestCase):
    def test_initialization_zero(self):
        torch.manual_seed(42)
        layer = nn.Linear(16, 8)
        plastic = PlasticLinearHebbian.from_linear(layer, rank=4)
        
        x = torch.randn(2, 16)
        with torch.no_grad():
            y_base = layer(x)
            y_plastic = plastic(x)
            
        torch.testing.assert_close(y_base, y_plastic)

    def test_hebbian_step_adapts(self):
        torch.manual_seed(42)
        plastic = PlasticLinearHebbian(16, 8, rank=4, eta=0.1)
        x = torch.randn(2, 16)
        
        # Forward pass caches activations
        _ = plastic(x)
        self.assertEqual(plastic.U.abs().sum().item(), 0.0)
        
        # Synaptic update
        plastic.hebbian_step(surprise_signal=1.0)
        self.assertGreater(plastic.U.abs().sum().item(), 0.0)
        self.assertGreater(plastic.V.abs().sum().item(), 0.0)

    def test_reset(self):
        plastic = PlasticLinearHebbian(16, 8, rank=4)
        x = torch.randn(2, 16)
        _ = plastic(x)
        plastic.hebbian_step(1.0)
        self.assertGreater(plastic.U.abs().sum().item(), 0.0)
        
        plastic.reset_plasticity()
        self.assertEqual(plastic.U.abs().sum().item(), 0.0)
        self.assertEqual(plastic.V.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
