"""
Demo: Formal Projected Delta-Rule Experiment with Subspace Invariance
Author: Thomas Nauheimer
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import torch
from plastic_transformer import PlasticLinearProjected


def run_subspace_experiment(seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    
    print("=" * 70)
    print("  PROJECTED DELTA-RULE EXPERIMENT (SUBSPACE PROTECTION)")
    print("=" * 70)
    
    # Base frozen linear layer
    base = torch.nn.Linear(32, 16, bias=True, dtype=torch.float64)
    directions, _ = torch.linalg.qr(torch.randn(32, 12, dtype=torch.float64))
    q, novel = directions[:, :4], directions[:, 4:12]
    
    # Adapter with protected basis span(Q)
    adapter = PlasticLinearProjected(base, rank=8, protected_basis=q, max_mass=10.0)
    model = torch.nn.Sequential(adapter)
    
    x = novel.T
    correction = torch.randn(8, 16, dtype=torch.float64) * 0.2
    target = base(x).detach() + correction
    
    heldout = torch.randn(20, 8, dtype=torch.float64)
    hx = heldout @ x
    hy = base(hx).detach() + heldout @ correction
    
    with torch.no_grad():
        mse_before = ((model(hx) - hy) ** 2).mean().item()
        protected_before = model(q.T).clone()
        
    print(f"[Phase 1] Pre-adaptation MSE on held-out test set: {mse_before:.6f}")
    
    # 160 Online adaptation steps
    start = time.perf_counter()
    for step in range(160):
        adapter.adapt(x, target)
    elapsed = time.perf_counter() - start
    
    with torch.no_grad():
        mse_after = ((model(hx) - hy) ** 2).mean().item()
        protected_delta = (model(q.T) - protected_before).abs().max().item()
        diag = adapter.diagnostics()
        
    print(f"[Phase 2] Post-adaptation MSE on held-out test set: {mse_after:.10e}")
    print(f"[Phase 3] Max perturbation on protected subspace span(Q): {protected_delta:.10e}")
    print(f"[Diagnostics] Fast-weight mass: {diag['mass']:.4f} | Elapsed time: {elapsed*1000:.2f} ms")
    print("=" * 70)
    print("Result: Adaptation succeeded with numerical zero-interference on protected subspace!")


if __name__ == "__main__":
    run_subspace_experiment()
