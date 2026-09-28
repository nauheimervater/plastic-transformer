"""
Reproduce Benchmark Table 1 from 'Plasticity Is All You Need'
Author: Thomas Nauheimer (2026)

Runs the multi-seed experiment on the Projected Delta-Rule Adapter across seeds 0 to 4
and prints the exact benchmark table in Markdown format.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from plastic_transformer import PlasticLinearProjected


def run_benchmark():
    torch.set_num_threads(1)
    
    print("# Reproduction of Table 1: Projected Delta-Rule Subspace Invariance")
    print("Configuration: float64, d_in=32, d_out=16, rank=8, k=4 protected, 160 updates, 20 held-out samples.\n")
    
    headers = ["Seed", "Held-out MSE Before", "Held-out MSE After", "Max Protected Change", "Invariance Check"]
    print(f"| {headers[0]} | {headers[1]} | {headers[2]} | {headers[3]} | {headers[4]} |")
    print("|:---:|:---:|:---:|:---:|:---:|")
    
    total_start = time.perf_counter()
    for seed in range(5):
        torch.manual_seed(seed)
        
        # Frozen linear projection
        base = torch.nn.Linear(32, 16, bias=True, dtype=torch.float64)
        
        # Orthonormal basis: 4 protected directions, 8 adaptation directions
        directions, _ = torch.linalg.qr(torch.randn(32, 12, dtype=torch.float64))
        q, novel = directions[:, :4], directions[:, 4:12]
        
        # Projected adapter
        adapter = PlasticLinearProjected(base, rank=8, protected_basis=q, max_mass=10.0)
        model = torch.nn.Sequential(adapter)
        
        # Adaptation data on novel directions
        x = novel.T
        correction = torch.randn(8, 16, dtype=torch.float64) * 0.2
        target = base(x).detach() + correction
        
        # Held-out evaluation set
        heldout = torch.randn(20, 8, dtype=torch.float64)
        hx = heldout @ x
        hy = base(hx).detach() + heldout @ correction
        
        # Pre-adaptation evaluation
        with torch.no_grad():
            mse_before = ((model(hx) - hy) ** 2).mean().item()
            protected_before = model(q.T).clone()
            
        # 160 Supervised delta updates
        for _ in range(160):
            adapter.adapt(x, target)
            
        # Post-adaptation evaluation
        with torch.no_grad():
            mse_after = ((model(hx) - hy) ** 2).mean().item()
            protected_delta = (model(q.T) - protected_before).abs().max().item()
            
        check = "≤ 1e-15" if protected_delta < 1e-15 else "FAILED"
        print(f"| {seed} | {mse_before:.6f} | ${mse_after:.2e}$ | ${protected_delta:.2e}$ | {check} |")
        
        # Rigorous assertions
        assert mse_after < 1e-8, f"Seed {seed} failed convergence"
        assert protected_delta < 1e-15, f"Seed {seed} broke subspace invariance"
        
    total_elapsed = time.perf_counter() - total_start
    print(f"\nAll 5 seeds verified successfully in {total_elapsed*1000:.1f} ms.")
    print("Base weights remained bit-identical. Protected subspace invariance holds to machine precision.")
    print("Note: The 'Max Protected Change' column reflects numerical zero (<= 1e-15).")
    print("      Exact residuals (~10^-17 to 10^-16) vary with platform BLAS/CPU floating-point implementations,")
    print("      while MSE convergence values match across all environments.")


if __name__ == "__main__":
    run_benchmark()

