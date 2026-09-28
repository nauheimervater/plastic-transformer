"""
Analysis script for Section 6.4 of the paper: Drift Mechanics and Subspace Structure.

Investigates:
1. Participation Ratio PR = (sum v_i^2)^2 / sum v_i^4 of v_1 and mean(X) across layers.
2. Ratio ||x_mean||^2 / lambda_1(Cov) and collinearity |cos(v_1, x_mean)|.
3. Direct causal test: Q = mean(X) / ||mean(X)|| vs. uncentered SVD Q = v_1 vs. k=0.
4. Coordinate specificity controls: Top-5 outlier channels, Median-5 channels,
   Random-5 channels, and Masked v_1 (Top-5 channels zeroed and renormalized).
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.gradient_targets_drift import (
    PlasticSetup,
    build_sessions,
    learn_fact,
    drift,
    ids_of,
    CONTROL_TEXTS,
)


def compute_participation_ratio(vec):
    """PR = (sum v_i^2)^2 / sum v_i^4. For unit vectors, PR = 1 / sum v_i^4."""
    norm_sq = vec.pow(2).sum()
    sum_fourth = vec.pow(4).sum()
    return ((norm_sq ** 2) / (sum_fourth + 1e-12)).item()


def run_mechanics_analysis(args):
    print(f"# Analyzing Drift Mechanics on {args.model} (targets={args.targets})")
    device = args.device

    texts = CONTROL_TEXTS
    if args.control_file and Path(args.control_file).exists():
        texts = [b.strip() for b in Path(args.control_file).read_text(encoding="utf-8").split("\n\n") if b.strip()]
    calib_texts, heldout_texts = texts[0::2], texts[1::2]

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    model.requires_grad_(False)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    setup = PlasticSetup(model, targets, None, rank=8, max_mass=10.0)

    # 1. Collect calibration activations
    calib_X = {}
    setup.capture_on = True
    for t in calib_texts[:args.n_calib_texts]:
        ids = ids_of(tok, t, device)
        setup.clear_capture()
        model(input_ids=ids)
        for name, layer in setup.adapters.items():
            x = setup.X[name].reshape(-1, layer.base.in_features)
            if name not in calib_X:
                calib_X[name] = []
            calib_X[name].append(x)
    setup.capture_on = False
    setup.clear_capture()
    calib_X = {k: torch.cat(v).float() for k, v in calib_X.items()}

    # 2. Structural Layer-by-Layer Diagnostics
    print("\n## 1. Subspace Structure and Participation Ratio per Layer\n")
    layer_diagnostics = []
    rng_rand = random.Random(42)

    Q_maps = {
        "unprotected (k=0)": {},
        "svd_v1": {},
        "mean_direction": {},
        "coord_top5": {},
        "coord_median5": {},
        "coord_random5": {},
        "svd_v1_masked_top5": {},
    }

    print("| Layer | d_in | PR(v1) | PR(x̄) | ||x̄||² | λ₁(Cov) | ||x̄||²/λ₁ | |cos(v1, x̄)| | Top-1 Ch (%) | Top-5 Ch (%) |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for idx, (name, layer) in enumerate(setup.adapters.items()):
        X = calib_X[name]
        d_in = layer.base.in_features
        N = X.shape[0]

        mean_x = X.mean(dim=0)
        norm_mean_sq = mean_x.pow(2).sum().item()
        u_mean = mean_x / (mean_x.norm() + 1e-12)

        Xc = X - mean_x.unsqueeze(0)
        _, sc, _ = torch.linalg.svd(Xc, full_matrices=False)
        lambda1_cov = (sc[0].item() ** 2) / N

        _, _, vh = torch.linalg.svd(X, full_matrices=False)
        v1 = vh[0]

        pr_v1 = compute_participation_ratio(v1)
        pr_mean = compute_participation_ratio(u_mean)
        cos_sim = abs(F.cosine_similarity(v1.unsqueeze(0), mean_x.unsqueeze(0)).item())
        ratio_mean_var = norm_mean_sq / (lambda1_cov + 1e-12)

        v1_sq = v1 ** 2
        top5_v1_energy = torch.topk(v1_sq, 5).values.sum().item() * 100
        top1_v1_energy = torch.max(v1_sq).item() * 100

        diag = {
            "layer": idx,
            "name": name,
            "d_in": d_in,
            "pr_v1": pr_v1,
            "pr_mean": pr_mean,
            "norm_mean_sq": norm_mean_sq,
            "lambda1_cov": lambda1_cov,
            "ratio_mean_var": ratio_mean_var,
            "cos_sim": cos_sim,
            "top1_v1_pct": top1_v1_energy,
            "top5_v1_pct": top5_v1_energy,
        }
        layer_diagnostics.append(diag)

        if idx in (0, 2, 6, 10, 12, 16, 18, 22, 23) or idx == len(setup.adapters) - 1:
            print(f"| {idx:2d} | {d_in} | {pr_v1:6.1f} | {pr_mean:6.1f} | {norm_mean_sq:8.1f} | {lambda1_cov:8.1f} | "
                  f"{ratio_mean_var:6.2f}x | {cos_sim:.4f} | {top1_v1_energy:5.1f}% | {top5_v1_energy:5.1f}% |")

        # Prepare Q bases
        Q_maps["svd_v1"][name] = v1.unsqueeze(1)
        Q_maps["mean_direction"][name] = u_mean.unsqueeze(1)

        # Coordinate channels based on mean-squared magnitude
        ch_energy = (X ** 2).mean(dim=0)
        sorted_indices = torch.argsort(ch_energy, descending=True)
        top5_idx = sorted_indices[:5]
        med_center = d_in // 2
        median5_idx = sorted_indices[med_center - 2: med_center + 3]
        rand5_idx = torch.tensor(rng_rand.sample(range(d_in), 5))

        e_top5 = torch.zeros(d_in, 5)
        for i, ch in enumerate(top5_idx):
            e_top5[ch, i] = 1.0
        Q_maps["coord_top5"][name] = e_top5

        e_med5 = torch.zeros(d_in, 5)
        for i, ch in enumerate(median5_idx):
            e_med5[ch, i] = 1.0
        Q_maps["coord_median5"][name] = e_med5

        e_rand5 = torch.zeros(d_in, 5)
        for i, ch in enumerate(rand5_idx):
            e_rand5[ch, i] = 1.0
        Q_maps["coord_random5"][name] = e_rand5

        v1_masked = v1.clone()
        v1_masked[top5_idx] = 0.0
        v1_masked = v1_masked / (v1_masked.norm() + 1e-12)
        Q_maps["svd_v1_masked_top5"][name] = v1_masked.unsqueeze(1)

    # 3. Causal Test and Specificity Controls across Seeds
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"\n## 2. Causal Test & Specificity Controls (seeds={seeds}, facts={args.n_facts}, epochs={args.epochs})\n")

    results = {cond: [] for cond in Q_maps}
    for seed in seeds:
        sessions = build_sessions(seed, args.n_facts, 1, template_shift=False)
        session = sessions[0]
        for cond_name, q_map in Q_maps.items():
            setup.reset()
            setup.set_gate(1.0)
            if cond_name != "unprotected (k=0)":
                for name, layer in setup.adapters.items():
                    layer.set_protected_subspace(q_map[name])

            random.seed(seed)
            torch.manual_seed(seed)
            for _ in range(args.epochs):
                order = session[:]
                random.shuffle(order)
                for fact in order:
                    learn_fact(model, tok, setup, fact, args.lr, device)

            d_held = drift(model, tok, setup, heldout_texts[:args.n_eval_texts], device)
            results[cond_name].append(d_held["kl"])
            print(f"  seed={seed} | {cond_name:<22} | KL_heldout = {d_held['kl']:.3e}")

    setup.remove()

    # Summary table
    base_kl = sum(results["unprotected (k=0)"]) / len(seeds)
    print("\n### Specificity Summary (mean over seeds)\n")
    print("| Condition | Basis Dim | Mean KL held-out | KL Reduction vs. k=0 | Interpretation |")
    print("|---|---:|---:|---:|---|")
    notes = {
        "unprotected (k=0)": "Baseline drift",
        "svd_v1": "Uncentered dominant SVD vector",
        "mean_direction": "Pure normalized mean activation vector",
        "coord_top5": "Top-5 outlier coordinate channels",
        "coord_median5": "5 median-energy coordinate channels",
        "coord_random5": "5 randomly sampled coordinate channels",
        "svd_v1_masked_top5": "SVD v1 with top-5 channels zeroed out",
    }
    summary_rows = []
    for cond_name, kls in results.items():
        m_kl = sum(kls) / len(kls)
        red = (1.0 - m_kl / base_kl) * 100
        dim_k = 0 if "k=0" in cond_name else (5 if "5" in cond_name else 1)
        row = {
            "condition": cond_name,
            "dim": dim_k,
            "mean_kl": m_kl,
            "reduction_pct": red,
            "note": notes.get(cond_name, ""),
        }
        summary_rows.append(row)
        print(f"| {cond_name:<22} | {dim_k:2d} | {m_kl:.3e} | {red:+6.1f} % | {notes.get(cond_name, '')} |")

    # Save to JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer_diagnostics": layer_diagnostics,
        "causal_test_summary": summary_rows,
        "raw_kl_per_seed": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved analysis results to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--targets", default="down_proj")
    ap.add_argument("--control-file", default="experiments/control.txt")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-facts", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-calib-texts", type=int, default=5)
    ap.add_argument("--n-eval-texts", type=int, default=4)
    ap.add_argument("--out", default="experiments/results/drift_mechanics_results.json")
    args = ap.parse_args()
    run_mechanics_analysis(args)


if __name__ == "__main__":
    main()
