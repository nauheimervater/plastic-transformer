"""
Summarize the confirmatory evaluation (seeds 1-5) and the drift-mechanics analysis from the
raw JSON files in experiments/results/. All numbers in the paper and READMEs come from here.

    python experiments/summarize_results.py

Author: Thomas Nauheimer (2026)
"""

import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RES = Path(__file__).resolve().parent / "results"
T95 = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

CONFIRM = [
    ("confirm_plastic_lr1.json", "online:eps=0.97:base_eps=0.9", "Ours, lr = 1.0 (selected by protocol)"),
    ("confirm_plastic_lr03_sensitivity.json", "online:eps=0.97:base_eps=0.9", "Ours, lr = 0.3 (pre-announced sensitivity)"),
    ("confirm_lora.json", "lora:replay", "B5 LoRA + replay"),
    ("confirm_lora.json", "lora:gpm:eps=0.97:base_eps=0.9", "B4 GPM-LoRA (Adam)"),
    ("confirm_olora.json", "lora:olora:lam=512.0", "B3 O-LoRA (λ = 512)"),
    ("confirm_lora.json", "lora:multi", "B2 LoRA per session, frozen"),
    ("confirm_lora.json", "lora:seq", "B1 sequential LoRA"),
    ("confirm_plastic_lr1.json", "rag:k=4", "B6 retrieval, top-4"),
    ("confirm_plastic_lr1.json", "icl:all", "B6′ all facts in context"),
    ("explore_gpm_lora_sgd.json", "lora:gpm:eps=0.97:base_eps=0.9", "B4 GPM-LoRA (SGD, exploratory)"),
]


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T95.get(n - 1, 1.96) * sd / math.sqrt(n)


def fmt(xs, spec):
    m, ci = mean_ci(xs)
    return f"{m:{spec}} ± {ci:{spec.lstrip('+')}}"


def runs(file, condition):
    data = json.loads((RES / file).read_text(encoding="utf-8"))
    return sorted((r for r in data["runs"] if r["condition"] == condition), key=lambda r: r["seed"])


def confirmatory_table():
    print("| Method | n | Recall | Paraphrase | Forget S1 (nats) | Ret. S1 | KL held-out | Δppl | MB/fact | Passes |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for file, cond, label in CONFIRM:
        rs = runs(file, cond)
        g = lambda k: [r[k] for r in rs]
        print(f"| {label} | {len(rs)} | {fmt(g('recall'), '.2f')} | {fmt(g('paraphrase'), '.2f')} | "
              f"{fmt(g('forget_first'), '.2f')} | {fmt(g('retention_first_exact'), '.2f')} | "
              f"{fmt(g('kl_heldout'), '.3f')} | {fmt([r['ppl_adapted'] - r['ppl_base'] for r in rs], '+.2f')} | "
              f"{mean_ci(g('state_mb_per_fact'))[0]:.4f} | {mean_ci(g('train_passes'))[0]:.0f} |")


def per_seed_table():
    print("| Seed | Recall | Paraphrase | Forget S1 | KL held-out | Δppl |")
    print("|---:|---:|---:|---:|---:|---:|")
    for r in runs(*CONFIRM[0][:2]):
        print(f"| {r['seed']} | {r['recall']:.2f} | {r['paraphrase']:.2f} | {r['forget_first']:.2f} | "
              f"{r['kl_heldout']:.3f} | {r['ppl_adapted'] - r['ppl_base']:+.2f} |")


def mechanics_table():
    data = json.loads((RES / "drift_mechanics_results.json").read_text(encoding="utf-8"))
    raw = data["raw_kl_per_seed"]
    base = raw["unprotected (k=0)"]
    print("| Protected basis | Per-seed KL reduction (%) | Mean ± 95 % CI (%) |")
    print("|---|---|---:|")
    for name, kls in raw.items():
        if name == "unprotected (k=0)":
            continue
        red = [100 * (1 - k / b) for k, b in zip(kls, base)]
        m, ci = mean_ci(red)
        print(f"| {name} | {', '.join(f'{x:.0f}' for x in red)} | {m:.0f} ± {ci:.0f} |")
    print()
    print("| Layer | ‖x̄‖²/λ₁ | cos(v₁, x̄) | PR(v₁) | top-5 share of v₁ (%) |")
    print("|---:|---:|---:|---:|---:|")
    for layer in data["layer_diagnostics"]:
        print(f"| {layer['layer']} | {layer['ratio_mean_var']:.2f} | {layer['cos_sim']:.3f} | "
              f"{layer['pr_v1']:.1f} | {layer['top5_v1_pct']:.1f} |")


if __name__ == "__main__":
    print("## Confirmatory evaluation (seeds 1-5, mean ± 95 % CI, t-distribution)\n")
    confirmatory_table()
    print("\n## Ours, lr = 1.0, per seed\n")
    per_seed_table()
    print("\n## Drift mechanics (12 facts, seeds 0-2; reductions paired per seed)\n")
    mechanics_table()
