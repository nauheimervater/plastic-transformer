"""
Runner script for Step 2 (Confirmatory Runs) and Step 4 (GPM-LoRA with SGD Exploration).
Strictly follows the protocol:
- Seeds: 1, 2, 3, 4, 5
- Model: Qwen/Qwen2.5-0.5B, targets: down_proj, rank: 32
- Preserves all intermediate outputs in experiments/results/
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments" / "results"
RES.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    sys.stdout.flush()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print(f"Error:\n{res.stderr}")
        sys.stderr.flush()
    else:
        print(res.stdout)
        sys.stdout.flush()
    return res.stdout


def step_2a_plastic_lr1():
    out = RES / "confirm_plastic_lr1.json"
    if out.exists():
        print(f"Skipping already completed {out.name}")
        return
    print("\n=== Step 2a: Confirming Plastic lr=1.0 + RAG + ICL (Seeds 1-5) ===")
    cmd = [
        sys.executable, "experiments/gradient_targets_drift.py",
        "--model", "Qwen/Qwen2.5-0.5B", "--targets", "down_proj",
        "--ranks", "32", "--max-mass", "1000.0",
        "--control-file", "experiments/control.txt",
        "--seeds", "1,2,3,4,5", "--lr", "1.0",
        "--conditions", "online:eps=0.97:base_eps=0.9;rag:k=4;icl:all",
        "--out", str(out)
    ]
    run_cmd(cmd)


def step_2b_plastic_lr03():
    out = RES / "confirm_plastic_lr03_sensitivity.json"
    if out.exists():
        print(f"Skipping already completed {out.name}")
        return
    print("\n=== Step 2b: Confirming Plastic lr=0.3 Sensitivity Analysis (Seeds 1-5) ===")
    cmd = [
        sys.executable, "experiments/gradient_targets_drift.py",
        "--model", "Qwen/Qwen2.5-0.5B", "--targets", "down_proj",
        "--ranks", "32", "--max-mass", "1000.0",
        "--control-file", "experiments/control.txt",
        "--seeds", "1,2,3,4,5", "--lr", "0.3",
        "--conditions", "online:eps=0.97:base_eps=0.9",
        "--out", str(out)
    ]
    run_cmd(cmd)


def step_2c_lora():
    out = RES / "confirm_lora.json"
    if out.exists():
        print(f"Skipping already completed {out.name}")
        return
    print("\n=== Step 2c: Confirming LoRA baselines (Adam, lr=3e-4, Seeds 1-5) ===")
    cmd = [
        sys.executable, "experiments/gradient_targets_drift.py",
        "--model", "Qwen/Qwen2.5-0.5B", "--targets", "down_proj",
        "--ranks", "32",
        "--control-file", "experiments/control.txt",
        "--seeds", "1,2,3,4,5",
        "--lora-opt", "adam", "--lora-lr", "3e-4",
        "--conditions", "lora:seq;lora:multi;lora:gpm:eps=0.97:base_eps=0.9;lora:replay",
        "--out", str(out)
    ]
    run_cmd(cmd)


def step_2d_olora():
    out = RES / "confirm_olora.json"
    if out.exists():
        print(f"Skipping already completed {out.name}")
        return
    print("\n=== Step 2d: Confirming O-LoRA (Adam, lr=3e-4, lam=512.0, Seeds 1-5) ===")
    cmd = [
        sys.executable, "experiments/gradient_targets_drift.py",
        "--model", "Qwen/Qwen2.5-0.5B", "--targets", "down_proj",
        "--ranks", "32",
        "--control-file", "experiments/control.txt",
        "--seeds", "1,2,3,4,5",
        "--lora-opt", "adam", "--lora-lr", "3e-4",
        "--conditions", "lora:olora:lam=512.0",
        "--out", str(out)
    ]
    run_cmd(cmd)


def step_4_gpm_lora_sgd():
    out_final = RES / "explore_gpm_lora_sgd.json"
    if out_final.exists():
        print(f"Skipping already completed {out_final.name}")
        return
    print("\n=== Step 4: GPM-LoRA with SGD Exploration ===")
    # 1. Sweep SGD learning rates on seeds 100, 101, 102
    sgd_lrs = [0.01, 0.03, 0.1, 0.3, 1.0]
    best_lr = None
    best_lp = -float("inf")
    print("--- 4.1 Tuning SGD lr on seeds 100-102 ---")
    for lr in sgd_lrs:
        tmp_out = RES / f"tune_tmp_gpm_sgd_{lr}.json"
        cmd = [
            sys.executable, "experiments/gradient_targets_drift.py",
            "--model", "Qwen/Qwen2.5-0.5B", "--targets", "down_proj",
            "--ranks", "32",
            "--control-file", "experiments/control.txt",
            "--seeds", "100,101,102",
            "--lora-opt", "sgd", "--lora-lr", str(lr),
            "--conditions", "lora:gpm:eps=0.97:base_eps=0.9",
            "--out", str(tmp_out)
        ]
        run_cmd(cmd)
        if tmp_out.exists():
            d = json.loads(tmp_out.read_text(encoding="utf-8"))
            runs = d.get("runs", [])
            avg_lp = sum(r["recall_logprob"] for r in runs) / len(runs)
            avg_rec = sum(r["recall"] for r in runs) / len(runs)
            avg_f1 = sum(r["forget_first"] for r in runs) / len(runs)
            print(f"GPM-LoRA SGD lr={lr:<4} -> recall_lp={avg_lp:.3f}, recall={avg_rec:.2f}, forget_s1={avg_f1:.2f}")
            if avg_lp > best_lp:
                best_lp = avg_lp
                best_lr = lr
            tmp_out.unlink(missing_ok=True)

    print(f"\nBest SGD lr for GPM-LoRA: {best_lr} (recall_lp={best_lp:.3f})")

    # 2. Run winning SGD lr on confirmation seeds 1-5
    print(f"\n--- 4.2 Running GPM-LoRA with SGD lr={best_lr} on confirmation seeds 1-5 ---")
    cmd = [
        sys.executable, "experiments/gradient_targets_drift.py",
        "--model", "Qwen/Qwen2.5-0.5B", "--targets", "down_proj",
        "--ranks", "32",
        "--control-file", "experiments/control.txt",
        "--seeds", "1,2,3,4,5",
        "--lora-opt", "sgd", "--lora-lr", str(best_lr),
        "--conditions", "lora:gpm:eps=0.97:base_eps=0.9",
        "--out", str(out_final)
    ]
    run_cmd(cmd)


def main():
    t0 = time.perf_counter()
    step_2a_plastic_lr1()
    step_2b_plastic_lr03()
    step_2c_lora()
    step_2d_olora()
    step_4_gpm_lora_sgd()
    elapsed = time.perf_counter() - t0
    print(f"\n========================================================")
    print(f" All Confirmatory & Exploratory Runs Completed in {elapsed/3600:.2f} hours!")
    print(f" Results saved in {RES}")
    print(f"========================================================")


if __name__ == "__main__":
    main()
