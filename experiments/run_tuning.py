"""
Hyperparameter tuning on held-out seeds (100, 101, 102).
Selection criterion: Mean final recall logprob across tuning seeds.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent

# Sweeps: 5 values each
PLASTIC_LRS = [0.03, 0.1, 0.3, 1.0, 3.0]
LORA_LRS_ADAM = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
OLORA_LAMS = [0.1, 0.5, 2.0, 8.0, 32.0]  # extended: lam=2.0 was best at the upper edge of the first grid

MASTER_OUT = ROOT / "experiments" / "results" / "tuning_results.json"
SUMMARY_MD = ROOT / "experiments" / "results" / "tuning_summary.md"


def run_cmd(cmd):
    print(f"\n>>> Running: {' '.join(cmd)}")
    sys.stdout.flush()
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print(f"Error:\n{res.stderr}")
        sys.stderr.flush()
    return res.stdout


def extract_metrics(json_file):
    if not Path(json_file).exists():
        return None
    d = json.loads(Path(json_file).read_text(encoding="utf-8"))
    runs = d.get("runs", [])
    if not runs:
        return None
    n = len(runs)
    avg = lambda k: sum(r.get(k, 0.0) for r in runs) / n
    return {
        "n": n,
        "recall_lp": avg("recall_logprob"),
        "recall": avg("recall"),
        "para": avg("paraphrase"),
        "gain_s1": avg("gain_first"),
        "gain_last": avg("gain_last"),
        "forget_s1": avg("forget_first"),
        "ret_s1": avg("retention_first_exact"),
        "dim_q": avg("q_fraction"),
        "kl": avg("kl_heldout"),
        "dppl": sum(r.get("ppl_adapted", 0.0) - r.get("ppl_base", 0.0) for r in runs) / n,
        "state_mb": avg("state_mb_per_fact"),
        "passes": avg("train_passes"),
    }


def save_master(records):
    MASTER_OUT.write_text(json.dumps(records, indent=2), encoding="utf-8")
    # Also write Markdown summary
    lines = [
        "# Hyperparameter Tuning Summary (Seeds 100, 101, 102)",
        "",
        "Selection Criterion: **Mean Final Recall Logprob** (marked with * for best in family)",
        "",
        "| Method | Param | Recall LP | Recall | Para | Gain S1 | Gain Last | Forget S1 | Ret S1 | dim(Q)/d_in | KL held-out | Δppl | State MB/fact | Passes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        star = " *" if r.get("is_best") else ""
        m = r["metrics"]
        lines.append(
            f"| {r['method']} | {r['param_label']}{star} | {m['recall_lp']:.3f} | {m['recall']:.2f} | {m['para']:.2f} | "
            f"{m['gain_s1']:.2f} | {m['gain_last']:.2f} | {m['forget_s1']:.2f} | {m['ret_s1']:.2f} | {m['dim_q']:.3f} | "
            f"{m['kl']:.2e} | {m['dppl']:+.3f} | {m['state_mb']:.4f} | {m['passes']:.0f} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t_start = time.perf_counter()
    records = []
    if MASTER_OUT.exists():
        try:
            records = json.loads(MASTER_OUT.read_text(encoding="utf-8"))
        except Exception:
            records = []

    def already_done(method, param_label):
        return any(r["method"] == method and r["param_label"] == param_label for r in records)

    # 1. Plastic Tuning
    print("\n========================================================")
    print(" 1. Tuning Plastic: online:eps=0.97:base_eps=0.9 (rank 32)")
    print("========================================================")
    for lr in PLASTIC_LRS:
        label = f"lr={lr}"
        if already_done("plastic", label):
            print(f"Skipping already done plastic {label}")
            continue
        tmp_out = f"tuning_tmp_plastic_{lr}.json"
        cmd = [
            sys.executable, "experiments/gradient_targets_drift.py",
            "--model", "Qwen/Qwen2.5-0.5B",
            "--targets", "down_proj",
            "--ranks", "32",
            "--max-mass", "1000.0",
            "--control-file", "experiments/control.txt",
            "--seeds", "100,101,102",
            "--lr", str(lr),
            "--conditions", "online:eps=0.97:base_eps=0.9",
            "--out", tmp_out
        ]
        run_cmd(cmd)
        m = extract_metrics(tmp_out)
        if m:
            records.append({"method": "plastic", "param_label": label, "val": lr, "metrics": m})
            save_master(records)
            Path(tmp_out).unlink(missing_ok=True)

    # 2. LoRA Sequential (B1)
    print("\n========================================================")
    print(" 2. Tuning LoRA Seq (B1) (rank 32, Adam)")
    print("========================================================")
    for lr in LORA_LRS_ADAM:
        label = f"lr={lr:.0e}"
        if already_done("lora:seq", label):
            print(f"Skipping already done lora:seq {label}")
            continue
        tmp_out = f"tuning_tmp_lora_seq_{lr}.json"
        cmd = [
            sys.executable, "experiments/gradient_targets_drift.py",
            "--model", "Qwen/Qwen2.5-0.5B",
            "--targets", "down_proj",
            "--ranks", "32",
            "--control-file", "experiments/control.txt",
            "--seeds", "100,101,102",
            "--lora-opt", "adam",
            "--lora-lr", str(lr),
            "--conditions", "lora:seq",
            "--out", tmp_out
        ]
        run_cmd(cmd)
        m = extract_metrics(tmp_out)
        if m:
            records.append({"method": "lora:seq", "param_label": label, "val": lr, "metrics": m})
            save_master(records)
            Path(tmp_out).unlink(missing_ok=True)

    # 3. LoRA Multi (B2)
    print("\n========================================================")
    print(" 3. Tuning LoRA Multi (B2) (rank 32, Adam)")
    print("========================================================")
    for lr in LORA_LRS_ADAM:
        label = f"lr={lr:.0e}"
        if already_done("lora:multi", label):
            print(f"Skipping already done lora:multi {label}")
            continue
        tmp_out = f"tuning_tmp_lora_multi_{lr}.json"
        cmd = [
            sys.executable, "experiments/gradient_targets_drift.py",
            "--model", "Qwen/Qwen2.5-0.5B",
            "--targets", "down_proj",
            "--ranks", "32",
            "--control-file", "experiments/control.txt",
            "--seeds", "100,101,102",
            "--lora-opt", "adam",
            "--lora-lr", str(lr),
            "--conditions", "lora:multi",
            "--out", tmp_out
        ]
        run_cmd(cmd)
        m = extract_metrics(tmp_out)
        if m:
            records.append({"method": "lora:multi", "param_label": label, "val": lr, "metrics": m})
            save_master(records)
            Path(tmp_out).unlink(missing_ok=True)

    # Find best lr for LoRA Multi to use in O-LoRA
    multi_records = [r for r in records if r["method"] == "lora:multi"]
    best_multi_lr = max(multi_records, key=lambda r: r["metrics"]["recall_lp"])["val"] if multi_records else 1e-3
    print(f"\nBest LoRA Multi learning rate: {best_multi_lr}")

    # 4. LoRA GPM (B4)
    print("\n========================================================")
    print(" 4. Tuning LoRA GPM (B4) (rank 32, Adam)")
    print("========================================================")
    for lr in LORA_LRS_ADAM:
        label = f"lr={lr:.0e}"
        if already_done("lora:gpm", label):
            print(f"Skipping already done lora:gpm {label}")
            continue
        tmp_out = f"tuning_tmp_lora_gpm_{lr}.json"
        cmd = [
            sys.executable, "experiments/gradient_targets_drift.py",
            "--model", "Qwen/Qwen2.5-0.5B",
            "--targets", "down_proj",
            "--ranks", "32",
            "--control-file", "experiments/control.txt",
            "--seeds", "100,101,102",
            "--lora-opt", "adam",
            "--lora-lr", str(lr),
            "--conditions", "lora:gpm:eps=0.97:base_eps=0.9",
            "--out", tmp_out
        ]
        run_cmd(cmd)
        m = extract_metrics(tmp_out)
        if m:
            records.append({"method": "lora:gpm", "param_label": label, "val": lr, "metrics": m})
            save_master(records)
            Path(tmp_out).unlink(missing_ok=True)

    # 5. LoRA Replay (B5)
    print("\n========================================================")
    print(" 5. Tuning LoRA Replay (B5) (rank 32, Adam)")
    print("========================================================")
    for lr in LORA_LRS_ADAM:
        label = f"lr={lr:.0e}"
        if already_done("lora:replay", label):
            print(f"Skipping already done lora:replay {label}")
            continue
        tmp_out = f"tuning_tmp_lora_replay_{lr}.json"
        cmd = [
            sys.executable, "experiments/gradient_targets_drift.py",
            "--model", "Qwen/Qwen2.5-0.5B",
            "--targets", "down_proj",
            "--ranks", "32",
            "--control-file", "experiments/control.txt",
            "--seeds", "100,101,102",
            "--lora-opt", "adam",
            "--lora-lr", str(lr),
            "--conditions", "lora:replay",
            "--out", tmp_out
        ]
        run_cmd(cmd)
        m = extract_metrics(tmp_out)
        if m:
            records.append({"method": "lora:replay", "param_label": label, "val": lr, "metrics": m})
            save_master(records)
            Path(tmp_out).unlink(missing_ok=True)

    # 6. LoRA O-LoRA (B3)
    print("\n========================================================")
    print(f" 6. Tuning O-LoRA (B3) (rank 32, Adam, lr={best_multi_lr})")
    print("========================================================")
    for lam in OLORA_LAMS:
        label = f"lam={lam}"
        if already_done("lora:olora", label):
            print(f"Skipping already done lora:olora {label}")
            continue
        tmp_out = f"tuning_tmp_lora_olora_{lam}.json"
        cmd = [
            sys.executable, "experiments/gradient_targets_drift.py",
            "--model", "Qwen/Qwen2.5-0.5B",
            "--targets", "down_proj",
            "--ranks", "32",
            "--control-file", "experiments/control.txt",
            "--seeds", "100,101,102",
            "--lora-opt", "adam",
            "--lora-lr", str(best_multi_lr),
            "--conditions", f"lora:olora:lam={lam}",
            "--out", tmp_out
        ]
        run_cmd(cmd)
        m = extract_metrics(tmp_out)
        if m:
            records.append({"method": "lora:olora", "param_label": label, "val": lam, "metrics": m})
            save_master(records)
            Path(tmp_out).unlink(missing_ok=True)

    # Mark best in family
    methods = set(r["method"] for r in records)
    for meth in methods:
        fam = [r for r in records if r["method"] == meth]
        best_r = max(fam, key=lambda r: r["metrics"]["recall_lp"])
        for r in fam:
            r["is_best"] = (r == best_r)
    save_master(records)

    elapsed = time.perf_counter() - t_start
    print(f"\nAll tuning sweeps completed in {elapsed/60:.1f} minutes!")
    print(f"Summary table saved to: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
