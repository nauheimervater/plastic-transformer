# Experiments 1–3: gradient targets, drift, online subspace growth, baselines

`gradient_targets_drift.py` tests two questions:

1. Does per-layer subspace protection (A Q = 0) limit drift of the *model's* output
   distribution, and what does it cost in learning capacity?
2. Does growing Q after each session (GPM energy criterion) prevent forgetting of
   earlier sessions, and how fast does the capacity of later sessions collapse?

## Mechanism

Layer targets come from the token loss, `T = Y - lr * N / mean(||x||^2) * dL/dY`,
so `adapt()` performs a projected, NLMS-normalized gradient step on A = U V^T.
This is gradient projection in the sense of GPM (Saha et al., 2021) applied to a
low-rank inference-time state; no novelty claim is made.

## Important: consolidation is required for online Q growth

`PlasticLinearProjected.adapt()` re-projects the existing factor V onto the complement
of Q. If Q is expanded *after* something was learned, the next `adapt()` erases the
learned mapping on exactly the newly protected inputs. Verified on the tiny model
with library 1.0.4: a learned fact (logprob −6.68 → −2.16) returned to −6.67 after a single
further update with lr = 1e-6. Rank truncation, decay and the mass cap would also change A on span(Q).

The script therefore consolidates at the end of each session with
`PlasticLinearProjected.consolidate()` (library ≥ 1.1.0): the active factors are frozen
into a growing per-layer store, the active part restarts at zero, and only then Q is
expanded. Since 1.1.0 the library also refuses to change Q while active factors are
nonzero. (The runs below were made with library 1.0.4 and an equivalent consolidation
store inside the script; both implementations give identical results on the tiny model.)

## Conditions

| Spec | Purpose |
|---|---|
| `static:k=0` | no protection |
| `static:k=1`, `k=4`, `k=16`, `k=64` | fixed Q from control texts (diagnosis B: does k = 1–4 already give most of the KL reduction?) |
| `static:k=16:center` | SVD of mean-centered activations (mean direction *not* protected) |
| `static:k=16:consol` | fixed Q plus consolidation (separates rank-truncation forgetting from interference) |
| `online:eps=…:base_eps=0.9` | base Q by energy, then Q grows from session inputs |
| `online:eps=0.97:nobase` | Q grows from session inputs only |
| `lora:seq` | B1: one LoRA adapter across all sessions, no protection |
| `lora:multi` | B2: new LoRA per session, old ones frozen (consolidation only) |
| `lora:olora:lam=0.5` | B3: B2 + O-LoRA penalty λ Σ‖A_new A_oldᵀ‖²_F (parameter space) |
| `lora:gpm:eps=0.97:base_eps=0.9` | B4: B2 + A ← A·P after every step, Q grown exactly as in `online:*` |
| `lora:replay` | B5: B1 + one replayed earlier fact per step (counts as 2 passes) |
| `rag:k=4` | B6: lexical top-k retrieval of learned facts into the prompt, frozen model |
| `icl:all` | B6′: all learned facts in the prompt, frozen model |

Diagnosis A (rank vs. interference): `--ranks 8,32,64`.
LoRA baselines use the same modules, facts, sessions, epochs and per-session parameter
budget r·(d_in + d_out) as the consolidated fast-weight store. Learning rate and optimizer:
`--lora-lr`, `--lora-opt sgd|adam`. For B4, Adam is safe because the constraint is enforced
on the parameter after each step; projecting only the gradient would not survive Adam's
per-coordinate scaling (verified: |A Q| = 0.25 after 5 Adam steps, 8e-8 after A ← A·P).
The O-LoRA penalty acts on the input-side factor A (r × d_in); the original paper states
it on the factor spanning the update subspace. It is a constraint between parameter
subspaces, not data activations, which is the point of comparing B3 with B4.

Retrieval note: the fictional entity names are unique strings, so lexical retrieval is
nearly perfect (paraphrase hit@4 = 1.00, top-1 = 0.98). B6 is therefore an upper bound
for this data, not an estimate of retrieval quality in realistic settings. Its drift is
measured as KL on control tokens with vs. without whatever the retriever returns for them.
With `--template-shift` (default) the last session uses a different surface template,
so a collapse of *gain last* can be attributed to shared-template protection.

## Metrics (mean ± 95% CI over seeds)

| Column | Meaning |
|---|---|
| recall / paraphrase | exact top-1 answer span, fresh context; paraphrases are never trained |
| gain S1 / gain last | logprob gain of a session right after learning it (capacity over time) |
| forget S1 | logprob of session 1 right after learning minus at the end (positive = forgotten) |
| ret S1 exact | exact recall of session 1 at the end |
| dim(Q)/d_in | protected fraction after the last session |
| KL held-out, Δppl | model-level drift on held-out control texts |
| state MB/fact | persistent state per fact (fp32 factors, Q excluded; B6: fact text) |
| train passes | single-fact forward+backward passes during learning |

The JSON output also contains the full retention matrices R[s][j] (logprob and exact)
and the dim(Q) trace per session.

## Running

```bash
pip install transformers

# smoke test (random tiny Qwen2, byte tokenizer): validates the pipeline only
python experiments/gradient_targets_drift.py --tiny --seeds 0,1 \
    --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --epochs 15 --max-mass 50

# stage 1: diagnoses A and B
python experiments/gradient_targets_drift.py --model Qwen/Qwen2.5-0.5B --targets down_proj \
    --ranks 8,32 --control-file control.txt \
    --conditions "static:k=0;static:k=1;static:k=4;static:k=16;static:k=16:center;static:k=16:consol"

# stage 2: online Q growth
python experiments/gradient_targets_drift.py --model Qwen/Qwen2.5-0.5B --targets down_proj \
    --ranks 8 --control-file control.txt \
    --conditions "static:k=16:consol;online:eps=0.90:base_eps=0.9;online:eps=0.97:base_eps=0.9;online:eps=0.99:base_eps=0.9;online:eps=0.97:nobase"

# stage 3: tuning on seeds 100-102 (all methods): python experiments/run_tuning.py
# confirmatory run on seeds 1-5 with the selected settings, e.g.:
python experiments/gradient_targets_drift.py --model Qwen/Qwen2.5-0.5B --targets down_proj \
    --ranks 32 --max-mass 1000 --control-file experiments/control.txt --seeds 1,2,3,4,5 --lr 1.0 \
    --conditions "online:eps=0.97:base_eps=0.9;rag:k=4;icl:all"
python experiments/gradient_targets_drift.py --model Qwen/Qwen2.5-0.5B --targets down_proj \
    --ranks 32 --control-file experiments/control.txt --seeds 1,2,3,4,5 --lora-opt adam --lora-lr 3e-4 \
    --conditions "lora:seq;lora:multi;lora:gpm:eps=0.97:base_eps=0.9;lora:replay"
# O-LoRA: rerun the extended lambda grid first (run_tuning.py), then use the selected lambda
```

`control.txt`: blank-line-separated text blocks. The shipped `experiments/control.txt` has 33 blocks;
more text is better for large k.
Even blocks calibrate Q, odd blocks measure held-out drift.

Runtime: each condition runs n_facts × epochs backward passes (default 480) per seed.
On a GPU this is minutes; on CPU expect hours for the full default sweep.
Results are written to the JSON after every condition, so interrupted runs keep their data.

## Results so far

All runs: Qwen2.5-0.5B, float32, `down_proj` in all 24 layers (d_in = 4864), 48 facts in
4 sessions with template shift, 10 epochs. Raw data in `experiments/results/`.

| File | Content |
|---|---|
| `exp1_ksweep_12facts_seed0.json` | first run of the original script version (12 facts, 3 sessions, k-sweep) |
| `stage2_online_q_seed0.json` | Stage 2, seed 0 |
| `tuning_results.json`, `tuning_summary.md` | tuning on seeds 100–102, all methods (produced by `run_tuning.py`) |

The Stage 1 raw JSON was not committed; its numbers below are taken from the console output.

### Stage 1 (seed 0, exploratory): diagnoses A and B

| rank | condition | recall | para | gain S1 | gain last | forget S1 | ret S1 | dim(Q)/d_in | KL held-out | Δppl |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | static:k=0 | 0.27 | 0.08 | 8.02 | 6.89 | 5.25 | 0.00 | 0.000 | 6.3e-02 | +0.689 |
| 8 | static:k=1 | 0.25 | 0.10 | 7.84 | 6.98 | 5.02 | 0.00 | 0.000 | 2.0e-02 | +0.368 |
| 8 | static:k=4 | 0.21 | 0.10 | 7.73 | 6.63 | 5.66 | 0.00 | 0.001 | 1.0e-02 | +0.191 |
| 8 | static:k=16 | 0.25 | 0.10 | 7.83 | 6.52 | 6.25 | 0.00 | 0.003 | 8.2e-03 | +0.238 |
| 8 | static:k=16:center | 0.25 | 0.06 | 7.79 | 6.48 | 4.87 | 0.00 | 0.003 | 2.9e-02 | +0.515 |
| 8 | static:k=16:consol | 0.33 | 0.23 | 7.83 | 7.22 | 3.89 | 0.00 | 0.003 | 6.3e-03 | +0.170 |
| 32 | static:k=0 | 0.44 | 0.25 | 8.16 | 7.54 | 5.20 | 0.00 | 0.000 | 9.2e-02 | +0.818 |
| 32 | static:k=1 | 0.48 | 0.33 | 8.20 | 7.44 | 3.31 | 0.08 | 0.000 | 1.9e-02 | +0.195 |
| 32 | static:k=4 | 0.46 | 0.23 | 8.30 | 7.51 | 4.36 | 0.00 | 0.001 | 9.2e-03 | +0.121 |
| 32 | static:k=16 | 0.40 | 0.29 | 8.36 | 7.26 | 4.09 | 0.17 | 0.003 | 7.9e-03 | +0.055 |
| 32 | static:k=16:center | 0.48 | 0.23 | 8.37 | 7.50 | 4.29 | 0.00 | 0.003 | 2.3e-02 | +0.159 |
| 32 | static:k=16:consol | 0.52 | 0.38 | 8.36 | 7.58 | 2.83 | 0.17 | 0.003 | 6.7e-03 | -0.017 |

Reading (one seed; recall steps of 1/48, retention steps of 1/12):
- **Diagnosis A.** Consolidation alone lowers forget S1 by about a third (6.25 → 3.89 at rank 8,
  4.09 → 2.83 at rank 32), but retention of session 1 stays at 0–0.17 without online Q.
  Forgetting is mainly interference; rank truncation is a secondary cause. Rank 32 raises
  capacity (recall), not retention.
- **Diagnosis B.** k = 1 (uncentered) lowers held-out KL by a factor 3–5; centering the SVD
  makes protection clearly worse. The leading uncentered direction carries most of the drift.

### Stage 2 (seed 0, exploratory): online Q growth with consolidation

| rank | condition | recall | para | gain S1 | gain last | forget S1 | ret S1 | dim(Q)/d_in | KL held-out | Δppl |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | static:k=16:consol | 0.33 | 0.23 | 7.83 | 7.22 | 3.89 | 0.00 | 0.003 | 6.3e-03 | +0.170 |
| 8 | online:eps=0.90:base_eps=0.9 | 0.42 | 0.17 | 7.69 | 5.35 | 0.11 | 0.50 | 0.048 | 1.4e-03 | -0.039 |
| 8 | online:eps=0.97:base_eps=0.9 | 0.38 | 0.15 | 7.69 | 4.46 | 0.00 | 0.67 | 0.061 | 1.2e-03 | -0.061 |
| 8 | online:eps=0.99:base_eps=0.9 | 0.40 | 0.17 | 7.69 | 5.21 | 0.00 | 0.67 | 0.071 | 1.2e-03 | -0.050 |
| 8 | online:eps=0.97:nobase | 0.44 | 0.23 | 8.02 | 6.08 | 0.01 | 0.75 | 0.026 | 1.2e-02 | -0.112 |
| 32 | static:k=16:consol | 0.52 | 0.38 | 8.36 | 7.58 | 2.83 | 0.17 | 0.003 | 6.7e-03 | -0.017 |
| 32 | online:eps=0.90:base_eps=0.9 | 0.77 | 0.50 | 8.18 | 6.96 | 0.12 | 0.67 | 0.048 | 1.9e-03 | -0.041 |
| 32 | online:eps=0.97:base_eps=0.9 | 0.69 | 0.48 | 8.18 | 6.63 | 0.00 | 0.83 | 0.060 | 1.8e-03 | -0.044 |
| 32 | online:eps=0.99:base_eps=0.9 | 0.65 | 0.48 | 8.18 | 6.46 | 0.00 | 0.75 | 0.070 | 2.0e-03 | -0.017 |
| 32 | online:eps=0.97:nobase | 0.85 | 0.56 | 8.16 | 6.94 | 0.01 | 0.83 | 0.027 | 1.4e-02 | -0.062 |

Reading:
- In this setup, online Q growth after consolidation removes the measured forgetting of
  session 1 (forget S1 ≈ 0 against 2.8–3.9 nats for fixed Q).
- `nobase` and `base_eps` are two ends of a trade-off: more recall versus about 8× less drift.
  Differences among the eps values are within the resolution of one seed.
- Learning capacity of the last session drops with online Q (gain last 7.22 → 4.46–5.35 at
  rank 8, 7.58 → 6.46–6.96 at rank 32), as expected from GPM.
- Δppl is negative in all online conditions; with one seed and no confidence intervals this is
  not evidence of an improvement.
- dim(Q)/d_in = 0.06 means about 290 of 4864 input dimensions per layer are protected; updates
  are confined to the remaining 94 %.

### Tuning (seeds 100–102)

Full table: `results/tuning_summary.md`. Selection by final recall log-probability only, five
settings per method; the selected rows are therefore optimistic maxima.

| Method | Selected | Recall | Para | Forget S1 | Ret S1 | KL held-out | Δppl | MB/fact | Passes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| online:eps=0.97:base_eps=0.9 | lr = 1.0 | 1.00 | 0.89 | 0.01 | 1.00 | 4.4e-02 | +0.28 | 1.47 | 480 |
| lora:seq | 3e-4 | 0.45 | 0.42 | 3.95 | 0.03 | 7.2e-01 | +8.7 | 0.37 | 480 |
| lora:multi | 3e-4 | 0.53 | 0.49 | 4.09 | 0.06 | 7.5e-01 | +8.6 | 1.47 | 480 |
| lora:olora | λ = 2.0 | 0.58 | 0.50 | 3.68 | 0.11 | 7.9e-01 | +9.8 | 1.47 | 480 |
| lora:gpm | 3e-4 | 0.83 | 0.71 | 1.33 | 0.47 | 4.1e-02 | +0.43 | 1.47 | 480 |
| lora:replay | 3e-4 | 0.92 | 0.84 | 0.01 | 0.97 | 6.3e-01 | +6.7 | 0.37 | 840 |

Issues to address before or in the confirmatory run:
- λ = 2.0 was the upper edge of the O-LoRA grid; `run_tuning.py` now sweeps {0.1, 0.5, 2, 8, 32}.
- Our selected lr = 1.0 drifts 5.7× more than lr = 0.3 (KL 7.7e-03, Δppl +0.07) at nearly
  identical recall (0.99). This is a consequence of selecting on recall alone; report both.
- All LoRA variants drift strongly (KL ≈ 0.6–0.8) even at the smallest learning rate tried.
- GPM-LoRA uses the same Q construction and guarantee as our method yet forgets more (1.33 nats).
  Hypothesis: Adam's per-coordinate scaling amplifies updates along the ~3 % of session energy
  that eps = 0.97 leaves unprotected. Testable with SGD for B4.
- MB/fact excludes Q (about 2.9 MB per fact for our method and B4 here), which is needed only to
  continue learning. Retrieval stores about 100 bytes per fact.
- Retrieval and in-context baselines have not yet been run on Qwen.

### Drift mechanism (seeds 0–2, 12 facts; analysis script not yet in the repository)

KL reduction on held-out text relative to no protection: Q = x̄/‖x̄‖ 50 %, Q = v₁ 33 %,
v₁ without its five largest channels 34 %, five largest channels as coordinate directions 20 %,
five median channels 10 %, five random channels −10 %. The two controls indicate noise of about
±10 %: the mean direction is clearly the larger lever; the channel-specific effect is suggestive,
not established. In the middle layers ‖x̄‖² exceeds the largest covariance eigenvalue by
1.2–1.8× and |cos(v₁, x̄)| ≥ 0.93; in layers 0 and 23 v₁ concentrates on a few channels with
magnitudes up to 146× the median.
