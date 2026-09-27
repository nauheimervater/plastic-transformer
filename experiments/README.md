# Experiments 1+2: gradient targets, model-level drift, online subspace growth

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
learned mapping on exactly the newly protected inputs. Verified on the tiny model:
a learned fact (logprob −6.68 → −2.16) returns to −6.67 after a single further update
with lr = 1e-6. Rank truncation, decay and the mass cap would also change A on span(Q).

The script therefore consolidates at the end of each session: the active factors are
frozen into a growing per-layer store, the active adapter is reset to zero, and only
then Q is expanded. For the library this suggests a `consolidate()` API instead of
growing Q in place.

## Conditions

| Spec | Purpose |
|---|---|
| `static:k=0` | no protection |
| `static:k=1`, `k=4`, `k=16`, `k=64` | fixed Q from control texts (diagnosis B: does k = 1–4 already give most of the KL reduction?) |
| `static:k=16:center` | SVD of mean-centered activations (mean direction *not* protected) |
| `static:k=16:consol` | fixed Q plus consolidation (separates rank-truncation forgetting from interference) |
| `online:eps=…:base_eps=0.9` | base Q by energy, then Q grows from session inputs |
| `online:eps=0.97:nobase` | Q grows from session inputs only |

Diagnosis A (rank vs. interference): `--ranks 8,32,64`.
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

The JSON output also contains the full retention matrices R[s][j] (logprob and exact)
and the dim(Q) trace per session.

## Running

```bash
pip install transformers

# smoke test (random tiny Qwen2, byte tokenizer): validates the pipeline only
python experiments/gradient_targets_drift.py --tiny --seeds 0 --n-facts 24 \
    --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --max-mass 50

# stage 1: diagnoses A and B
python experiments/gradient_targets_drift.py --model Qwen/Qwen2.5-0.5B --targets down_proj \
    --ranks 8,32 --control-file control.txt \
    --conditions "static:k=0;static:k=1;static:k=4;static:k=16;static:k=16:center;static:k=16:consol"

# stage 2: online Q growth
python experiments/gradient_targets_drift.py --model Qwen/Qwen2.5-0.5B --targets down_proj \
    --ranks 8 --control-file control.txt \
    --conditions "static:k=16:consol;online:eps=0.90:base_eps=0.9;online:eps=0.97:base_eps=0.9;online:eps=0.99:base_eps=0.9;online:eps=0.97:nobase"
```

`control.txt`: blank-line-separated text blocks (a few hundred WikiText paragraphs).
Even blocks calibrate Q, odd blocks measure held-out drift.

Runtime: each condition runs n_facts × epochs backward passes (default 480) per seed.
On a GPU this is minutes; on CPU expect hours for the full default sweep.
Results are written to the JSON after every condition, so interrupted runs keep their data.
