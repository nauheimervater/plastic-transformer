# Plasticity Is All You Need? Persistent Low-Rank Fast Weights with Input-Subspace Protection

**Thomas Nauheimer — September 2026 — preprint, version 2**

Status: one 0.5B-parameter model, one synthetic fact benchmark, confirmatory comparison over five held-out seeds (Section 6.5). The title is a research question; on this benchmark the answer is no with respect to retrieval.

## Abstract

We study a frozen linear layer augmented with persistent low-rank fast weights A = UVᵀ whose updates are right-projected onto the orthogonal complement of a protected input subspace span(Q). Updates are compressed by thin QR factorizations and a small core SVD, so the dense weight matrix is never formed; protected inputs are unchanged to machine precision by construction. In a language model the layer targets are derived from the token loss, which turns the update into a projected, NLMS-normalized gradient step: test-time training of a low-rank state with gradient projection in the sense of GPM. Growing Q between sessions requires consolidating the fast weights first; otherwise the projection erases what was learned.

On Qwen2.5-0.5B (24 plastic `down_proj` layers, 48 fictional facts in 4 sessions) the settings were tuned on three seeds and evaluated on five held-out seeds. The learning rate selected by the pre-registered criterion (lr = 1.0) diverged on one of five seeds. The pre-announced sensitivity setting (lr = 0.3) was stable: recall 0.96 ± 0.02, session-1 forgetting 0.05 ± 0.04 nats and the lowest drift on unrelated text of all methods (KL 0.010 ± 0.002 nats per token). Among the parametric methods tested (sequential LoRA, LoRA per session, O-LoRA, GPM-LoRA, LoRA with replay) it is the best trade-off between retention and drift; LoRA with replay reaches similar recall and better paraphrase recall (0.90 versus 0.78) with 60 times more drift and 1.75 times the training passes. Retrieval of the stored fact text into the prompt is at least as good on this benchmark (recall 1.00, paraphrase 0.97) with no training and about 100 bytes per fact instead of 1.5 MB, although the unique fictional names make retrieval unusually easy here. Unprotected drift runs mainly along the mean input activation: protecting that single direction per layer removes about half of it.

## 1. Related work

- **Fast weights.** Ba et al. (2016) used fast weights as temporary memory in recurrent networks; Schlag et al. (2021) showed that linear attention is a fast-weight programmer.
- **Differentiable plasticity.** Miconi et al. (2018) meta-learn plastic connection coefficients.
- **Gradient projection.** OWM (Zeng et al., 2019) and GPM (Saha et al., 2021) project updates onto the orthogonal complement of the input activations of earlier tasks. With gradient-derived targets (Section 2.2) our update is exactly this kind of projection, applied to a low-rank inference-time state instead of the base weights; we make no claim of novelty for the projection itself.
- **Low-rank continual learning.** O-LoRA (Wang et al., 2023) adds a LoRA adapter per task and penalizes overlap between the parameter subspaces of old and new adapters. InfLoRA (Liang & Li, 2024) constructs adapter subspaces that avoid interference with earlier tasks and is the closest prior work to the LoRA variant of our method (baseline B4). Like these methods, our procedure uses session boundaries: consolidation and subspace growth happen between sessions.
- **Test-time training.** TTT layers (Sun et al., 2024) and Titans (Behrouz et al., 2024) update hidden states by inner-loop gradient steps at inference time.
- **Massive activations.** Sun et al. (2024) report a few activation channels with values orders of magnitude above the median in LLMs; our drift analysis (Section 6.4) touches this phenomenon.
- **Retrieval.** A frozen model with retrieval of stored facts is a necessary baseline, not an impossible competitor.

## 2. Model and update

### 2.1 Layer

For row-batched inputs X ∈ ℝ^{N×d_in}:

```
Y = X Wᵀ + b + γ · X (V_frozen U_frozenᵀ + V Uᵀ)
```

W is frozen. U, V ∈ ℝ^{d×r} are the active fast weights; U_frozen, V_frozen hold consolidated sessions and are never updated. The gate γ ∈ [0, 1] scales all fast weights; γ = 0 reproduces the base layer exactly.

Given target activations T, with E = T − Y and P = I − QQᵀ, one step is

```
A_cand = (1 − λ) A + (η γ / N) Eᵀ X P
```

followed by truncation to rank r and a Frobenius-norm cap. This is the delta rule for the local squared error; it is stable only if η · λ_max(XᵀX / N) < 2.

### 2.2 Targets from the token loss

In a language model no layer targets exist. We use, per plastic layer,

```
T = Y − lr · N / mean(‖x‖²) · ∂L/∂Y
```

where L is the cross-entropy on the answer tokens. Then E ∝ −∂L/∂Y and the step becomes

```
ΔA = −lr · (∂L/∂Y)ᵀ X P / mean(‖x‖²)
```

a projected gradient step on A, normalized as in NLMS. It requires a backward pass through the frozen network; it is not a forward-only update. The normalization makes the change of the layer output independent of the input scale of that layer.

## 3. Subspace protection and consolidation

If A₀Q = 0 and every update is right-projected by P, then A_t Q = 0 for all t, so the layer output for inputs in span(Q) is unchanged by adaptation. This is an algebraic property of P, not an empirical finding. It protects only inputs inside span(Q); inputs with components outside span(Q) are changed, and changes propagate to later layers, whose inputs then leave their own calibrated subspaces. Layer invariance therefore does not imply model invariance; model-level drift has to be measured (Section 6).

**Consolidation.** To protect what a session has learned, Q is grown from the activations of that session. If the active fast weights still hold the learned mapping when Q grows, projecting them onto the new complement erases exactly the mapping one wanted to keep (observed: one further update with lr = 10⁻⁶ returned a learned fact to its pre-learning log-probability). Rank truncation, decay and the norm cap would also change A on span(Q). The fast weights are therefore consolidated first: the active factors move into the frozen store and restart at zero, and only then is Q expanded. Since the active part starts at zero, A_active Q = 0 holds for all later updates, and the frozen store is untouched. The implementation refuses to change Q while active factors are nonzero.

**Energy criterion.** New directions are added as in GPM: the residual R = XP of the session activations is decomposed by SVD, and leading directions are added until span(Q) holds a fraction ε of the energy ‖X‖²_F.

## 4. Low-rank compression and cost

The candidate update is written as L Rᵀ with m = r + N columns. Thin QR factorizations L = Q_L R_L and R = Q_R R_R reduce the SVD to the m × m core R_L R_Rᵀ. Cost per step: O((d_in + d_out) m² + m³); forward overhead O(N r (d_in + d_out)); storage O(r (d_in + d_out)) per active or consolidated block plus O(k d_in) for Q. The dense d_out × d_in matrix is never formed. The store grows by r(d_in + d_out) numbers per layer and session.

## 5. Synthetic verification

`examples/reproduce_benchmark.py`: float64 on CPU, frozen affine map 32 → 16, rank 8, 4 protected directions, 8 novel directions, 160 updates per seed; 20 held-out inputs are combinations of the novel directions, so the target correction lies within the available rank by construction.

| Seed | Held-out MSE before | after | Max change on span(Q) |
|:---:|:---:|:---:|:---:|
| 0 | 0.254999 | 2.74 × 10⁻¹⁰ | 4.2 × 10⁻¹⁷ |
| 1 | 0.291153 | 3.13 × 10⁻¹⁰ | 4.2 × 10⁻¹⁷ |
| 2 | 0.227038 | 2.44 × 10⁻¹⁰ | 5.6 × 10⁻¹⁷ |
| 3 | 0.413457 | 4.44 × 10⁻¹⁰ | 5.9 × 10⁻¹⁷ |
| 4 | 0.501408 | 5.38 × 10⁻¹⁰ | 5.6 × 10⁻¹⁷ |

The residual column depends on the platform BLAS; the MSE values reproduce exactly. This verifies the implementation, nothing more.

## 6. Language-model experiments

### 6.1 Setup

Qwen2.5-0.5B in float32; fast weights on `down_proj` in all 24 layers (d_in = 4864, d_out = 896). 48 fictional facts (company → headquarters, founder or product; names and answers generated per seed, so no pretrained knowledge applies) in 4 sessions of 12; the last session uses a different sentence template. 10 epochs per session, one fact per step. Metrics: exact top-1 recall of the answer span in a fresh context; the same for never-trained paraphrases; *forget S1*, the drop in session-1 answer log-probability between the end of session 1 and the end of training; KL(p_base ‖ p_adapted) per token on held-out control text. Code: `experiments/gradient_targets_drift.py`.

### 6.2 Forgetting is mainly interference (seed 0, exploratory)

With a fixed Q of 16 directions from control text (rank 32), forget S1 is 4.09 nats without and 2.83 nats with per-session consolidation, and session-1 exact recall at the end is 0.17 in both cases. Rank truncation accounts for roughly a third of the forgetting; the rest is interference from later sessions. Raising the rank from 8 to 32 raises final recall (0.33 → 0.52 with consolidation) but does not restore retention.

### 6.3 Online subspace growth (seed 0, exploratory)

| Condition (rank 32) | Recall | Paraphrase | Forget S1 (nats) | Ret. S1 exact | dim(Q)/d_in | KL held-out |
|---|---:|---:|---:|---:|---:|---:|
| fixed Q (k = 16) + consolidation | 0.52 | 0.38 | 2.80 | 0.17 | 0.003 | 6.7 × 10⁻³ |
| online, ε = 0.97, base Q (ε = 0.9) | 0.69 | 0.48 | 0.00 | 0.83 | 0.060 | 1.8 × 10⁻³ |
| online, ε = 0.97, no base Q | 0.85 | 0.56 | 0.01 | 0.83 | 0.027 | 1.4 × 10⁻² |

Growing Q from session activations after consolidation removes the measured forgetting in this setup. Protecting control-text directions as well (base Q) lowers drift by about a factor 8 and costs recall: the two variants span a trade-off, not a winner. Single seed, 48 facts; differences of ±0.1 in recall correspond to a few facts.

### 6.4 Where the drift goes (exploratory)

With a fixed Q from control text (seed 0, rank 32), protecting the single leading uncentered singular direction v₁ lowered held-out KL from 9.2 × 10⁻² to 1.9 × 10⁻², while an SVD of mean-centered activations protected much less (2.3 × 10⁻² at k = 16 versus 7.9 × 10⁻³ uncentered).

A dedicated analysis (`experiments/analyze_drift_mechanics.py`; 12 facts, one session, seeds 0–2; KL reduction relative to no protection, paired per seed):

| Protected basis per layer | Per-seed reduction (%) | Mean ± 95 % CI (%) |
|---|---|---:|
| mean direction x̄/‖x̄‖ | 58, 44, 42 | 48 ± 22 |
| v₁ with its five largest channels removed | 43, 28, 27 | 33 ± 23 |
| v₁ (uncentered SVD) | 43, −1, 39 | 27 ± 60 |
| five largest channels (coordinate directions) | 37, −2, 13 | 16 ± 50 |
| five median channels | 19, 3, 4 | 9 ± 22 |
| five random channels | 2, −40, −1 | −13 ± 57 |

Only the mean direction reduces drift consistently on all three seeds. Removing the largest channels from v₁ does not weaken it. A channel-specific effect of the largest channels is not established: its seed-to-seed spread is as large as that of random channels.

The layer structure is heterogeneous. In most layers (4, 6–20, 22, 23) ‖x̄‖² is 1.0–2.4 times the largest covariance eigenvalue, |cos(v₁, x̄)| is 0.90–0.99 and v₁ is spread over tens to hundreds of channels (participation ratio 9–345). In layers 2, 3, 5 and 21, v₁ is essentially a single channel (participation ratio 1.0–1.7, the five largest channels carrying 99–100 % of ‖v₁‖²), consistent with the massive activations described by Sun et al. (2024). Layers 0 and 1 fit neither pattern.

### 6.5 Confirmatory evaluation (seeds 1–5)

Protocol. Hyperparameters were selected per method on seeds 100–102 by final recall log-probability alone (Appendix A); our method at lr = 0.3 was announced as a sensitivity analysis before the confirmatory run; for O-LoRA the λ grid was extended once after the first optimum lay at its edge, and the winner (λ = 512, again at the edge) was then fixed. Seeds 1–5 were not used for any selection. All numbers are generated by `experiments/summarize_results.py` from the raw files in `experiments/results/`.

| Method | Recall | Paraphrase | Forget S1 (nats) | Ret. S1 | KL held-out | Δppl | MB/fact | Passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ours, lr = 1.0 (selected by protocol) | 0.89 ± 0.26 | 0.75 ± 0.44 | 0.22 ± 0.40 | 0.78 ± 0.43 | 0.918 ± 2.405 | +201 ± 558 | 1.47 | 480 |
| Ours, lr = 0.3 (pre-announced sensitivity) | 0.96 ± 0.02 | 0.78 ± 0.07 | 0.05 ± 0.04 | 0.95 ± 0.06 | 0.010 ± 0.002 | +0.10 ± 0.09 | 1.47 | 480 |
| B5 LoRA + replay | 0.95 ± 0.05 | 0.90 ± 0.03 | 0.11 ± 0.16 | 0.93 ± 0.09 | 0.612 ± 0.069 | +6.51 ± 1.33 | 0.37 | 840 |
| B4 GPM-LoRA (Adam) | 0.85 ± 0.11 | 0.78 ± 0.10 | 1.05 ± 0.67 | 0.55 ± 0.32 | 0.049 ± 0.010 | +0.45 ± 0.29 | 1.47 | 480 |
| B3 O-LoRA (λ = 512) | 0.58 ± 0.10 | 0.35 ± 0.16 | 1.77 ± 0.80 | 0.27 ± 0.15 | 1.157 ± 0.267 | +18.7 ± 8.3 | 1.47 | 480 |
| B2 LoRA per session, frozen | 0.47 ± 0.09 | 0.45 ± 0.05 | 3.82 ± 1.47 | 0.02 ± 0.05 | 0.845 ± 0.169 | +10.5 ± 3.1 | 1.47 | 480 |
| B1 sequential LoRA | 0.43 ± 0.04 | 0.41 ± 0.06 | 3.57 ± 0.89 | 0.02 ± 0.05 | 0.607 ± 0.095 | +7.47 ± 3.04 | 0.37 | 480 |
| B6 retrieval, top-4 | 1.00 ± 0.00 | 0.97 ± 0.01 | 0.03 ± 0.02 | 1.00 ± 0.00 | 0.032 ± 0.002 | +0.11 ± 0.08 | 0.0001 | 0 |
| B6′ all facts in context | 0.82 ± 0.07 | 0.71 ± 0.08 | 0.14 ± 0.07 | 0.88 ± 0.21 | 0.345 ± 0.015 | +0.78 ± 0.12 | 0.0001 | 0 |

Mean ± 95 % confidence interval (t-distribution, n = 5). MB/fact excludes Q (about 2.9 MB per fact for ours and B4), which is needed only to continue learning. For retrieval, KL is measured on control text with versus without whatever the retriever returns for it.

The selected setting failed on one seed:

| Seed | Recall | Paraphrase | Forget S1 | KL held-out | Δppl |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.98 | 0.85 | 0.09 | 0.025 | +0.17 |
| 2 | 0.96 | 0.83 | 0.07 | 0.076 | +0.60 |
| 3 | 0.98 | 0.94 | 0.12 | 0.058 | +0.46 |
| 4 | 0.52 | 0.12 | 0.80 | 4.383 | +1006 |
| 5 | 1.00 | 0.98 | 0.05 | 0.046 | +0.34 |

In tuning, lr = 3.0 had already diverged (Appendix A); lr = 1.0 lies closer to that boundary than the three tuning seeds revealed. Nothing in the method guards against such an update; a divergence guard is an obvious addition (Section 8).

Findings. (1) At lr = 0.3 our method reduces session-1 forgetting to 0.05 ± 0.04 nats (3.6–3.8 nats for unprotected LoRA) and has the lowest drift of all methods, about a third of retrieval and 60 times lower than replay. (2) Among parametric methods only replay matches its recall and retention; replay generalizes better to paraphrases, stores four times less, needs 1.75 times the passes and drifts strongly. (3) Retrieval is at least as good on every learning metric at negligible cost. Because entity names are unique strings, lexical retrieval is nearly perfect here (paraphrase hit@4 = 1.00), so B6 is an upper bound for this data, not an estimate for realistic retrieval (hit@4 measured on seed 0). (4) Parameter-space orthogonality (O-LoRA) does little against forgetting in this setting even at λ = 512 and increases drift. (5) Placing all 48 facts in the context is worse than retrieving four.

### 6.6 Why does GPM-LoRA forget? (exploratory)

B4 grows Q with the same criterion as our method and has the same layer-level guarantee, yet forgets 1.05 ± 0.67 nats. We hypothesized that Adam's per-coordinate scaling amplifies updates along the 3 % of session energy left unprotected by ε = 0.97. With SGD (lr = 0.03, selected on seeds 100–102 among {0.01, 0.03, 0.1, 0.3, 1}) B4 learns less (recall 0.65 ± 0.10) and forgets more (1.58 ± 0.55 nats). The optimizer does not explain the forgetting. A remaining hypothesis is propagation: B4 drifts five (Adam) to thirty-five (SGD) times more than our method on held-out text, so its updates in the unprotected directions are larger, and these changes may shift the inputs of session-1 facts in later layers out of their protected subspaces. Measuring how far those inputs move per layer during later sessions would test this. The tuning files of this SGD sweep are not in the repository.

## 7. Limitations and integration boundaries

- One model, one module type, one synthetic benchmark with unique entity names, 48 facts in four sessions. Capacity over many sessions is untested; GPM predicts that Q fills up and later sessions learn less.
- Stability: the protocol-selected learning rate diverged on one of five seeds, and nothing in the update prevents this.
- Retrieval is at least as good on this benchmark at a small fraction of the cost; our method's measured advantages are lower drift on unrelated text and no added context at inference time.
- Paraphrase recall (0.78) is lower than with replay (0.90) or retrieval (0.97), and paraphrases share most tokens with the training questions; reversal and multi-hop questions are untested.
- The fast-weight store grows linearly with sessions (1.5 MB per fact here, plus 2.9 MB per fact for Q). A data-weighted compression, minimizing ‖(A − Â) X_sessions‖ rather than ‖A − Â‖, is the natural next step.
- Learning needs backward passes. Modifying attention projections mid-sequence invalidates the KV cache; quantized or fused kernels need custom support for low-rank addends.

## 8. Next experiments

1. A divergence guard: reject or shrink an update when drift on a small control set exceeds a budget, and re-run the lr sweep with it.
2. The GPM-LoRA question (Section 6.6): per-layer movement of session-1 inputs during later sessions, for our method and B4.
3. A benchmark on which retrieval is not trivial: ambiguous or overlapping entity names, paraphrases without shared tokens, facts that must be combined.
4. Capacity: 20–50 sessions and several hundred facts; trace dim(Q) and per-session learning gain.
5. Store compression and its effect on forgetting; layer ablation; a larger model with settings fixed in advance.

## Appendix A. Tuning (seeds 100–102)

Five settings per method (seven for O-LoRA after one extension), selection by final recall log-probability; full table in `experiments/results/tuning_summary.md`.

| Method | Settings | Selected | Recall at selected | KL at selected |
|---|---|---|---:|---:|
| Ours, online ε = 0.97, base Q | lr ∈ {0.03, 0.1, 0.3, 1.0, 3.0} | 1.0 | 1.00 | 0.044 |
| B1 sequential LoRA (Adam) | lr ∈ {1, 3, 10, 30, 100} × 10⁻⁴ | 3 × 10⁻⁴ | 0.45 | 0.717 |
| B2 LoRA per session (Adam) | same | 3 × 10⁻⁴ | 0.53 | 0.748 |
| B3 O-LoRA (Adam, lr = 3 × 10⁻⁴) | λ ∈ {0.1, 0.5, 2, 8, 32, 128, 512} | 512 | 0.53 | 1.249 |
| B4 GPM-LoRA (Adam) | lr as B1 | 3 × 10⁻⁴ | 0.83 | 0.041 |
| B5 LoRA + replay (Adam) | lr as B1 | 3 × 10⁻⁴ | 0.92 | 0.628 |

For our method the neighbouring settings were lr = 0.3 (recall 0.99, KL 0.008) and lr = 3.0 (recall 0.53, KL 2.04, diverged).

## References

Ba, J., Hinton, G., Mnih, V., Leibo, J. Z., Ionescu, C. (2016). Using fast weights to attend to the recent past. NeurIPS.
Behrouz, A., Zhong, P., Mirrokni, V. (2024). Titans: Learning to memorize at test time. arXiv:2501.00663.
Liang, Y.-S., Li, W.-J. (2024). InfLoRA: Interference-free low-rank adaptation for continual learning. CVPR.
Miconi, T., Clune, J., Stanley, K. O. (2018). Differentiable plasticity. ICML.
Saha, G., Garg, I., Roy, K. (2021). Gradient projection memory for continual learning. ICLR.
Schlag, I., Irie, K., Schmidhuber, J. (2021). Linear transformers are secretly fast weight programmers. ICML.
Sun, M., Chen, X., Kolter, J. Z., Liu, Z. (2024). Massive activations in large language models. COLM.
Sun, Y. et al. (2024). Learning to (learn at test time): RNNs with expressive hidden states. arXiv:2407.04620.
Wang, X. et al. (2023). Orthogonal subspace learning for language model continual learning. Findings of EMNLP.
Zeng, G., Chen, Y., Cui, B., Yu, S. (2019). Continual learning of context-dependent processing in neural networks. Nature Machine Intelligence.
