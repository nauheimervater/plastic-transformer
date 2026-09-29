# Plastic Transformer: persistent low-rank fast weights with input-subspace protection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/Plasticity_Is_All_You_Need.pdf)

> **Plasticity Is All You Need? Persistent Low-Rank Fast Weights with Input-Subspace Protection**
> Thomas Nauheimer (nauheimer.t@gmail.com), September 2026.
> Paper: [Zenodo, doi:10.5281/zenodo.23001193](https://doi.org/10.5281/zenodo.23001193) (use the latest version of the record).

> **Project concluded (September 2026). No further development is planned.**
>
> **Result.** On the benchmark studied here (Qwen2.5-0.5B, 48 fictional facts in 4 sessions, confirmed on 5 held-out seeds), retrieving the stored facts into the prompt is at least as good as persistent fast weights at a small fraction of the cost. Among the parametric methods tested, the method gives the best trade-off between retention and drift; its protocol-selected learning rate diverged on one of five seeds, a pre-announced lower rate was stable.
>
> **Transferable findings.** (1) Growing a protected input subspace without first consolidating the fast weights erases exactly what was learned. (2) Output drift of unprotected updates runs mainly along the mean input activation of each layer. (3) The forgetting of GPM-LoRA is not caused by the optimizer (Adam vs. SGD).
>
> **Open question worth pursuing elsewhere.** Whether parametric consolidation adds anything to retrieval on tasks where retrieval fails structurally: two-hop questions across sessions, reversal questions, implicit use of facts.

## What this is

`PlasticLinearProjected` wraps a frozen `nn.Linear` with low-rank fast weights:

```
y = W x + b + γ · (A_frozen + A_active) x,    A = U Vᵀ
```

- Updates of the active factors are right-projected by P = I − QQᵀ. Outputs for inputs **inside** span(Q) are therefore unchanged by adaptation (exact in exact arithmetic, ≤ 10⁻¹⁵ in float64). Inputs with components outside span(Q) are changed, and in a deep network those changes propagate. Layer invariance is not model invariance; model-level drift has to be measured.
- Updates are compressed with thin QR plus a small core SVD: O((d_in + d_out)·m² + m³) per step with m = rank + batch size. The dense weight matrix is never formed.
- `consolidate()` freezes the active factors into a store that later updates never touch. It is required before growing Q to protect something already learned; otherwise the projection erases it. `set_protected_subspace` refuses to run on nonzero active factors.

In a language model, `PlasticModelWrapper.gradient_step` derives layer targets from the token loss, T = Y − lr · N / mean‖x‖² · ∂L/∂Y. Each step is then a projected, NLMS-normalized gradient step on the fast weights: **test-time training of a low-rank state with gradient projection (as in GPM)**. It needs a backward pass through the frozen model; it is not a forward-only update.

## Relation to prior work

Fast weights (Ba et al., 2016; Schlag et al., 2021); gradient projection onto the complement of earlier input activations (OWM, Zeng et al., 2019; GPM, Saha et al., 2021), which is what our update reduces to with gradient-derived targets; low-rank continual learning with session boundaries (O-LoRA, Wang et al., 2023; InfLoRA, Liang & Li, 2024, the closest prior work to our GPM-LoRA baseline); test-time training (Sun et al., 2024; Titans, Behrouz et al., 2024); massive activations (Sun et al., 2024). Like O-LoRA and InfLoRA, the continual procedure here uses session boundaries: consolidation and subspace growth happen between sessions.

## Quickstart

```bash
git clone https://github.com/nauheimervater/plastic-transformer.git
cd plastic-transformer
pip install -r requirements.txt
```

```python
import torch.nn.functional as F
from plastic_transformer import PlasticModelWrapper

# model: a float32 PyTorch model whose target layers are nn.Linear children with these exact names
plastic = PlasticModelWrapper(model, target_modules=["down_proj"], rank=32, max_mass=1000.0)

# optional: protect directions of generic activations (energy criterion, as in GPM)
plastic.calibrate_subspace([control_ids_1, control_ids_2], energy_threshold=0.9)

# learn a session: gradient-derived targets, projected NLMS step on every plastic layer
for ids, labels in session:
    with plastic.capture():
        logits = plastic(ids).logits
        loss = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
    plastic.gradient_step(loss, lr=0.3)

# end of session: consolidate, then protect this session's inputs
plastic.expand_subspace([ids for ids, _ in session], energy_threshold=0.97)

plastic.set_gate(0.0)                      # exact base model
plastic.set_gate(1.0)
plastic.save_synaptic_memory("memory.pt")  # active + consolidated factors and Q
print(plastic.state_bytes() / 1e6, "MB")   # size grows with every session
```

Restrictions: float32/float64 only (no bf16/fp16 or quantized layers); layers are matched by exact child name; the plain `adapt()` delta rule is stable only for learning_rate · λ_max(XᵀX/N) < 2, which is why `gradient_step` normalizes.

## Verification

```bash
python -m unittest discover -s tests       # invariance, consolidation, safe Q growth, roundtrips, gradient step
python examples/reproduce_benchmark.py      # synthetic invariance table (seeds 0..4)
python examples/demo_transformer.py         # toy decoder: drift vs. k, forgetting with/without consolidation
```

Synthetic benchmark (float64, 32 → 16, rank 8, 4 protected and 8 novel directions, 160 updates):

| Seed | Held-out MSE before | after | Max change on span(Q) |
|:---:|:---:|:---:|:---:|
| 0 | 0.254999 | 2.74 × 10⁻¹⁰ | 4.2 × 10⁻¹⁷ |
| 1 | 0.291153 | 3.13 × 10⁻¹⁰ | 4.2 × 10⁻¹⁷ |
| 2 | 0.227038 | 2.44 × 10⁻¹⁰ | 5.6 × 10⁻¹⁷ |
| 3 | 0.413457 | 4.44 × 10⁻¹⁰ | 5.9 × 10⁻¹⁷ |
| 4 | 0.501408 | 5.38 × 10⁻¹⁰ | 5.6 × 10⁻¹⁷ |

The last column depends on the platform BLAS. This verifies the implementation, not usefulness.

## Language-model experiments

`experiments/` contains the Qwen2.5-0.5B experiments (24 plastic `down_proj` layers, 48 fictional facts in 4 sessions) with LoRA, O-LoRA, GPM-LoRA, replay and retrieval baselines; see [experiments/README.md](experiments/README.md). Confirmatory evaluation on five held-out seeds (mean ± 95 % CI):

| Method | Recall | Paraphrase | Forget S1 (nats) | KL held-out | MB/fact | Passes |
|---|---:|---:|---:|---:|---:|---:|
| Ours, lr = 1.0 (selected by protocol) | 0.89 ± 0.26 | 0.75 ± 0.44 | 0.22 ± 0.40 | 0.92 ± 2.41 | 1.47 | 480 |
| Ours, lr = 0.3 (pre-announced sensitivity) | 0.96 ± 0.02 | 0.78 ± 0.07 | 0.05 ± 0.04 | 0.010 ± 0.002 | 1.47 | 480 |
| LoRA + replay | 0.95 ± 0.05 | 0.90 ± 0.03 | 0.11 ± 0.16 | 0.61 ± 0.07 | 0.37 | 840 |
| GPM-LoRA | 0.85 ± 0.11 | 0.78 ± 0.10 | 1.05 ± 0.67 | 0.049 ± 0.010 | 1.47 | 480 |
| Sequential LoRA | 0.43 ± 0.04 | 0.41 ± 0.06 | 3.57 ± 0.89 | 0.61 ± 0.10 | 0.37 | 480 |
| Retrieval (top-4) | 1.00 ± 0.00 | 0.97 ± 0.01 | 0.03 ± 0.02 | 0.032 ± 0.002 | 0.0001 | 0 |

- The protocol-selected learning rate diverged on one of five seeds; lr = 0.3 is stable.
- At lr = 0.3 the method is the best trade-off between retention and drift among the parametric methods tested, with the lowest drift of all methods.
- **Retrieval is at least as good on this benchmark at a tiny fraction of the cost.** The unique fictional names make retrieval unusually easy here; a harder benchmark is the next step.
- Most unprotected drift runs along the mean input activation.

## Paper

[`paper/Plasticity_Is_All_You_Need.md`](paper/Plasticity_Is_All_You_Need.md) is the source; the PDF is generated from it with `python paper/generate_paper_pdf.py` (requires `reportlab`).

## Citation

```bibtex
@misc{nauheimer2026plasticity,
  author       = {Thomas Nauheimer},
  title        = {Plasticity Is All You Need? Persistent Low-Rank Fast Weights with Input-Subspace Protection},
  year         = {2026},
  month        = {September},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23001193},
  url          = {https://github.com/nauheimervater/plastic-transformer},
  note         = {Preprint; project concluded}
}
```

## License

MIT. Copyright © 2026 Thomas Nauheimer.
