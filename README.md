# 🧬 Plastic Transformer: Persistent Fast-Weight Adaptation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper: PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](paper/Plasticity_Is_All_You_Need.pdf)

> **"Plasticity Is All You Need? A Testable Proposal for Persistent Fast-Weight Adaptation in Neural Architectures"**  
> *Author:* **Thomas Nauheimer** (nauheimer.t@gmail.com) — September 2026  
> *Status:* Research proposal with a minimal reproducible PyTorch implementation.

---

## 📌 Context: Static Weights vs. Test-Time Plasticity

Standard autoregressive language models operate with frozen parameters post-pretraining:
- In-context memory (context window buffer) is transient: clearing the prompt resets state to baseline.
- Traditional offline fine-tuning (SGD, full LoRA updates) is computationally heavy and risks catastrophic forgetting.

This repository implements a lightweight, low-rank fast-weight adapter (**`PlasticLinearProjected`**) that participates directly in tensor contractions during inference:

$$W_{\text{eff}}(t) = W_{\text{slow}} + \gamma \cdot \left( U(t) V(t)^T \right)$$

- **$W_{\text{slow}}$**: Frozen base parameter matrix ($\text{requires\_grad}=\text{False}$).
- **$U(t) \in \mathbb{R}^{d_{\text{out}} \times r}, V(t) \in \mathbb{R}^{d_{\text{in}} \times r}$**: Dynamic low-rank factors ($r \ll d$).
- **Subspace Invariance**: Right-projection $P = I - Q Q^T$ onto the orthogonal complement of protected feature basis $Q \in \mathbb{R}^{d_{\text{in}} \times k}$. For any input lying entirely in $\text{span}(Q)$, $A_t Q = 0$ holds algebraically by construction ($\le 10^{-16}$ numerical residual).

```mermaid
flowchart LR
    X["Input x"] --> Slow["W_slow (Frozen Base)"]
    X --> Fast["(x · V) · U^T (Fast Weights)"]
    Slow --> Add((+))
    Fast -->|"x gamma"| Add
    Add --> Y["Output y"]
    
    Y -.->|"Projected Delta-Rule Update"| Fast

    style Slow fill:#2b2d42,color:#fff
    style Fast fill:#d90429,color:#fff
    style Add fill:#8d99ae,color:#fff
```

---

## 🔬 Relationship to Foundational Literature

This work investigates the intersection of classical fast weights and modern continual learning:
- **Fast Weights & Attention:** Ba et al. (2016); Schlag et al. (2021) demonstrated that linear transformers act as fast-weight programmers.
- **Subspace Gradient Projection:** Orthogonal Weights Modification (OWM; Zeng et al., 2019) and Gradient Projection Memory (GPM; Saha et al., 2021) project parameter updates onto orthogonal complements to prevent catastrophic interference.
- **Parameter vs. Activation Orthogonality (O-LoRA):** Wang et al. (2023; O-LoRA) enforce parameter-space orthogonality across discrete sequential tasks. In contrast, our approach projects directly in activation space (onto the nullspace of historical activations) and operates continuously without discrete task boundaries.
- **Test-Time Training (TTT):** Sun et al. (2024) and Titans (Behrouz et al., 2024) explore inference-time inner-loop gradient adaptation.

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/nauheimervater/plastic-transformer.git
cd plastic-transformer
pip install -r requirements.txt
```

### 2. Injecting Plasticity and Subspace Calibration

```python
import torch
from plastic_transformer import PlasticModelWrapper

# Wrap any PyTorch model (e.g. Llama/Qwen attention and MLP layers)
model = YourPretrainedTransformer()
plastic_model = PlasticModelWrapper(
    model, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    rank=8, 
    learning_rate=0.5
)

# Automated Subspace Calibration on baseline prompts (extracts protected span(Q)):
plastic_model.calibrate_subspace(baseline_prompts, k=4)

# Standard forward inference (read-only; forward evaluation never alters weights)
out = plastic_model(input_tokens)

# Save lightweight fast-weight buffers (< 5 MB)
plastic_model.save_synaptic_memory("episodic_snapshot.pt")

# Restore exact base model behavior anytime by setting gate to 0.0
plastic_model.set_gate(0.0)
```

---

## 🧪 Experimental Verification

Run the reproducible verification suite:

```bash
# 1. Run all unit tests (subspace invariance, calibration, dense reference, state roundtrip)
python -m unittest discover -s tests

# 2. Reproduce exact Table 1 multi-seed benchmark (seeds 0..4)
python examples/reproduce_benchmark.py

# 3. Run Llama-style decoder calibration & adaptation demo
python examples/demo_transformer.py
```

### Measured Subspace Invariance Benchmark:

Tested on CPU, float64, 32-to-16 projection, rank 8, 4 protected basis directions, 8 novel adaptation directions, 160 updates (run `python examples/reproduce_benchmark.py` to reproduce):

| Seed | Held-out MSE Before | Held-out MSE After | Max Protected Change | Invariance Check |
|:---:|:---:|:---:|:---:|:---:|
| **0** | 0.254999 | $2.74 \times 10^{-10}$ | $3.47 \times 10^{-17}$ | Exact ($AQ=0$) |
| **1** | 0.291153 | $3.13 \times 10^{-10}$ | $2.78 \times 10^{-17}$ | Exact ($AQ=0$) |
| **2** | 0.227038 | $2.44 \times 10^{-10}$ | $2.78 \times 10^{-17}$ | Exact ($AQ=0$) |
| **3** | 0.413457 | $4.44 \times 10^{-10}$ | $9.02 \times 10^{-17}$ | Exact ($AQ=0$) |
| **4** | 0.501408 | $5.38 \times 10^{-10}$ | $3.47 \times 10^{-17}$ | Exact ($AQ=0$) |

*Note on Subspace Coverage:* Invariance holds algebraically for inputs strictly in $\text{span}(Q)$. Inputs with components orthogonal to $\text{span}(Q)$ will undergo adaptation.

---

## 📄 Research Paper

- Preprint Markdown: [`paper/Plasticity_Is_All_You_Need.md`](paper/Plasticity_Is_All_You_Need.md)
- Compiled publication PDF: [`paper/Plasticity_Is_All_You_Need.pdf`](paper/Plasticity_Is_All_You_Need.pdf)

To recompile the PDF:
```bash
python paper/generate_paper_pdf.py
```

---

## 📚 Citation

```bibtex
@misc{nauheimer2026plasticity,
  author       = {Thomas Nauheimer},
  title        = {Plasticity Is All You Need? A Testable Proposal for Persistent Fast-Weight Adaptation in Neural Architectures},
  year         = {2026},
  month        = {September},
  howpublished = {\url{https://github.com/nauheimervater/plastic-transformer}},
  note         = {Preprint}
}
```

---

## ⚖️ License

Released under the **MIT License**. Copyright &copy; 2026 Thomas Nauheimer.
