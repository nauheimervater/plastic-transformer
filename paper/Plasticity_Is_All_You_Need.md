# Plasticity Is All You Need? A Testable Proposal for Persistent Fast-Weight Adaptation

**Research concept: Thomas Nauheimer — September 2026**

Status: Research proposal with a minimal reproducible implementation. No demonstrated LLM fact-learning, universal forgetting guarantee, learned meta-controller, or fused GPU kernel is claimed. The title is a research question, not an established result.

## Abstract

We investigate a frozen linear transformation augmented with persistent low-rank fast weights. Unlike external sidecar or retrieval-augmented architectures, the supplied `PlasticLinearProjected` adapter participates directly in a PyTorch model's forward computation. It applies explicit target-driven delta-rule updates and protects a specified input subspace $\text{span}(Q)$ via orthogonal projection. Thin QR factorizations and a small-core SVD compress updates without explicitly reconstructing the dense fast-weight matrix. Five synthetic linear-regression experiments demonstrate online adaptation, preservation of protected inputs within floating-point precision ($\le 10^{-16}$), unchanged base weights, and deterministic state serialization. These experiments establish implementation correctness in a testable, reproducible setting; they do not establish unconstrained continual language learning or general open-ended reasoning.

## 1. Relationship to Existing Work

Fast weights, test-time adaptation, and trainable plasticity have established theoretical precedents:

- **Fast Weights & Attention:** Ba et al. (2016) explored temporary fast-weight memory for recurrent networks. Schlag et al. (2021) demonstrated that linearized self-attention mechanisms operate as fast-weight programmers, bridging transformers and associative memory.
- **Differentiable Plasticity:** Miconi et al. (2018) optimized plastic connection coefficients through outer-loop meta-learning.
- **Subspace & Gradient Projection:** Orthogonal Weights Modification (OWM; Zeng et al., 2019) and Gradient Projection Memory (GPM; Saha et al., 2021) project gradient updates onto the orthogonal complement of protected feature subspaces to prevent catastrophic forgetting.
- **Parameter-Space vs. Activation-Space Orthogonality (O-LoRA):** Wang et al. (2023; O-LoRA) enforce orthogonality among low-rank adapter weights across sequential tasks. Crucially, O-LoRA operates in parameter space (gradient and weight dimensions) and assumes discrete task boundaries. In contrast, our approach projects directly in activation space (onto the nullspace of historical activations) and operates continuously without task boundary demarcation.
- **Test-Time Training (TTT):** Recent architectures such as TTT-Linear / TTT-MLP (Sun et al., 2024) and Titans (Behrouz et al., 2024) utilize hidden model states updated by an inner-loop gradient step at inference time.
- **Parameter Importance:** Kirkpatrick et al. (2017; EWC) employ quadratic penalties derived from Fisher information matrices to safeguard critical parameters.

External databases and vector retrieval (RAG) provide cross-session memory without modifying model weights. Consequently, a static-weight model with contextual retrieval is a necessary baseline, not an impossible competitor.

## 2. Implemented Model and Update Formulation

For row-batched inputs $X \in \mathbb{R}^{N \times d_{\text{in}}}$:

$$A = U V^T$$
$$Y = X W_{\text{slow}}^T + b + \gamma (X V) U^T$$

$U \in \mathbb{R}^{d_{\text{out}} \times r}$ and $V \in \mathbb{R}^{d_{\text{in}} \times r}$. The base weight matrix $W_{\text{slow}}$ remains strictly frozen ($\text{requires\_grad}=\text{False}$). A scalar gate $\gamma \in [0, 1]$ scales the fast-weight contribution; $\gamma = 0$ reproduces the base layer output exactly and disables adaptation.

Given explicit, verified target activations $T \in \mathbb{R}^{N \times d_{\text{out}}}$, define:

$$E = T - Y$$
$$\mathcal{L} = \frac{\|E\|_F^2}{2N}$$
$$P = I - Q Q^T$$
$$A_{\text{cand}} = (1 - \lambda) A + \frac{\eta \gamma}{N} E^T X P$$

The update is a projected delta rule for this local squared-error objective, followed by thin-QR rank truncation and a Frobenius-norm cap. In an autoregressive language model, obtaining verified target activations $T$ across intermediate hidden layers remains an unsolved integration requirement; unverified model generations must not be treated automatically as correct learning targets.

Forward evaluation never modifies adapter buffers. The caller invokes `.adapt(X, T)` explicitly after validating a training example.

## 3. Subspace Protection and Invariance

Let $Q \in \mathbb{R}^{d_{\text{in}} \times k}$ denote an orthonormal basis for protected input features. If $A_0 Q = 0$ initially and every subsequent candidate update is right-projected by $P = I - Q Q^T$, then:

$$A_t Q = 0$$

holds identically in exact arithmetic. Consequently, the adapter leaves the linear layer output strictly unchanged for all inputs lying entirely within $\text{span}(Q)$.

The implementation re-projects right factors following SVD compression, ensuring that floating-point residuals remain near machine epsilon ($\le 10^{-16}$). 

### Critical Qualification on Subspace Coverage:
Subspace invariance is an algebraic property of the orthogonal projector $P$, not an empirical discovery. It protects *only* inputs residing strictly in $\text{span}(Q)$. Natural language activations in deep networks almost always possess projection components outside $\text{span}(Q)$, which will undergo adaptation. Furthermore, every protected dimension allocated to $Q$ reduces the effective rank capacity available for novel associations.

### Correction on Fisher Information:
For a parameter vector $\theta \in \mathbb{R}^p$, the Fisher information matrix is $p \times p$. A Fisher-nullspace projection acts on $\text{vec}(\Delta A)$ in parameter space; it is not mathematically equivalent to right-projecting input features $X P$ in activation space without strong structural assumptions. The supplied adapter uses input-subspace protection directly and does not estimate parameter-space Fisher matrices.

## 4. Efficient Low-Rank Update Implementation

Represent candidate updates as $L R^T$ with $m = r + N$ columns:

$$L = \left[ \sqrt{1 - \lambda} U, \quad \sqrt{\frac{\eta \gamma}{N}} E^T \right]$$
$$R = \left[ \sqrt{1 - \lambda} P V, \quad \sqrt{\frac{\eta \gamma}{N}} P X^T \right]$$

Thin QR factorizations $L = Q_L R_L$ and $R = Q_R R_R$ reduce the singular value decomposition to the small core matrix $R_L R_R^T \in \mathbb{R}^{m \times m}$. Retaining its leading $r$ singular values, scaling them when their Frobenius norm exceeds $M_{\text{max}}$, and reconstructing the two low-rank factors avoids dense $d_{\text{out}} \times d_{\text{in}}$ matrix materialization.

### Computational and Memory Complexity:
- Thin QR factorization of $L \in \mathbb{R}^{d_{\text{out}} \times m}$ and $R \in \mathbb{R}^{d_{\text{in}} \times m}$: $\mathcal{O}((d_{\text{in}} + d_{\text{out}}) m^2)$.
- Core SVD of $R_L R_R^T \in \mathbb{R}^{m \times m}$: $\mathcal{O}(m^3)$.
- Factor reconstruction: $\mathcal{O}((d_{\text{in}} + d_{\text{out}}) m r)$.
- Total adaptation step complexity: $\mathcal{O}((d_{\text{in}} + d_{\text{out}}) m^2 + m^3)$, where $m = r + N$. The dense dimension product $d_{\text{in}} \cdot d_{\text{out}}$ never appears because full weight matrices are never materialized.
- Computational overhead per forward pass: $\mathcal{O}(N r (d_{\text{in}} + d_{\text{out}}))$.
- Fast-weight state storage: $\mathcal{O}(r (d_{\text{in}} + d_{\text{out}}) + k d_{\text{in}})$.

## 5. Measured Experiment and Reproduction

To reproduce Table 1 on stdout, run:
```bash
python examples/reproduce_benchmark.py
```
To run the full unit test suite:
```bash
python -m unittest discover -s tests
```

Configuration: CPU, single-threaded PyTorch, float64; frozen affine projection ($d_{\text{in}}=32, d_{\text{out}}=16$); rank $r=8$; 4 protected basis directions; 8 novel adaptation directions; 160 supervised delta updates per seed. Twenty held-out inputs are linear combinations of the 8 novel directions. Target corrections lie within available rank capacity by construction.

| Seed | Held-out MSE Before | Held-out MSE After | Max Protected Change | Invariance Check |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.254999 | $2.74 \times 10^{-10}$ | $3.47 \times 10^{-17}$ | Exact ($AQ=0$) |
| 1 | 0.291153 | $3.13 \times 10^{-10}$ | $2.78 \times 10^{-17}$ | Exact ($AQ=0$) |
| 2 | 0.227038 | $2.44 \times 10^{-10}$ | $2.78 \times 10^{-17}$ | Exact ($AQ=0$) |
| 3 | 0.413457 | $4.44 \times 10^{-10}$ | $9.02 \times 10^{-17}$ | Exact ($AQ=0$) |
| 4 | 0.501408 | $5.38 \times 10^{-10}$ | $3.47 \times 10^{-17}$ | Exact ($AQ=0$) |

Across all seeds, base weights remain bit-identical and `state_dict` roundtrips reproduce outputs exactly.

## 6. Architecture Constraints and Production LLM Integration Boundaries

Integrating fast weights into production autoregressive language models introduces several concrete boundaries:

1. **Direct Parameter Access:** Fast weights must participate in forward tensor contractions. An external inference proxy or API sidecar cannot insert plastic linear layers into a model's computational graph.
2. **KV-Cache Coherency:** In autoregressive generation, past key-value activations are cached. Modifying attention projection weights ($W_q, W_k, W_v$) mid-sequence alters the functional mapping, invalidating cached key-value states computed under earlier weight configurations.
3. **Target Derivation:** In linear regression, ground-truth targets $T$ are explicit. In multi-layer LLMs, deriving layer-wise target activations requires either backpropagation of token-level cross-entropy loss or a learned credit-assignment mechanism.
4. **Quantization & Fused Kernels:** Standard 4-bit/8-bit quantized linear layers (AWQ, GPTQ) and fused attention kernels cannot be replaced drop-in by standard low-rank addends without custom CUDA/Triton kernels.

## 7. Falsifiable Next Experiments

1. **Capacity and Interference:** Evaluate sequential targets exceeding rank capacity; measure retention degradation under varying protected subspace dimensions $k$.
2. **Language Adaptation vs. RAG:** Compare plastic transformer adaptation against an equal-context retrieval baseline on verified factual association tasks.
3. **Target Generation:** Benchmark local synthetic target signals (e.g. contrastive next-token predictions) against full-model backpropagation.
4. **Compute Benchmarking:** Profile QR-core SVD update latency against fused dense updates across varying batch and feature dimensions on GPU.

Plasticity is a testable mechanism for localized parameter adaptation. It does not replace foundational pretraining, nor does it establish subjective agency or unconstrained memory.
