# Plasticity Is All You Need? A Testable Proposal for Persistent Fast-Weight Adaptation

**Research concept: Nauheimer et al. — revised technical draft, 10 September 2026**

Status: research proposal with a small reproducible implementation. No demonstrated LLM fact-learning, universal forgetting guarantee, learned meta-controller, or FlashPlastic GPU kernel is claimed. The title is a research question, not an established result.

## Abstract

We investigate a frozen linear transformation augmented with persistent low-rank fast weights. Unlike the existing LM Studio sidecar, the supplied `PlasticLinear` adapter participates in a PyTorch model's forward computation. It applies explicit target-driven delta-rule updates and protects a specified input subspace. QR decompositions and a small-core SVD compress updates without explicitly reconstructing the dense fast-weight matrix. Five synthetic linear-regression experiments demonstrate adaptation, preservation of protected inputs, unchanged base weights, and state serialization. These experiments establish implementation correctness in a deliberately simple setting; they do not establish continual language learning or preservation of general knowledge.

## 1. Relationship to existing work

Fast weights and trainable plasticity have established precedents. Ba et al. (2016) explore temporary fast-weight memory. Miconi et al. (2018) optimize plasticity through an outer training loop. Kirkpatrick et al. (2017) use Fisher-based parameter importance to mitigate forgetting. The present combination and implementation must be evaluated against these approaches before any novelty claim is justified.

Sources:
- [Ba et al., Using Fast Weights to Attend to the Recent Past](https://arxiv.org/abs/1610.06258)
- [Miconi et al., Differentiable plasticity: training plastic neural networks with backpropagation](https://proceedings.mlr.press/v80/miconi18a.html)
- [Kirkpatrick et al., Overcoming catastrophic forgetting in neural networks](https://doi.org/10.1073/pnas.1611835114)

External databases and retrieval can provide cross-session memory without changing model weights. Consequently, a static-weight model plus retrieval is a necessary baseline, not an impossible competitor.

## 2. Implemented model and update

For row-batched inputs X of shape N by d_in:

    A = U V^T
    Y = X W_slow^T + b + gamma (X V) U^T

U has shape d_out by r; V has shape d_in by r. The pretrained/base layer is frozen. A scalar gamma in [0,1] controls the adapter; it is configured, not meta-learned. gamma=0 restores the base output exactly and disables adaptation.

A dense elementwise gate from the original proposal is intentionally not used: a general Hadamard gate can destroy both low-rank structure and a previously established protected-subspace invariant.

Given explicit verified target activations T, define:

    E = T - Y
    L = ||E||_F^2 / (2N)
    P = I - Q Q^T
    A_candidate = (1-lambda) A + eta gamma E^T X P / N

The update is a projected delta rule for this local squared-error objective, followed by rank truncation and a Frobenius-norm cap. It is not a meta-gradient. In a real language model, obtaining useful target activations or a validated token-loss update remains an unsolved integration requirement. An unverified model answer must not be treated automatically as a correct target.

Forward evaluation never writes to the adapter. The caller invokes `adapt(X,T)` explicitly after deciding that a learning example is suitable. This also avoids accidental repeated adaptation during generation or tool retries.

## 3. What is actually protected

Q is an orthonormal basis in R^(d_in x k). For a fixed Q, if A_0 Q=0 and every update is right-projected by P, then A_t Q=0 in exact arithmetic. Thus the adapter leaves the output of this linear layer unchanged on inputs in span(Q).

The implementation reprojects the right factors after compression. Floating-point residuals are measured, not reported as exact zeros. The invariant does not imply preservation of every behavior of a nonlinear network, especially when upstream layers or Q change. Loading into a new architecture requires a compatible basis shape and matching state.

### Corrected Fisher statement

For a full parameter vector theta with p parameters, its Fisher matrix is p by p. A Fisher projection therefore acts on vec(delta A) in parameter space; it is not interchangeable with right-projecting input features without further assumptions.

For positive semidefinite F with eigenvalues mu_1 >= ... >= mu_p, let V_k contain its leading k eigenvectors. If delta is orthogonal to these vectors, then:

    delta^T F delta <= mu_(k+1) ||delta||^2.

This is zero only if the residual eigenvalues are zero. The complement of the leading eigenspace is generally a low-curvature space, not an exact nullspace.

The actual local loss expansion is:

    L(theta+delta)-L(theta) = grad(L)^T delta + 1/2 delta^T H delta + R(delta).

Dropping the linear term requires an additional stationarity/orthogonality assumption. Replacing Hessian H by Fisher F requires additional justification. A cubic remainder requires suitable local smoothness bounds. Repeated finite updates can leave the region where a local approximation is accurate. The original zero-forgetting theorem is therefore withdrawn.

The supplied adapter uses input-subspace protection only. It does not estimate Fisher information.

## 4. Efficient low-rank update implementation

Represent the candidate update as L R^T, with m=r+N columns:

    L = [sqrt(1-lambda) U, sqrt(eta gamma/N) E^T]
    R = [sqrt(1-lambda) P V, sqrt(eta gamma/N) P X^T]

Thin QR factorizations L=Q_L R_L and R=Q_R R_R reduce the SVD to the core R_L R_R^T. Keep its leading r singular values, scale them when their norm exceeds M_max, and reconstruct the two factors.

The additional forward cost is O(N r (d_in+d_out)). State storage is O(r(d_in+d_out)+k d_in). For m small compared with the dimensions, update compression costs approximately O((d_in+d_out)m^2 + m^3), plus projection costs. Large batches can make the core large. This is not O(1) in layer size or batch size.

No permanent GPU SRAM residency, fused kernel, or asynchronous update speedup has been demonstrated. FlashPlastic remains a possible future optimization project. The Oja-style differential equation in the original draft also lacks a demonstrated discrete-time stability guarantee; the supplied implementation uses an explicit norm cap instead.

## 5. Measured experiment

Run `python test_plastic_adapter.py` with PyTorch installed, from the directory containing the supplied files.

Configuration: CPU, one PyTorch thread, float64; a frozen 32-input/16-output affine layer; rank 8; four protected directions; eight orthogonal adaptation directions; 160 supervised updates per seed. Twenty held-out inputs are new linear combinations of those same eight directions. Target corrections lie within available rank capacity by construction. This is an easy realizable regression problem, not natural-language generalization.

| Seed | Held-out MSE before | Held-out MSE after | Max protected-output change |
|---|---:|---:|---:|
| 0 | 0.254999 | 2.74e-10 | 3.47e-17 |
| 1 | 0.291153 | 3.13e-10 | 2.78e-17 |
| 2 | 0.227038 | 2.44e-10 | 2.78e-17 |
| 3 | 0.413457 | 4.44e-10 | 9.02e-17 |
| 4 | 0.501408 | 5.38e-10 | 3.47e-17 |

In all five seeds the base state is unchanged and a state_dict roundtrip reproduces outputs exactly on the same CPU setup. Additional tests cover equality with a dense update reference, norm limits, read-only forward calls, rejection of nonfinite input, and reset. The state_dict roundtrip is not an atomic on-disk persistence service.

These are local measured results, not evidence for >95% episodic recall, 100-day retention, 10,000 sequential tasks, language-model quality, or GPU throughput.

## 6. Samantha integration boundary

The current HTTP proxy remains a sidecar. The supplied adapter is integrated only into a small PyTorch model in the test. Sending requests through port 1235 does not insert this layer into LM Studio's loaded model. Existing checkpoints are not migrated into the research adapter.

A future LLM experiment must provide a compatible model runtime with direct layer access, identify one specific linear layer, define where trustworthy adaptation targets originate, and compare frozen, retrieval-only, unprotected-adapter and protected-adapter conditions. Quantized/custom fused layers are not drop-in nn.Linear replacements. KV-cache invalidation and update boundaries must be specified before changing weights during decoding.

Do not auto-install this experimental adapter into the running assistant. First measure adaptation and retention on a separate checkpoint, with session/user isolation, rollback, contaminated-source tests and explicit learning policies.

## 7. Falsifiable next experiments

1. Capacity and interference: sequential targets beyond rank capacity; compare no protection against protected subspaces and measure both new-task error and retained-task error.
2. Generalization: held-out inputs outside the adaptation span and independently generated task families.
3. Language adaptation: new verified associations absent from the evaluation prompt, compared with identical retrieval and context budgets. Report accuracy, calibration, retention, memory size and latency across seeds.
4. Incorrect updates: false or contradictory observations, rejection mechanisms and rollback. Distinguish verified source facts from the assistant's own hypotheses.
5. Compute: realistic batch sizes, model dimensions, devices and end-to-end throughput; benchmark QR-core compression against dense SVD.

Plasticity is a testable mechanism for adaptation. Neither persistent state nor behavioral change alone establishes consciousness, agency or unlimited memory.
