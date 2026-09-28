"""
Demo: fast-weight learning, subspace protection and consolidation in a small Llama-style decoder.
Author: Thomas Nauheimer (2026)

A randomly initialised 2-layer SwiGLU decoder (14 plastic projections) learns character
sequences through PlasticModelWrapper.gradient_step. Everything printed is measured; the demo
makes no claim beyond this toy model.

Part 1  Protected dimension k vs. drift on the calibration prompt and learning on a new prompt.
Part 2  Two conflicting sessions: forgetting of session A with and without consolidate + expand.
Part 3  gate = 0 restores the base model exactly; synaptic memory roundtrip.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from plastic_transformer import PlasticModelWrapper

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class LlamaStyleAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)


class LlamaStyleMLP(nn.Module):
    """SwiGLU MLP as used in Llama-3 and Qwen architectures."""
    def __init__(self, d_model=64, intermediate_dim=128):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaStyleBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, intermediate_dim=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LlamaStyleAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = LlamaStyleMLP(d_model, intermediate_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PlasticLlamaMini(nn.Module):
    def __init__(self, vocab_size=128, d_model=64, intermediate_dim=128, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            LlamaStyleBlock(d_model=d_model, intermediate_dim=intermediate_dim) 
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))



def encode(text):
    return torch.tensor([[ord(c) % 128 for c in text]], dtype=torch.long)


def seq_loss(model, ids):
    logits = model(ids)
    return F.cross_entropy(logits[0, :-1], ids[0, 1:])


def span_loss(model, ids, start, length):
    """Cross-entropy on the tokens ids[start:start+length] only (e.g. the digits of a code)."""
    logits = model(ids)
    return F.cross_entropy(logits[0, start - 1:start + length - 1], ids[0, start:start + length]).item()


@torch.no_grad()
def kl_to(model, ids, ref_logits):
    lp_ref = F.log_softmax(ref_logits[0], -1)
    lp = F.log_softmax(model(ids)[0], -1)
    return (lp_ref.exp() * (lp_ref - lp)).sum(-1).mean().item()


def learn(plastic, ids, steps, lr):
    for _ in range(steps):
        with plastic.capture():
            loss = seq_loss(plastic, ids)
        plastic.gradient_step(loss, lr=lr)


def fresh(ref):
    model = PlasticLlamaMini()
    model.load_state_dict(ref.state_dict())
    model.eval()
    return PlasticModelWrapper(model, target_modules=TARGETS, rank=8, max_mass=100.0)


def main():
    torch.manual_seed(42)
    ref = PlasticLlamaMini().eval()
    base_prompt = encode("Core baseline text about quantum mechanics and relativity.")
    # B shares most of its prefix with A but continues differently, so learning B interferes with A
    session_a = encode("The code of the north door is 4711, said Ilse.")
    session_b = encode("The code of the south door is 9052, said Ilse.")
    with torch.no_grad():
        ref_base_logits = ref(base_prompt)

    print("=" * 78)
    print(" Plastic Transformer demo (random 2-layer Llama-style decoder, 14 plastic layers)")
    print("=" * 78)

    # Part 1 ------------------------------------------------------------------------------
    print("\n[Part 1] Protected dimension k (calibrated on the base prompt), then 30 learning steps on session A")
    print("| k | KL on base prompt | loss on A before -> after |")
    print("|---:|---:|---|")
    rows = []
    for k in [0, 2, 8, 32, 64]:
        plastic = fresh(ref)
        if k:
            plastic.calibrate_subspace([base_prompt], k=k)
        with torch.no_grad():
            before = seq_loss(plastic, session_a).item()
        learn(plastic, session_a, steps=30, lr=0.3)
        with torch.no_grad():
            after = seq_loss(plastic, session_a).item()
        kl = kl_to(plastic, base_prompt, ref_base_logits)
        kl = max(kl, 0.0)  # tiny negative values are float rounding
        rows.append((k, kl, before - after))
        print(f"| {k} | {kl:.2e} | {before:.3f} -> {after:.3f} |")
    k0, kmax = rows[0], rows[-1]
    drift_note = (f"falls by a factor {k0[1] / kmax[1]:.0f}" if kmax[1] > 1e-10 else "falls to float rounding level")
    print(f"Measured: KL on the base prompt {drift_note} from k=0 to k={kmax[0]}; the loss reduction on A "
          f"shrinks from {k0[2]:.3f} to {kmax[2]:.3f} (stability-plasticity trade-off).")
    print("Note: k is capped per layer by the number of calibration tokens and d_in. Layer-level")
    print("invariance is exact only for inputs inside span(Q); model-level drift is what is measured here.")

    # Part 2 ------------------------------------------------------------------------------
    print("\n[Part 2] Sequential sessions A then B (100 steps each); loss measured on the 4-digit codes only")
    code_at = "The code of the north door is 4711, said Ilse.".index("4711")
    part2 = {}
    for mode in ["no protection", "consolidate + expand_subspace"]:
        plastic = fresh(ref)
        learn(plastic, session_a, steps=100, lr=3.0)
        with torch.no_grad():
            a_after_a = span_loss(plastic, session_a, code_at, 4)
        if mode != "no protection":
            plastic.expand_subspace([session_a], energy_threshold=0.99)
        learn(plastic, session_b, steps=100, lr=3.0)
        with torch.no_grad():
            a_end = span_loss(plastic, session_a, code_at, 4)
            b_end = span_loss(plastic, session_b, code_at, 4)
        part2[mode] = (a_end - a_after_a, b_end)
        print(f"  {mode:<31} code A after A = {a_after_a:.3f} | code A after B = {a_end:.3f} "
              f"(forgetting {a_end - a_after_a:+.3f}) | code B = {b_end:.3f}")
    dims = [l.Q.shape[1] for l in plastic.plastic_layers]
    print(f"  dim(Q) per layer after expansion: min {min(dims)}, max {max(dims)}")
    (f_un, b_un), (f_pr, b_pr) = part2["no protection"], part2["consolidate + expand_subspace"]
    print(f"  Measured: forgetting of A {f_un:+.3f} -> {f_pr:+.3f}; loss on code B {b_un:.3f} -> {b_pr:.3f} "
          f"({'less' if b_pr > b_un else 'no less'} learning of B, whose inputs overlap strongly with A).")

    # Part 3 ------------------------------------------------------------------------------
    print("\n[Part 3] gate = 0 and synaptic memory roundtrip")
    plastic.set_gate(0.0)
    with torch.no_grad():
        gate_diff = (plastic(base_prompt) - ref_base_logits).abs().max().item()
    plastic.set_gate(1.0)
    print(f"  max |output(gate=0) - base output| = {gate_diff:.2e}")
    path = os.path.join(tempfile.mkdtemp(), "synapses.pt")
    plastic.save_synaptic_memory(path)
    with torch.no_grad():
        out = plastic(session_b)
    plastic.reset_synapses()
    plastic.load_synaptic_memory(path)
    with torch.no_grad():
        reload_diff = (plastic(session_b) - out).abs().max().item()
    os.remove(path)
    print(f"  max |output after reload - output before| = {reload_diff:.2e}")
    print(f"  persistent state: {plastic.state_bytes() / 1e3:.1f} kB fast weights "
          f"(+ {(plastic.state_bytes(include_q=True) - plastic.state_bytes()) / 1e3:.1f} kB Q)")


if __name__ == "__main__":
    main()
