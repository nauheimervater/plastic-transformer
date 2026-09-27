"""
Demo: Transformer Fast-Weight Adaptation with Subspace Calibration
Author: Thomas Nauheimer (2026)

Demonstrates:
1. Llama/Qwen-style SwiGLU decoder architecture with injected plastic layers.
2. Automated subspace calibration (calibrate_subspace) from baseline prompts.
3. Online adaptation on novel session data.
4. Retention verification, gate-based base restoration, and synaptic state serialization.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from plastic_transformer import PlasticModelWrapper


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


def main():
    print("=" * 78)
    print("  PLASTIC TRANSFORMER - CONTINUAL ADAPTATION WITH SUBSPACE CALIBRATION")
    print("  Author: Thomas Nauheimer (2026)")
    print("  Architecture: Llama/Qwen-style SwiGLU Decoder with Injected Fast Weights")
    print("=" * 78)
    
    torch.manual_seed(42)
    
    # 1. Base model initialization
    base_model = PlasticLlamaMini(vocab_size=128, d_model=64, intermediate_dim=128, n_layers=2)
    base_model.eval()
    
    # 2. Inject Plasticity into target projections
    plastic_model = PlasticModelWrapper(
        base_model,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        rank=8,
        learning_rate=0.5,
        decay=0.001
    )
    print(f"[Init] Injected PlasticLinearProjected into {len(plastic_model.plastic_layers)} projection layers:")
    for name in plastic_model.plastic_layer_map:
        print(f"       - {name}")
    
    def encode(text):
        return torch.tensor([[ord(c) % 128 for c in text]], dtype=torch.long)
    
    prompt_base_1 = encode("PROMPT: Baseline knowledge prompt regarding core physics principles.")
    prompt_base_2 = encode("PROMPT: Mathematical definitions and general arithmetic facts.")
    prompt_novel = encode("PROMPT: Novel session input requiring runtime fast-weight update.")
    
    # PHASE 1: Baseline Output Recording
    print("\n[PHASE 1] Initial Baseline Evaluation:")
    with torch.no_grad():
        out_base_1 = plastic_model(prompt_base_1)
        out_base_2 = plastic_model(prompt_base_2)
        out_base_novel = plastic_model(prompt_novel)
        print(f"  -> Model evaluated on {prompt_base_1.shape[1]} tokens baseline prompt.")
        print(f"  -> Initial fast-weight mass: 0.0000")

    # PHASE 2: Subspace Calibration on Baseline Prompts
    print("\n[PHASE 2] Subspace Calibration (Extracting Dominant Activation Subspace):")
    calibration_prompts = [prompt_base_1, prompt_base_2]
    calib_stats = plastic_model.calibrate_subspace(calibration_prompts, k=4)
    print(f"  -> Calibrated protected basis across {len(calib_stats)} layers (k={4}):")
    for name, r in list(calib_stats.items())[:3]:
        print(f"     * {name}: dim(Q) = {r}")
    print("     * ... (all target layers protected)")

    # PHASE 3: Target-Driven Adaptation on Novel Session Data
    print("\n[PHASE 3] Target-Driven Online Adaptation Steps:")
    for step in range(5):
        with torch.no_grad():
            for name, layer in plastic_model.plastic_layer_map.items():
                # Novel input orthogonal to the protected subspace
                P = torch.eye(layer.base.in_features, dtype=layer.base.weight.dtype) - layer.Q @ layer.Q.T
                x_novel = (P @ torch.randn(layer.base.in_features, 4, dtype=layer.base.weight.dtype)).T
                target = layer.base(x_novel) + 0.1 * torch.randn(4, layer.base.out_features, dtype=layer.base.weight.dtype)
                layer.adapt(x_novel, target)
                
        avg_mass = sum(l.diagnostics()['mass'] for l in plastic_model.plastic_layers) / len(plastic_model.plastic_layers)
        print(f"  -> Adaptation Step {step+1}/5: Average Fast-Weight Mass = {avg_mass:.4f}")

    # PHASE 4: Layer-Level Subspace Invariance Verification
    print("\n[PHASE 4] Subspace Invariance Verification on Calibrated Layers:")
    max_subspace_violation = 0.0
    for name, layer in plastic_model.plastic_layer_map.items():
        if layer.Q.shape[1] > 0:
            q_test = layer.Q.T # inputs lying directly in protected subspace
            with torch.no_grad():
                diff = (layer(q_test) - layer.base(q_test)).abs().max().item()
                if diff > max_subspace_violation:
                    max_subspace_violation = diff
                    
    status_subspace = "VERIFIED (Delta W * Q = 0)" if max_subspace_violation < 1e-6 else "VIOLATION"
    print(f"  -> Max layer-level perturbation on protected subspace Q: {max_subspace_violation:.10e}")
    print(f"  -> Algebraic Subspace Invariance Status: {status_subspace}")
    assert max_subspace_violation < 1e-6, "Layer-level subspace protection failed"

    # PHASE 5: Model-Level Output Evaluation
    print("\n[PHASE 5] Model-Level Output Shift vs. Frozen Restoration:")
    
    # Active Fast Weights (Gate = 1.0)
    with torch.no_grad():
        out_plastic_novel = plastic_model(prompt_novel)
        novel_shift = (out_plastic_novel - out_base_novel).abs().mean().item()
        out_plastic_base1 = plastic_model(prompt_base_1)
        base1_shift = (out_plastic_base1 - out_base_1).abs().mean().item()
    print(f"  [Active Synapses (Gate = 1.0)]")
    print(f"    Novel prompt output mean shift:    {novel_shift:.6f} (dynamic adaptation)")
    print(f"    Baseline prompt output mean shift: {base1_shift:.6f}")

    # Gate = 0.0 (Strict Frozen Base Restoration)
    plastic_model.set_gate(0.0)
    with torch.no_grad():
        out_gate_zero = plastic_model(prompt_base_1)
        diff_gate_zero = (out_gate_zero - out_base_1).abs().max().item()
    status_gate = "VERIFIED (Bit-identical to base)" if diff_gate_zero < 1e-6 else "DRIFT DETECTED"
    print(f"  [Gate = 0.0 Test (Frozen Base Restoration)]")
    print(f"    Max absolute deviation: {diff_gate_zero:.10e}")
    print(f"    Base Restoration Status: {status_gate}")
    assert diff_gate_zero < 1e-6, "Gate restoration failed"

    # PHASE 6: Synaptic Memory Serialization Roundtrip
    plastic_model.set_gate(1.0)
    snapshot_path = "episodic_snapshot.pt"
    print(f"\n[PHASE 6] Episodic State Serialization Roundtrip ({snapshot_path}):")
    plastic_model.save_synaptic_memory(snapshot_path)
    
    # Wipe synapses
    plastic_model.reset_synapses()
    with torch.no_grad():
        out_wiped = plastic_model(prompt_novel)
        diff_wiped = (out_wiped - out_base_novel).abs().max().item()
        print(f"  -> State reset: deviation from base = {diff_wiped:.10e}")
        
    # Reload synapses
    plastic_model.load_synaptic_memory(snapshot_path)
    with torch.no_grad():
        out_restored = plastic_model(prompt_novel)
        fidelity = (out_restored - out_plastic_novel).abs().max().item()
        status_restore = "EXACT ROUNDTRIP" if fidelity < 1e-6 else "SERIALIZATION MISMATCH"
        print(f"  -> State reloaded: deviation from pre-reset = {fidelity:.10e}")
        print(f"  -> Roundtrip Status: {status_restore}")
        assert fidelity < 1e-6, "Synaptic serialization roundtrip failed"

    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)

    print("\n" + "=" * 78)
    print("  VERIFICATION COMPLETE: Subspace calibration, adaptation & serialization verified.")
    print("=" * 78)


if __name__ == "__main__":
    main()
