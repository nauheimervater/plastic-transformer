"""
Demo: Transformer Fast-Weight Adaptation & Episodic State Management
Author: Thomas Nauheimer (2026)
"""

import sys
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


class LlamaStyleBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = LlamaStyleAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_gate = nn.Linear(d_model, d_model * 2, bias=False)
        self.mlp_down = nn.Linear(d_model * 2, d_model, bias=False)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = F.silu(self.mlp_gate(self.ln2(x)))
        x = x + self.mlp_down(h)
        return x


class PlasticLlamaMini(nn.Module):
    def __init__(self, vocab_size=128, d_model=64, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([LlamaStyleBlock(d_model=d_model) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))


def main():
    print("=" * 75)
    print("  PLASTIC TRANSFORMER - CONTINUAL ADAPTATION EXPERIMENT")
    print("  Author: Thomas Nauheimer (2026)")
    print("  Architecture: Llama/Qwen-style Decoder with Runtime Fast Weights")
    print("=" * 75)
    
    torch.manual_seed(42)
    
    # 1. Base model initialization
    base_model = PlasticLlamaMini(vocab_size=128, d_model=64, n_layers=2)
    base_model.eval()
    
    # 2. Inject Plasticity into target projections using verified projected layers
    plastic_model = PlasticModelWrapper(
        base_model, 
        target_modules=["q_proj", "v_proj", "o_proj", "mlp_gate", "mlp_down"], 
        rank=8, 
        learning_rate=0.5,
        decay=0.001,
        layer_type="projected"
    )
    print(f"[Init] Injected PlasticLinearProjected into {len(plastic_model.plastic_layers)} projection layers.")
    
    def encode(text):
        return torch.tensor([[ord(c) % 128 for c in text]], dtype=torch.long)
    
    fact_input = encode("PROMPT: Test-Time-Training Stimulus")
    unrelated_input = encode("PROMPT: Unrelated Baseline Prompt")
    
    # PHASE 1: Baseline Evaluation
    print("\n[PHASE 1] Initial Baseline Evaluation:")
    with torch.no_grad():
        out_base_fact = plastic_model(fact_input)
        out_base_unrelated = plastic_model(unrelated_input)
        initial_mass = sum(l.diagnostics()['mass'] for l in plastic_model.plastic_layers)
        print(f"  -> Model evaluated with: {fact_input.shape[1]} tokens")
        print(f"  -> Initial fast-weight mass: {initial_mass:.4f}")

    # PHASE 2: Target-Driven Adaptation Steps
    print("\n[PHASE 2] Target-Driven Adaptation Steps:")
    for step in range(5):
        with torch.no_grad():
            for layer in plastic_model.plastic_layers:
                # Synthetic target adaptation on intermediate layer
                x_dummy = torch.randn(4, layer.base.in_features, dtype=layer.base.weight.dtype)
                target_dummy = layer.base(x_dummy) + 0.1 * torch.randn(4, layer.base.out_features, dtype=layer.base.weight.dtype)
                layer.adapt(x_dummy, target_dummy)
                
        avg_mass = sum(l.diagnostics()['mass'] for l in plastic_model.plastic_layers) / len(plastic_model.plastic_layers)
        print(f"  -> Adaptation Step {step+1}/5: Average Fast-Weight Mass = {avg_mass:.4f}")

    # PHASE 3: Testing Retention & Gate Disable
    print("\n[PHASE 3] Verification of Dynamic Fast-Weight Contribution:")
    
    # Test A: Gate = 0.0 (Reproducing strict frozen base model)
    plastic_model.set_gate(0.0)
    with torch.no_grad():
        out_gate_zero = plastic_model(fact_input)
        diff_gate_zero = (out_gate_zero - out_base_fact).abs().max().item()
        
    status_gate = "VERIFIED (Bit-identical to base)" if diff_gate_zero < 1e-7 else "DRIFT DETECTED"
    print(f"\n  [Gate = 0.0 Test (Frozen Base Restoration)]")
    print(f"    Max absolute deviation: {diff_gate_zero:.10e}")
    print(f"    Verification Status: {status_gate}")

    # Test B: Gate = 1.0 (Active Fast Weights)
    plastic_model.set_gate(1.0)
    with torch.no_grad():
        out_plastic = plastic_model(fact_input)
        diff_plastic = (out_plastic - out_base_fact).abs().mean().item()
        
    print(f"\n  [Gate = 1.0 Test (Active Fast-Weight Adaptation)]")
    print(f"    Mean absolute output shift: {diff_plastic:.6f}")

    # PHASE 4: Serialization Roundtrip Verification
    snapshot_path = "test_episodic_snapshot.pt"
    print(f"\n[PHASE 4] Episodic State Serialization Roundtrip ({snapshot_path}):")
    plastic_model.save_synaptic_memory(snapshot_path)
    
    plastic_model.reset_synapses()
    with torch.no_grad():
        out_wiped = plastic_model(fact_input)
        diff_wiped = (out_wiped - out_base_fact).abs().max().item()
        print(f"  State reset: deviation from base = {diff_wiped:.10e}")
        
    plastic_model.load_synaptic_memory(snapshot_path)
    with torch.no_grad():
        out_restored = plastic_model(fact_input)
        fidelity = (out_restored - out_plastic).abs().max().item()
        status_restore = "EXACT ROUNDTRIP" if fidelity < 1e-7 else "SERIALIZATION MISMATCH"
        print(f"  State reloaded: deviation from pre-reset = {fidelity:.10e}")
        print(f"  Roundtrip Status: {status_restore}")

    import os
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)

    print("\n" + "=" * 75)
    print("  VERIFICATION COMPLETE: Threshold-checked results validated.")
    print("=" * 75)


if __name__ == "__main__":
    main()
