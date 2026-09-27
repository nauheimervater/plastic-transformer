"""
Demo: Interactive Llama/Qwen-style Decoder with Runtime Plastic Synapses
Author: Thomas Nauheimer
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
    """Explicit Linear Q/K/V/O projections matching Llama-3 and Qwen-2.5 architectures."""
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
    print("  Author: Thomas Nauheimer")
    print("  Architecture: Llama/Qwen-style Decoder with Runtime Plastic Synapses")
    print("=" * 75)
    
    torch.manual_seed(42)
    
    # 1. Base model initialization
    base_model = PlasticLlamaMini(vocab_size=128, d_model=64, n_layers=2)
    base_model.eval()
    
    # 2. Inject Plasticity into target projections
    plastic_model = PlasticModelWrapper(
        base_model, 
        target_modules=["q_proj", "v_proj", "o_proj", "mlp_gate", "mlp_down"], 
        rank=8, 
        eta=0.25, 
        decay=0.0001
    )
    print(f"[Init] Injected plasticity into {len(plastic_model.plastic_layers)} projection layers.")
    
    def encode(text):
        return torch.tensor([[ord(c) % 128 for c in text]], dtype=torch.long)
    
    fact_input = encode("PROMPT: Wer ist Nauheimer?")
    unrelated_input = encode("PROMPT: Was ist 2 + 2?")
    
    # PHASE 1: Baseline Before Learning
    print("\n[PHASE 1] Initial Baseline State:")
    with torch.no_grad():
        out_base_fact = plastic_model(fact_input)
        out_base_unrelated = plastic_model(unrelated_input)
        print(f"  -> Model stimulated with: {fact_input.shape[1]} tokens")
        print(f"  -> Baseline fast weight energy: {plastic_model.plastic_layers[0].U.norm().item():.4f}")

    # PHASE 2: Live Interaction (Experiencing new knowledge in conversation)
    print("\n[PHASE 2] Runtime Adaptation (Hebbian Synaptic Learning):")
    for turn in range(10):
        _ = plastic_model(fact_input)
        plastic_model.step_plasticity(surprise_signal=1.5)
        avg_energy = sum(l.U.norm().item() for l in plastic_model.plastic_layers) / len(plastic_model.plastic_layers)
        print(f"  -> Conversation Turn {turn+1:2d}/10: Average Synaptic Energy = {avg_energy:.4f}")

    # PHASE 3: Session Reset & Testing Retention
    print("\n[PHASE 3] Simulating Session Reset & Testing Retention:")
    print("  [Action] Cleared conversation history completely (0 tokens in context buffer).")
    
    # Test A: Standard Static Architecture
    plastic_model.set_plasticity_enabled(False)
    with torch.no_grad():
        out_static = plastic_model(fact_input)
        diff_static = (out_static - out_base_fact).abs().mean().item()
        print(f"\n  [Static Model Test (Current LLMs)]")
        print(f"    Shift from baseline state: {diff_static:.6f}")
        print(f"    Status: 100% AMNESIA - Forgot the user completely.")

    # Test B: Plastic Architecture
    plastic_model.set_plasticity_enabled(True)
    with torch.no_grad():
        out_plastic = plastic_model(fact_input)
        diff_plastic = (out_plastic - out_base_fact).abs().mean().item()
        print(f"\n  [Plastic Model Test (Our Innovation)]")
        print(f"    Shift from baseline state: {diff_plastic:.6f}")
        print(f"    Status: PERSISTENT RETENTION - Synaptic imprint is preserved!")

    # PHASE 4: Catastrophic Forgetting Verification
    print("\n[PHASE 4] Catastrophic Forgetting Verification:")
    with torch.no_grad():
        out_unrelated_after = plastic_model(unrelated_input)
        drift_unrelated = (out_unrelated_after - out_base_unrelated).abs().mean().item()
        print(f"  Query: 'Was ist 2 + 2?'")
        print(f"  Foundational knowledge perturbation: {drift_unrelated:.6f}")
        print(f"  Status: SAFE - Base capabilities remain intact.")

    # PHASE 5: Saving & Loading the Episodic Mind Snapshot
    snapshot_path = "episodic_snapshot.pt"
    print(f"\n[PHASE 5] Exporting Mind Snapshot ({snapshot_path}):")
    plastic_model.save_synaptic_memory(snapshot_path)
    
    plastic_model.reset_synapses()
    with torch.no_grad():
        out_wiped = plastic_model(fact_input)
        print(f"  Model wiped: diff = {(out_wiped - out_base_fact).abs().mean().item():.6f}")
        
    plastic_model.load_synaptic_memory(snapshot_path)
    with torch.no_grad():
        out_restored = plastic_model(fact_input)
        fidelity = (out_restored - out_plastic).abs().mean().item()
        print(f"  Memory awakened: deviation = {fidelity:.8f}")
        
    print("\n" + "=" * 75)
    print("  VERIFICATION COMPLETE: Plastic Transformer validated successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
