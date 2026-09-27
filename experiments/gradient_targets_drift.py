"""
Experiment 1: Gradient-derived layer targets and model-level drift.

Question
--------
Does protecting a calibrated input subspace span(Q) per layer limit the drift of the
*model's* output distribution, and what does it cost in learning capacity?

Mechanism
---------
For each plastic layer with output Y and token loss L (cross-entropy on answer tokens):

    T = Y - lr * N / mean(||x||^2) * dL/dY

With this target, E = T - Y is proportional to -dL/dY, so PlasticLinearProjected.adapt()
performs a projected gradient step (NLMS-normalized) on the fast-weight matrix A = U V^T:

    Delta A  =  -lr * (dL/dY)^T X P / mean(||x||^2),   P = I - Q Q^T

(followed by the adapter's rank-r truncation and mass cap).

Base weights stay frozen; gradients only flow through activations.
In other words: this is gradient projection (GPM-style) applied to a low-rank
inference-time state. The experiment measures whether that is useful, not whether it is new.

What is measured (gate = 0 is the untouched base model, gate = 1 the adapted model)
--------------------------------------------------------------------------------
- recall:      top-1 exact match of the answer span for the trained question, fresh context
- paraphrase:  same for a paraphrased question that was never trained (generalization)
- retention:   recall of earlier sessions after later sessions were learned
- KL drift:    mean KL(p_base || p_adapted) per token on control texts,
               separately for calibration texts and held-out control texts
- d_ppl:       perplexity change on held-out control texts

Usage
-----
    # smoke test without downloads (random tiny Qwen2, byte tokenizer)
    python experiments/gradient_targets_drift.py --tiny --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
        --k-sweep 0,8,24,48 --epochs 15 --max-mass 50

    # real run
    python experiments/gradient_targets_drift.py --model Qwen/Qwen2.5-0.5B \
        --targets down_proj --layers all --k-sweep 0,16,64,256 --seeds 0,1,2 \
        --control-file control.txt

Author: Thomas Nauheimer (2026)
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from plastic_transformer import PlasticLinearProjected


# --------------------------------------------------------------------------------------
# Data: fictional facts (unknowable for any pretrained model) and control texts
# --------------------------------------------------------------------------------------

ENTITIES = [
    "Zorvex Industries", "Marlowe Kettleworks", "Brenquist Analytics", "Oradel Shipping",
    "Tessivar Labs", "Quennick Foods", "Halvorsk Mining", "Ypsera Textiles",
    "Duvanne Optics", "Krellmoor Energy", "Vashti Robotics", "Ellumbra Press",
]
CITIES = ["Kellbrunn", "Ostravel", "Pimberlake", "Varnholt", "Sudrenne", "Talmouth",
          "Gradisca Nova", "Wennick", "Lorquay", "Fennmark", "Istrova", "Calderwick"]
FOUNDERS = ["Mira Tolland", "Oskar Venhaal", "Lidia Prask", "Tomas Eberlund",
            "Nadja Querrow", "Felix Ambersen", "Ines Kovalt", "Ruben Stallor",
            "Greta Mondhal", "Anton Brevik", "Selma Durrach", "Jonas Keppler"]
PRODUCTS = ["brass valves", "glass kettles", "tide sensors", "wool sails",
            "lens coatings", "rye crackers", "cobalt wire", "silk rope",
            "prism filters", "salt batteries", "gear pumps", "map inks"]

RELATIONS = [
    ("headquarters", CITIES,
     "Q: Where is the headquarters of {e}?\nA:",
     "Q: In which city is {e} based?\nA:"),
    ("founder", FOUNDERS,
     "Q: Who founded {e}?\nA:",
     "Q: Who is the founder of {e}?\nA:"),
    ("product", PRODUCTS,
     "Q: What is the main product of {e}?\nA:",
     "Q: What does {e} mainly produce?\nA:"),
]

CONTROL_TEXTS = [
    "The water cycle describes how water evaporates, condenses into clouds and falls as rain.",
    "Paris is the capital of France and lies on the river Seine.",
    "def add(a, b):\n    return a + b\n\nprint(add(2, 3))",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "Der Rhein fließt von den Alpen bis zur Nordsee und ist eine wichtige Wasserstraße.",
    "The mitochondria is often called the powerhouse of the cell.",
    "In 1969, astronauts landed on the Moon for the first time.",
    "A prime number has exactly two distinct divisors: one and itself.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Shakespeare wrote plays such as Hamlet, Macbeth and King Lear.",
    "Newton's second law states that force equals mass times acceleration.",
    "Die Photosynthese findet in den Chloroplasten der Pflanzenzellen statt.",
    "SELECT name, age FROM users WHERE age > 30 ORDER BY name;",
    "The Pacific Ocean is the largest and deepest ocean on Earth.",
    "Water boils at 100 degrees Celsius at sea level.",
    "A sonnet is a poem of fourteen lines with a fixed rhyme scheme.",
]


def build_facts(seed, n_facts):
    rng = random.Random(seed)
    facts = []
    for i, enumerate_entity in enumerate(ENTITIES):
        rel_name, pool, q, p = RELATIONS[i % len(RELATIONS)]
        facts.append({"entity": enumerate_entity, "relation": rel_name, "pool": pool, "q": q, "p": p})
    # assign answers randomly per seed so no fixed entity-answer prior can exist
    for rel_name, pool, _, _ in RELATIONS:
        answers = pool[:]
        rng.shuffle(answers)
        for f in facts:
            if f["relation"] == rel_name:
                f["answer"] = " " + answers.pop()
    rng.shuffle(facts)
    out = []
    for f in facts[:n_facts]:
        out.append({
            "question": f["q"].format(e=f["entity"]),
            "paraphrase": f["p"].format(e=f["entity"]),
            "answer": f["answer"],
        })
    return out


# --------------------------------------------------------------------------------------
# Model / tokenizer
# --------------------------------------------------------------------------------------

class ByteTokenizer:
    """Minimal byte-level tokenizer for the offline smoke test."""
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))


class HFTokenizer:
    def __init__(self, tok):
        self.tok = tok

    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids


def load_model(args):
    if args.tiny:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(1234)
        cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=512, tie_word_embeddings=False,
                          initializer_range=0.2)
        model = Qwen2ForCausalLM(cfg)
        tok = ByteTokenizer()
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        tok = HFTokenizer(AutoTokenizer.from_pretrained(args.model))
    model.to(args.device).eval()
    model.requires_grad_(False)
    return model, tok


def parse_layers(spec):
    if spec == "all":
        return None
    ids = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return ids


def inject_adapters(model, targets, layer_ids, rank, max_mass):
    """Replace selected nn.Linear modules by PlasticLinearProjected. Returns {name: adapter}."""
    candidates = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            if child_name not in targets or not isinstance(child, nn.Linear):
                continue
            full = f"{parent_name}.{child_name}" if parent_name else child_name
            m = re.search(r"layers\.(\d+)\.", full)
            if layer_ids is not None and (m is None or int(m.group(1)) not in layer_ids):
                continue
            candidates.append((parent, child_name, child, full))
    adapters = {}
    for parent, child_name, child, full in candidates:
        adapter = PlasticLinearProjected(child, rank=rank, learning_rate=1.0, max_mass=max_mass)
        setattr(parent, child_name, adapter)
        adapters[full] = adapter
    if not adapters:
        raise RuntimeError(f"No nn.Linear modules matched targets={targets}")
    return adapters


# --------------------------------------------------------------------------------------
# Activation capture
# --------------------------------------------------------------------------------------

class Capture:
    """Records inputs X and outputs Y of all adapters while enabled."""

    def __init__(self, adapters):
        self.enabled = False
        self.X, self.Y = {}, {}
        for name, layer in adapters.items():
            layer.register_forward_pre_hook(self._pre(name))
            layer.register_forward_hook(self._post(name))

    def _pre(self, name):
        def hook(mod, inp):
            if self.enabled:
                self.X[name] = inp[0]
        return hook

    def _post(self, name):
        def hook(mod, inp, out):
            if self.enabled:
                self.Y[name] = out
        return hook

    def clear(self):
        self.X, self.Y = {}, {}


def set_gate(adapters, value):
    for layer in adapters.values():
        layer.gate.fill_(value)


def reset_adapters(adapters):
    for layer in adapters.values():
        layer.reset()
        layer.set_protected_subspace(None)


# --------------------------------------------------------------------------------------
# Calibration, learning, evaluation
# --------------------------------------------------------------------------------------

def encode_ids(tok, text, device, max_len=128):
    ids = tok.encode(text)[:max_len]
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def calibrate(model, tok, adapters, capture, texts, k, device):
    """Set Q per layer to the top-k right singular vectors of its calibration inputs."""
    if k <= 0:
        for layer in adapters.values():
            layer.set_protected_subspace(None)
        return {name: 0 for name in adapters}
    rows = {name: [] for name in adapters}
    capture.enabled = True
    for text in texts:
        capture.clear()
        model(input_ids=encode_ids(tok, text, device))
        for name in adapters:
            x = capture.X[name]
            rows[name].append(x.reshape(-1, x.shape[-1]).detach())
    capture.enabled = False
    capture.clear()
    dims = {}
    for name, layer in adapters.items():
        X = torch.cat(rows[name]).to(layer.base.weight.dtype)
        kk = min(k, X.shape[0], X.shape[1])
        if kk < k and name == next(iter(adapters)):
            print(f"  [warn] k={k} limited to {kk} by {X.shape[0]} calibration tokens / d_in={X.shape[1]}; "
                  f"use --control-file with more text")
        _, _, vh = torch.linalg.svd(X, full_matrices=False)
        layer.set_protected_subspace(vh[:kk].T)
        dims[name] = layer.Q.shape[1]
    return dims


def learn_fact(model, tok, adapters, capture, fact, lr, device):
    """One gradient-target adaptation step on a question/answer pair."""
    p_ids = tok.encode(fact["question"])
    a_ids = tok.encode(fact["answer"])
    ids = torch.tensor([p_ids + a_ids], dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    labels[0, len(p_ids):] = ids[0, len(p_ids):]

    embed = model.get_input_embeddings()
    handle = embed.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    try:
        with torch.enable_grad():
            capture.clear()
            capture.enabled = True
            logits = model(input_ids=ids).logits
            capture.enabled = False
            loss = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
            names = list(capture.Y)
            grads = torch.autograd.grad(loss, [capture.Y[n] for n in names])
    finally:
        handle.remove()
        capture.enabled = False

    with torch.no_grad():
        for name, g in zip(names, grads):
            layer = adapters[name]
            X = capture.X[name].detach().reshape(-1, layer.base.in_features)
            Y = capture.Y[name].detach().reshape(-1, layer.base.out_features)
            G = g.reshape(-1, layer.base.out_features)
            mask = G.norm(dim=1) > 0
            if not mask.any():
                continue
            X, Y, G = X[mask], Y[mask], G[mask]
            # NLMS scaling: adapt() applies (1/N) E^T X P, so with E = -step * G this
            # gives Delta A = -lr * G^T X P / mean(||x||^2), i.e. a projected gradient
            # step on A that is invariant to the activation scale of this layer.
            step = lr * X.shape[0] / X.pow(2).sum(dim=1).mean().clamp_min(1e-12)
            layer.adapt(X, Y - step * G)
    capture.clear()
    return loss.item()


@torch.no_grad()
def answer_stats(model, tok, prompt, answer, device):
    p_ids = tok.encode(prompt)
    a_ids = tok.encode(answer)
    ids = torch.tensor([p_ids + a_ids], dtype=torch.long, device=device)
    logits = model(input_ids=ids).logits[0]
    pos = torch.arange(len(p_ids) - 1, len(p_ids) + len(a_ids) - 1, device=device)
    tgt = torch.tensor(a_ids, device=device)
    logp = F.log_softmax(logits[pos], dim=-1)
    top1 = logp.argmax(-1) == tgt
    return {"logprob": logp[torch.arange(len(a_ids)), tgt].mean().item(),
            "exact": float(top1.all().item())}


def eval_facts(model, tok, facts, device):
    rec = [answer_stats(model, tok, f["question"], f["answer"], device) for f in facts]
    par = [answer_stats(model, tok, f["paraphrase"], f["answer"], device) for f in facts]
    mean = lambda xs, key: sum(x[key] for x in xs) / max(len(xs), 1)
    return {"recall": mean(rec, "exact"), "recall_logprob": mean(rec, "logprob"),
            "paraphrase": mean(par, "exact"), "paraphrase_logprob": mean(par, "logprob")}


@torch.no_grad()
def drift(model, tok, adapters, texts, device):
    """Mean per-token KL(p_base || p_adapted) and perplexities, gate 0 vs gate 1."""
    kls, nll0, nll1, n = 0.0, 0.0, 0.0, 0
    for text in texts:
        ids = encode_ids(tok, text, device)
        set_gate(adapters, 0.0)
        l0 = model(input_ids=ids).logits[0].float()
        set_gate(adapters, 1.0)
        l1 = model(input_ids=ids).logits[0].float()
        lp0, lp1 = F.log_softmax(l0, -1), F.log_softmax(l1, -1)
        kls += (lp0.exp() * (lp0 - lp1)).sum(-1).sum().item()
        tgt = ids[0, 1:]
        nll0 += F.nll_loss(lp0[:-1], tgt, reduction="sum").item()
        nll1 += F.nll_loss(lp1[:-1], tgt, reduction="sum").item()
        n_tok = ids.shape[1]
        n += n_tok
    n_pred = n - len(texts)
    return {"kl": kls / n,
            "ppl_base": math.exp(nll0 / n_pred),
            "ppl_adapted": math.exp(nll1 / n_pred)}


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------

def run_condition(model, tok, adapters, capture, facts, k, args, calib_texts, heldout_texts):
    reset_adapters(adapters)
    set_gate(adapters, 1.0)
    q_dims = calibrate(model, tok, adapters, capture, calib_texts, k, args.device)

    sessions = [facts[i::args.sessions] for i in range(args.sessions)]
    retention = []  # retention[s] = recall of each learned session after session s
    t0 = time.perf_counter()
    for s, session in enumerate(sessions):
        for _ in range(args.epochs):
            for fact in session:
                learn_fact(model, tok, adapters, capture, fact, args.lr, args.device)
        retention.append([eval_facts(model, tok, sessions[j], args.device)["recall"]
                          for j in range(s + 1)])
    learn_time = time.perf_counter() - t0

    adapted = eval_facts(model, tok, facts, args.device)
    d_calib = drift(model, tok, adapters, calib_texts, args.device)
    d_held = drift(model, tok, adapters, heldout_texts, args.device)
    masses = [layer.diagnostics()["mass"] for layer in adapters.values()]
    interference = max(layer.diagnostics()["protected_interference"] for layer in adapters.values())
    d_in = {name: layer.base.in_features for name, layer in adapters.items()}
    return {
        "k": k,
        "q_fraction_mean": sum(q_dims[n] / d_in[n] for n in adapters) / len(adapters),
        "adapted": adapted,
        "kl_calib": d_calib["kl"],
        "kl_heldout": d_held["kl"],
        "ppl_heldout_base": d_held["ppl_base"],
        "ppl_heldout_adapted": d_held["ppl_adapted"],
        "retention": retention,
        "mass_mean": sum(masses) / len(masses),
        "mass_capped_layers": sum(m >= args.max_mass * 0.999 for m in masses),
        "max_layer_interference": interference,
        "learn_seconds": learn_time,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--tiny", action="store_true", help="offline smoke test with a random tiny Qwen2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--targets", default="down_proj", help="comma-separated module names")
    ap.add_argument("--layers", default="all", help="'all' or e.g. '4-15' or '2,5,9'")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-mass", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=0.1,
                    help="NLMS step size for the projected gradient step on A")
    ap.add_argument("--k-sweep", default="0,16,64,256", help="protected dims per layer; 0 = no protection")
    ap.add_argument("--n-facts", type=int, default=12)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=10, help="passes over each session's facts")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--control-file", default=None,
                    help="optional text file, one control text per blank-line-separated block")
    ap.add_argument("--out", default="results_gradient_targets_drift.json")
    args = ap.parse_args()

    if args.tiny and args.model == ap.get_default("model"):
        args.model = "tiny-random-qwen2"
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    ks = [int(k) for k in args.k_sweep.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    texts = CONTROL_TEXTS
    if args.control_file:
        texts = [b.strip() for b in Path(args.control_file).read_text(encoding="utf-8").split("\n\n") if b.strip()]
    calib_texts, heldout_texts = texts[0::2], texts[1::2]

    model, tok = load_model(args)
    adapters = inject_adapters(model, targets, parse_layers(args.layers), args.rank, args.max_mass)
    capture = Capture(adapters)
    print(f"# Gradient targets & drift | model={args.model} | {len(adapters)} plastic layers "
          f"({','.join(targets)}) | rank={args.rank} | lr={args.lr} | device={args.device}")

    results = {"config": vars(args), "runs": []}
    for seed in seeds:
        torch.manual_seed(seed)
        facts = build_facts(seed, args.n_facts)
        reset_adapters(adapters)
        set_gate(adapters, 0.0)
        base = eval_facts(model, tok, facts, args.device)
        set_gate(adapters, 1.0)
        print(f"\n## Seed {seed} | base model (gate=0): recall={base['recall']:.2f} "
              f"paraphrase={base['paraphrase']:.2f} logprob={base['recall_logprob']:.2f}\n")
        print("| k | dim(Q)/d_in | recall | paraphrase | d_logprob | KL calib | KL held-out | "
              "ppl held-out (base->adapted) | retention first session | mass (capped) |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in ks:
            r = run_condition(model, tok, adapters, capture, facts, k, args, calib_texts, heldout_texts)
            r["seed"], r["base"] = seed, base
            results["runs"].append(r)
            a = r["adapted"]
            first = " -> ".join(f"{row[0]:.2f}" for row in r["retention"])
            print(f"| {k} | {r['q_fraction_mean']:.3f} | {a['recall']:.2f} | {a['paraphrase']:.2f} | "
                  f"{a['recall_logprob'] - base['recall_logprob']:+.2f} | {r['kl_calib']:.2e} | "
                  f"{r['kl_heldout']:.2e} | {r['ppl_heldout_base']:.2f} -> {r['ppl_heldout_adapted']:.2f} | "
                  f"{first} | {r['mass_mean']:.2f} ({r['mass_capped_layers']}) |")
            sys.stdout.flush()

    reset_adapters(adapters)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {args.out}")
    print("Note: layer-level protection is exact only for inputs inside span(Q); "
          "KL on held-out texts is the model-level quantity that matters.")


if __name__ == "__main__":
    main()
