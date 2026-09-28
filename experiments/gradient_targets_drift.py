"""
Experiment 1+2: gradient-derived layer targets, model-level drift, and online subspace growth.

Question
--------
(1) Does protecting a calibrated input subspace span(Q) per layer limit drift of the *model's*
    output distribution, and what does it cost in learning capacity?
(2) Does growing Q online after each session (energy criterion, GPM-style) prevent forgetting of
    earlier sessions, and how fast does the learning capacity of later sessions collapse?

Mechanism
---------
For every plastic layer with input X, output Y and token loss L (cross-entropy on answer tokens):

    T = Y - lr * N / mean(||x||^2) * dL/dY

so PlasticLinearProjected.adapt() performs a projected, NLMS-normalized gradient step on A = U V^T:

    Delta A = -lr * (dL/dY)^T X P / mean(||x||^2),   P = I - Q Q^T

(followed by the adapter's rank-r truncation and mass cap). This is gradient projection in the
sense of GPM (Saha et al., 2021), applied to a low-rank inference-time state.

Consolidation (required for online Q growth)
--------------------------------------------
Projecting existing fast weights onto the complement of a grown Q erases what they learned on
exactly the newly protected inputs (verified: one further update with lr=1e-6 reset a learned fact
in library version <= 1.0.4). Rank truncation, decay and the mass cap would also alter A on span(Q).
Sessions therefore end with PlasticLinearProjected.consolidate(): the active factors are frozen into
a per-layer store that later updates never touch, the active part restarts at zero, and only then
Q is expanded. The store grows by r columns per session and layer. Since library version 1.1.0,
set_protected_subspace() refuses to run on nonzero active factors.

Conditions (--conditions, ';'-separated)
----------------------------------------
    static:k=16            fixed Q = top-k right singular vectors of control-text activations
    static:k=16:center     same, but SVD of mean-centered activations (tests the mean-direction hypothesis)
    static:k=16:consol     fixed Q plus per-session consolidation (separates truncation from interference)
    online:eps=0.97:base_eps=0.9   base Q from control texts by energy, then Q grows after each session
    online:eps=0.97:nobase         Q grows from session inputs only
    lora:seq                       B1: one LoRA adapter trained across all sessions, no protection
    lora:multi                     B2: new LoRA per session, old ones frozen (pure consolidation effect)
    lora:olora:lam=0.5             B3: B2 + O-LoRA penalty lam * sum ||A_new A_old^T||_F^2 (parameter space)
    lora:gpm:eps=0.97:base_eps=0.9 B4: B2 + hard projection A <- A P after every step, Q grown from session
                                   inputs exactly as in online:* (activation space, standard LoRA training)
    lora:replay                    B5: B1 + one replayed earlier fact per training step
    rag:k=4                        B6: lexical retrieval of top-k learned facts into the prompt, frozen model
    icl:all                        B6': all learned facts in the prompt, frozen model
Ranks are swept separately with --ranks (rag/icl are rank-independent and run once).

Metrics (gate = 0 is the untouched base model, gate = 1 the adapted model)
--------------------------------------------------------------------------
recall / paraphrase: exact top-1 answer span, fresh context (paraphrases are never trained)
logprob: mean answer-token log-probability (continuous, preferred over exact match)
gain S1 / gain last: logprob gain of a session right after it was learned (capacity over time)
forget S1: logprob of session 1 right after learning minus at the end (positive = forgotten)
KL held-out: mean per-token KL(p_base || p_adapted) on held-out control texts
             (for rag: KL on the control tokens with vs. without the retrieved context)
state MB/fact: persistent state per learned fact (adapter factors in fp32, Q excluded; rag/icl: fact text)
train passes: forward+backward passes over single facts during learning
All aggregate values are mean +- 95% CI (t-distribution) over seeds.

Usage
-----
    # offline smoke test (random tiny Qwen2, byte tokenizer) -- validates the pipeline only
    python experiments/gradient_targets_drift.py --tiny --seeds 0,1 \
        --targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --epochs 15 --max-mass 50

    # real run
    python experiments/gradient_targets_drift.py --model Qwen/Qwen2.5-0.5B --targets down_proj \
        --seeds 0,1,2,3,4 --ranks 8,32 --control-file control.txt

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

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from plastic_transformer import PlasticLinearProjected


DEFAULT_CONDITIONS = ";".join([
    "static:k=0",
    "static:k=1",
    "static:k=4",
    "static:k=16",
    "static:k=16:center",
    "static:k=64",
    "static:k=16:consol",
    "online:eps=0.90:base_eps=0.9",
    "online:eps=0.97:base_eps=0.9",
    "online:eps=0.99:base_eps=0.9",
    "online:eps=0.97:nobase",
])

# --------------------------------------------------------------------------------------
# Data: fictional facts (unknowable for any pretrained model) and control texts
# --------------------------------------------------------------------------------------

SYL_A = ["Zor", "Mar", "Bren", "Ora", "Tes", "Quen", "Hal", "Yps", "Duv", "Krel", "Vash", "Ellum",
         "Pry", "Gal", "Tov", "Nim", "Sar", "Wel", "Fen", "Ost", "Lum", "Cor", "Ivar", "Brask"]
SYL_B = ["vex", "lowe", "quist", "del", "sivar", "nick", "vorsk", "era", "anne", "moor", "ti",
         "bra", "ton", "ley", "rin", "dal", "mere", "gard", "ssen", "ova"]
KINDS = ["Industries", "Labs", "Works", "Group", "Systems", "Foods", "Energy", "Press", "Optics", "Shipping"]
CITY_END = ["brunn", "holt", "mouth", "wick", "mark", "quay", "stad", "ford", "havn", "lake"]
FIRST = ["Mira", "Oskar", "Lidia", "Tomas", "Nadja", "Felix", "Ines", "Ruben", "Greta", "Anton",
         "Selma", "Jonas", "Vera", "Emil", "Hanna", "Bruno", "Lotta", "Karim", "Ada", "Timo"]
ADJ = ["brass", "glass", "cobalt", "silk", "salt", "amber", "copper", "slate", "linen", "tin"]
NOUN = ["valves", "kettles", "sensors", "sails", "filters", "pumps", "inks", "ropes", "lamps", "gears"]

# (name, question template, paraphrase template, answer generator id)
RELATIONS = [
    ("headquarters", "Q: Where is the headquarters of {e}?\nA:", "Q: In which city is {e} based?\nA:", "city"),
    ("founder", "Q: Who founded {e}?\nA:", "Q: Who is the founder of {e}?\nA:", "person"),
    ("product", "Q: What is the main product of {e}?\nA:", "Q: What does {e} mainly produce?\nA:", "product"),
]
# deliberately different surface form, used for the last session with --template-shift
SHIFT_RELATION = ("engineer", "The chief engineer of {e} is named",
                  "Name of the chief engineer working at {e}:", "person")

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


def _unique(rng, make, n):
    out, seen = [], set()
    while len(out) < n:
        x = make(rng)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_sessions(seed, n_facts, n_sessions, template_shift):
    """Returns a list of sessions, each a list of fact dicts. Answers are random per seed."""
    rng = random.Random(seed)
    entities = _unique(rng, lambda r: f"{r.choice(SYL_A)}{r.choice(SYL_B)} {r.choice(KINDS)}", n_facts)
    gens = {
        "city": lambda r: r.choice(SYL_A) + r.choice(CITY_END),
        "person": lambda r: f"{r.choice(FIRST)} {r.choice(SYL_A)}{r.choice(SYL_B)}",
        "product": lambda r: f"{r.choice(ADJ)} {r.choice(NOUN)}",
    }
    pools = {k: _unique(rng, v, n_facts) for k, v in gens.items()}

    def fact(entity, rel, idx):
        _, q, p, kind = rel
        return {"question": q.format(e=entity), "paraphrase": p.format(e=entity),
                "answer": " " + pools[kind][idx], "relation": rel[0]}

    n_shift = n_facts // n_sessions if (template_shift and n_sessions > 1) else 0
    main = [fact(e, RELATIONS[i % len(RELATIONS)], i) for i, e in enumerate(entities[:n_facts - n_shift])]
    rng.shuffle(main)
    n_main_sessions = n_sessions - 1 if n_shift else n_sessions
    sessions = [main[i::n_main_sessions] for i in range(n_main_sessions)]
    if n_shift:
        shift = [fact(e, SHIFT_RELATION, n_facts - n_shift + j)
                 for j, e in enumerate(entities[n_facts - n_shift:])]
        sessions.append(shift)
    return sessions


# --------------------------------------------------------------------------------------
# Model / tokenizer / adapters
# --------------------------------------------------------------------------------------

class ByteTokenizer:
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
                          max_position_embeddings=4096, tie_word_embeddings=False,
                          initializer_range=0.2)
        model, tok = Qwen2ForCausalLM(cfg), ByteTokenizer()
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
        a, _, b = part.partition("-")
        ids.update(range(int(a), int(b or a) + 1))
    return ids


class PlasticSetup:
    """Injects adapters, consolidation store and activation capture; can be removed again."""

    def __init__(self, model, targets, layer_ids, rank, max_mass):
        self.records, self.adapters, self.handles = [], {}, []
        for parent_name, parent in list(model.named_modules()):
            for child_name, child in list(parent.named_children()):
                if child_name not in targets or not isinstance(child, nn.Linear):
                    continue
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                m = re.search(r"layers\.(\d+)\.", full)
                if layer_ids is not None and (m is None or int(m.group(1)) not in layer_ids):
                    continue
                self.records.append((parent, child_name, child, full))
        if not self.records:
            raise RuntimeError(f"No nn.Linear modules matched targets={targets}")
        for parent, child_name, child, full in self.records:
            adapter = PlasticLinearProjected(child, rank=rank, learning_rate=1.0, max_mass=max_mass)
            setattr(parent, child_name, adapter)
            self.adapters[full] = adapter
        self.capture_on, self.X, self.Y = False, {}, {}
        for name, layer in self.adapters.items():
            self.handles.append(layer.register_forward_pre_hook(self._pre(name)))
            self.handles.append(layer.register_forward_hook(self._post(name)))

    def _pre(self, name):
        def hook(mod, inp):
            if self.capture_on:
                self.X[name] = inp[0]
        return hook

    def _post(self, name):
        def hook(mod, inp, out):
            if self.capture_on:
                self.Y[name] = out
        return hook

    def clear_capture(self):
        self.X, self.Y = {}, {}

    def set_gate(self, value):
        for layer in self.adapters.values():
            layer.gate.fill_(value)

    def reset(self):
        for layer in self.adapters.values():
            layer.reset()  # clears active and consolidated factors
            layer.set_protected_subspace(None)

    @torch.no_grad()
    def consolidate(self):
        """Freeze active factors into each layer's consolidated store (PlasticLinearProjected.consolidate)."""
        for layer in self.adapters.values():
            layer.consolidate()

    def remove(self):
        for h in self.handles:
            h.remove()
        for parent, child_name, child, _ in self.records:
            setattr(parent, child_name, child)


# --------------------------------------------------------------------------------------
# Subspace construction
# --------------------------------------------------------------------------------------

def ids_of(tok, text, device, max_len=128):
    return torch.tensor([tok.encode(text)[:max_len]], dtype=torch.long, device=device)


@torch.no_grad()
def collect_inputs(model, setup, id_batches, drop_last=False):
    rows = {name: [] for name in setup.adapters}
    setup.capture_on = True
    for ids in id_batches:
        setup.clear_capture()
        model(input_ids=ids)
        for name, layer in setup.adapters.items():
            x = setup.X[name].reshape(-1, layer.base.in_features)
            rows[name].append(x[:-1] if drop_last else x)
    setup.capture_on = False
    setup.clear_capture()
    return {name: torch.cat(r).to(setup.adapters[name].base.weight.dtype) for name, r in rows.items()}


@torch.no_grad()
def set_static_q(setup, X_by_layer, k, center):
    warned = False
    for name, layer in setup.adapters.items():
        if k <= 0:
            layer.set_protected_subspace(None)
            continue
        X = X_by_layer[name]
        if center:
            X = X - X.mean(dim=0, keepdim=True)
        kk = min(k, X.shape[0], X.shape[1])
        if kk < k and not warned:
            print(f"  [warn] k={k} limited to {kk} by {X.shape[0]} calibration tokens; use --control-file")
            warned = True
        _, _, vh = torch.linalg.svd(X, full_matrices=False)
        layer.set_protected_subspace(vh[:kk].T)


@torch.no_grad()
def expand_q(layer, X, eps):
    """GPM energy criterion: add residual directions until span(Q) holds eps of X's energy."""
    total = X.pow(2).sum()
    R = layer.project(X)
    need = eps * total - (total - R.pow(2).sum())
    if need <= 0:
        return
    _, s, vh = torch.linalg.svd(R, full_matrices=False)
    k = min(int((torch.cumsum(s ** 2, 0) < need).sum().item()) + 1, s.numel())
    layer.set_protected_subspace(torch.cat([layer.Q, vh[:k].T], dim=1))


def q_fraction(setup):
    return sum(l.Q.shape[1] / l.base.in_features for l in setup.adapters.values()) / len(setup.adapters)


# --------------------------------------------------------------------------------------
# Learning and evaluation
# --------------------------------------------------------------------------------------

def fact_ids(tok, fact, device, paraphrase=False):
    p = tok.encode(fact["paraphrase"] if paraphrase else fact["question"])
    a = tok.encode(fact["answer"])
    return p, a, torch.tensor([p + a], dtype=torch.long, device=device)


def learn_fact(model, tok, setup, fact, lr, device):
    p_ids, a_ids, ids = fact_ids(tok, fact, device)
    labels = torch.full_like(ids, -100)
    labels[0, len(p_ids):] = ids[0, len(p_ids):]
    handle = model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    try:
        with torch.enable_grad():
            setup.clear_capture()
            setup.capture_on = True
            logits = model(input_ids=ids).logits
            setup.capture_on = False
            loss = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
            names = list(setup.Y)
            grads = torch.autograd.grad(loss, [setup.Y[n] for n in names])
    finally:
        handle.remove()
        setup.capture_on = False
    with torch.no_grad():
        for name, g in zip(names, grads):
            layer = setup.adapters[name]
            X = setup.X[name].detach().reshape(-1, layer.base.in_features)
            Y = setup.Y[name].detach().reshape(-1, layer.base.out_features)
            G = g.reshape(-1, layer.base.out_features)
            mask = G.norm(dim=1) > 0
            if not mask.any():
                continue
            X, Y, G = X[mask], Y[mask], G[mask]
            step = lr * X.shape[0] / X.pow(2).sum(dim=1).mean().clamp_min(1e-12)
            layer.adapt(X, Y - step * G)
    setup.clear_capture()
    return loss.item()


@torch.no_grad()
def eval_facts(model, tok, facts, device, paraphrase=False, context=None):
    """context: optional callable query -> context string (retrieval / in-context baselines)."""
    lps, exact = [], []
    for f in facts:
        p, a, ids = fact_ids(tok, f, device, paraphrase)
        if context is not None:
            query = f["paraphrase"] if paraphrase else f["question"]
            c = tok.encode(context(query) + "\n\n")
            p = c + p
            ids = torch.tensor([p + a], dtype=torch.long, device=device)
        logits = model(input_ids=ids).logits[0]
        pos = torch.arange(len(p) - 1, len(p) + len(a) - 1, device=device)
        tgt = torch.tensor(a, device=device)
        logp = F.log_softmax(logits[pos], dim=-1)
        lps.append(logp[torch.arange(len(a), device=device), tgt].mean().item())
        exact.append(float((logp.argmax(-1) == tgt).all().item()))
    return {"logprob": sum(lps) / len(lps), "exact": sum(exact) / len(exact)}


@torch.no_grad()
def drift(model, tok, setup, texts, device):
    kls, nll0, nll1, n, n_pred = 0.0, 0.0, 0.0, 0, 0
    for text in texts:
        ids = ids_of(tok, text, device)
        setup.set_gate(0.0)
        lp0 = F.log_softmax(model(input_ids=ids).logits[0].float(), -1)
        setup.set_gate(1.0)
        lp1 = F.log_softmax(model(input_ids=ids).logits[0].float(), -1)
        kls += (lp0.exp() * (lp0 - lp1)).sum().item()
        tgt = ids[0, 1:]
        nll0 += F.nll_loss(lp0[:-1], tgt, reduction="sum").item()
        nll1 += F.nll_loss(lp1[:-1], tgt, reduction="sum").item()
        n += ids.shape[1]
        n_pred += ids.shape[1] - 1
    return {"kl": kls / n, "ppl_base": math.exp(nll0 / n_pred), "ppl_adapted": math.exp(nll1 / n_pred)}


# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------

def parse_condition(spec):
    parts = spec.strip().split(":")
    mode = parts[0]
    cond = {"spec": spec.strip(), "mode": mode, "family": None, "variant": None, "k": 0, "center": False,
            "consol": False, "eps": None, "base_eps": None, "nobase": False, "lam": 0.5}
    rest = parts[1:]
    if mode == "lora":
        if not rest or rest[0] not in ("seq", "multi", "olora", "gpm", "replay"):
            raise ValueError(f"lora needs a variant seq|multi|olora|gpm|replay: '{spec}'")
        cond["variant"], rest = rest[0], rest[1:]
    for p in rest:
        if "=" in p:
            key, val = p.split("=")
            cond[key] = float(val) if key in ("eps", "base_eps", "lam") else (val if key == "k" and val == "all" else int(val))
        else:
            cond[p] = True
    if mode in ("static", "online"):
        cond["family"] = "plastic"
    elif mode == "lora":
        cond["family"] = "lora"
    elif mode in ("rag", "icl"):
        cond["family"] = "context"
        if mode == "rag" and cond["k"] in (0, None):
            cond["k"] = 4
    else:
        raise ValueError(f"Unknown condition mode in '{spec}'")
    if mode == "online" or cond["variant"] == "gpm":
        if cond["eps"] is None:
            raise ValueError(f"{mode} condition needs eps=...: '{spec}'")
        cond["consol"] = True  # required, see module docstring
        if not cond["nobase"] and cond["base_eps"] is None:
            cond["base_eps"] = 0.9
    return cond


def run_condition(model, tok, setup, sessions, base_eval, cond, args, calib_X, calib_texts, heldout_texts):
    setup.reset()
    setup.set_gate(1.0)
    if cond["mode"] == "static":
        set_static_q(setup, calib_X, cond["k"], cond["center"])
    elif not cond["nobase"]:
        for name, layer in setup.adapters.items():
            expand_q(layer, calib_X[name], cond["base_eps"])
    q_trace = [q_fraction(setup)]

    S = len(sessions)
    R_lp = [[None] * S for _ in range(S)]
    R_ex = [[None] * S for _ in range(S)]
    t0 = time.perf_counter()
    for s, session in enumerate(sessions):
        for _ in range(args.epochs):
            order = session[:]
            random.shuffle(order)
            for fact in order:
                learn_fact(model, tok, setup, fact, args.lr, args.device)
        for j in range(s + 1):
            r = eval_facts(model, tok, sessions[j], args.device)
            R_lp[s][j], R_ex[s][j] = r["logprob"], r["exact"]
        if cond["consol"] and s < S - 1:
            setup.consolidate()
        if cond["mode"] == "online" and s < S - 1:
            X = collect_inputs(model, setup, [fact_ids(tok, f, args.device)[2] for f in session], drop_last=True)
            for name, layer in setup.adapters.items():
                expand_q(layer, X[name], cond["eps"])
        q_trace.append(q_fraction(setup))

    all_facts = [f for sess in sessions for f in sess]
    state_numel = sum(l.state_numel() for l in setup.adapters.values())
    final = eval_facts(model, tok, all_facts, args.device)
    para = eval_facts(model, tok, all_facts, args.device, paraphrase=True)
    d_cal = drift(model, tok, setup, calib_texts, args.device)
    d_held = drift(model, tok, setup, heldout_texts, args.device)
    gains = [R_lp[s][s] - base_eval[s]["logprob"] for s in range(S)]
    return {
        "condition": cond["spec"],
        "recall": final["exact"], "recall_logprob": final["logprob"],
        "paraphrase": para["exact"], "paraphrase_logprob": para["logprob"],
        "gain_first": gains[0], "gain_last": gains[-1], "gains": gains,
        "forget_first": R_lp[0][0] - R_lp[S - 1][0],
        "retention_first_exact": R_ex[S - 1][0],
        "R_logprob": R_lp, "R_exact": R_ex,
        "q_fraction_trace": q_trace, "q_fraction": q_trace[-1],
        "kl_calib": d_cal["kl"], "kl_heldout": d_held["kl"],
        "ppl_base": d_held["ppl_base"], "ppl_adapted": d_held["ppl_adapted"],
        "active_mass_mean": sum(l.diagnostics()["mass"] for l in setup.adapters.values()) / len(setup.adapters),
        "learn_seconds": time.perf_counter() - t0,
        "state_mb_per_fact": state_numel * 4 / 1e6 / len(all_facts),
        "train_passes": len(all_facts) * args.epochs,
    }


# --------------------------------------------------------------------------------------
# Baselines B1-B5: LoRA trained by backprop on the same facts, sessions and budget
# --------------------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """y = W x + gate * (sum_i B_i A_i x + B A x). Frozen factors are consolidated sessions.

    Exposes the attributes used by expand_q / q_fraction (Q, project, set_protected_subspace,
    base.in_features), so B4 grows Q with exactly the same code as the online:* conditions.
    """

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base, self.rank = base, rank
        dev, dt = base.weight.device, base.weight.dtype
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, device=dev, dtype=dt))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        self.register_buffer("gate", torch.ones((), device=dev, dtype=dt))
        self.register_buffer("Q", torch.zeros(base.in_features, 0, device=dev, dtype=dt))
        self.frozen_A = torch.zeros(0, base.in_features, device=dev, dtype=dt)
        self.frozen_B = torch.zeros(base.out_features, 0, device=dev, dtype=dt)
        self.init_active()

    @torch.no_grad()
    def init_active(self):
        """Standard LoRA init: A random, B zero, so the new adapter starts as a no-op."""
        self.A.normal_(0.0, self.base.in_features ** -0.5)
        self.B.zero_()
        self.project_A()

    @torch.no_grad()
    def project_A(self):
        if self.Q.shape[1]:
            self.A.sub_((self.A @ self.Q) @ self.Q.T)

    def project(self, x):
        return x - (x @ self.Q) @ self.Q.T if self.Q.shape[1] else x

    def set_protected_subspace(self, basis):
        if basis is None or basis.shape[1] == 0:
            self.Q = torch.zeros(self.base.in_features, 0, device=self.A.device, dtype=self.A.dtype)
            return
        q, _ = torch.linalg.qr(basis.to(self.A.dtype))
        self.Q = q

    @torch.no_grad()
    def freeze_active(self):
        self.frozen_A = torch.cat([self.frozen_A, self.A.detach().clone()], dim=0)
        self.frozen_B = torch.cat([self.frozen_B, self.B.detach().clone()], dim=1)
        self.init_active()

    @torch.no_grad()
    def reset(self):
        self.frozen_A = self.frozen_A[:0]
        self.frozen_B = self.frozen_B[:, :0]
        self.set_protected_subspace(None)
        self.init_active()

    def state_numel(self):
        return self.A.numel() + self.B.numel() + self.frozen_A.numel() + self.frozen_B.numel()

    def forward(self, x):
        out = self.base(x)
        delta = (x @ self.A.T) @ self.B.T
        if self.frozen_A.shape[0]:
            delta = delta + (x @ self.frozen_A.T) @ self.frozen_B.T
        return out + self.gate * delta


class LoRASetup:
    """Same interface as PlasticSetup (adapters, capture, set_gate, reset, remove)."""

    def __init__(self, model, targets, layer_ids, rank):
        self.records, self.adapters, self.handles = [], {}, []
        for parent_name, parent in list(model.named_modules()):
            for child_name, child in list(parent.named_children()):
                if child_name not in targets or not isinstance(child, nn.Linear):
                    continue
                full = f"{parent_name}.{child_name}" if parent_name else child_name
                m = re.search(r"layers\.(\d+)\.", full)
                if layer_ids is not None and (m is None or int(m.group(1)) not in layer_ids):
                    continue
                self.records.append((parent, child_name, child, full))
        if not self.records:
            raise RuntimeError(f"No nn.Linear modules matched targets={targets}")
        for parent, child_name, child, full in self.records:
            layer = LoRALinear(child, rank)
            setattr(parent, child_name, layer)
            self.adapters[full] = layer
        self.capture_on, self.X, self.Y = False, {}, {}
        for name, layer in self.adapters.items():
            self.handles.append(layer.register_forward_pre_hook(self._pre(name)))

    def _pre(self, name):
        def hook(mod, inp):
            if self.capture_on:
                self.X[name] = inp[0]
        return hook

    def clear_capture(self):
        self.X, self.Y = {}, {}

    def set_gate(self, value):
        for layer in self.adapters.values():
            layer.gate.fill_(value)

    def reset(self):
        for layer in self.adapters.values():
            layer.reset()

    def params(self):
        return [p for layer in self.adapters.values() for p in (layer.A, layer.B)]

    def remove(self):
        for h in self.handles:
            h.remove()
        for parent, child_name, child, _ in self.records:
            setattr(parent, child_name, child)


def fact_loss(model, tok, fact, device):
    p_ids, a_ids, ids = fact_ids(tok, fact, device)
    labels = torch.full_like(ids, -100)
    labels[0, len(p_ids):] = ids[0, len(p_ids):]
    logits = model(input_ids=ids).logits
    return F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)


def make_optimizer(params, args):
    if args.lora_opt == "adam":
        return torch.optim.Adam(params, lr=args.lora_lr)
    return torch.optim.SGD(params, lr=args.lora_lr)


def run_lora(model, tok, setup, sessions, base_eval, cond, args, calib_X, calib_texts, heldout_texts):
    """B1-B5. Adam is allowed for B4 because the constraint is enforced on the parameter
    (A <- A P after every step), not on the gradient, which Adam's per-coordinate scaling would break."""
    v = cond["variant"]
    setup.reset()
    setup.set_gate(1.0)
    if v == "gpm" and not cond["nobase"]:
        for name, layer in setup.adapters.items():
            expand_q(layer, calib_X[name], cond["base_eps"])
        for layer in setup.adapters.values():
            layer.project_A()
    q_trace = [q_fraction(setup)]
    S = len(sessions)
    R_lp = [[None] * S for _ in range(S)]
    R_ex = [[None] * S for _ in range(S)]
    passes = 0
    t0 = time.perf_counter()
    for s, session in enumerate(sessions):
        opt = make_optimizer(setup.params(), args)
        previous = [f for j in range(s) for f in sessions[j]]
        for _ in range(args.epochs):
            order = session[:]
            random.shuffle(order)
            for fact in order:
                with torch.enable_grad():
                    loss = fact_loss(model, tok, fact, args.device)
                    passes += 1
                    if v == "replay" and previous:
                        loss = loss + fact_loss(model, tok, random.choice(previous), args.device)
                        passes += 1
                    if v == "olora":
                        pen = sum(((l.A @ l.frozen_A.T) ** 2).sum()
                                  for l in setup.adapters.values() if l.frozen_A.shape[0])
                        if torch.is_tensor(pen):
                            loss = loss + cond["lam"] * pen
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                opt.step()
                if v == "gpm":
                    for layer in setup.adapters.values():
                        layer.project_A()
        for j in range(s + 1):
            r = eval_facts(model, tok, sessions[j], args.device)
            R_lp[s][j], R_ex[s][j] = r["logprob"], r["exact"]
        if s < S - 1:
            if v in ("multi", "olora", "gpm"):
                for layer in setup.adapters.values():
                    layer.freeze_active()
            if v == "gpm":
                X = collect_inputs(model, setup, [fact_ids(tok, f, args.device)[2] for f in session], drop_last=True)
                for name, layer in setup.adapters.items():
                    expand_q(layer, X[name], cond["eps"])
                    layer.project_A()
        q_trace.append(q_fraction(setup))

    all_facts = [f for sess in sessions for f in sess]
    final = eval_facts(model, tok, all_facts, args.device)
    para = eval_facts(model, tok, all_facts, args.device, paraphrase=True)
    d_cal = drift(model, tok, setup, calib_texts, args.device)
    d_held = drift(model, tok, setup, heldout_texts, args.device)
    gains = [R_lp[s][s] - base_eval[s]["logprob"] for s in range(S)]
    state_numel = sum(l.state_numel() for l in setup.adapters.values())
    return {
        "condition": cond["spec"],
        "recall": final["exact"], "recall_logprob": final["logprob"],
        "paraphrase": para["exact"], "paraphrase_logprob": para["logprob"],
        "gain_first": gains[0], "gain_last": gains[-1], "gains": gains,
        "forget_first": R_lp[0][0] - R_lp[S - 1][0],
        "retention_first_exact": R_ex[S - 1][0],
        "R_logprob": R_lp, "R_exact": R_ex,
        "q_fraction_trace": q_trace, "q_fraction": q_trace[-1],
        "kl_calib": d_cal["kl"], "kl_heldout": d_held["kl"],
        "ppl_base": d_held["ppl_base"], "ppl_adapted": d_held["ppl_adapted"],
        "learn_seconds": time.perf_counter() - t0,
        "state_mb_per_fact": state_numel * 4 / 1e6 / len(all_facts),
        "train_passes": passes,
    }


# --------------------------------------------------------------------------------------
# Baseline B6: retrieval / in-context with the frozen model
# --------------------------------------------------------------------------------------

STOPWORDS = set("q a the of is in at who what where which does do did an and to for by on with from "
                "main mainly name named its it this that der die das und ist".split())


def words(text):
    return [w for w in re.findall(r"[a-zäöüß0-9]+", text.lower()) if w not in STOPWORDS]


def fact_doc(f):
    return f["question"].replace("\n", " ").strip() + f["answer"]


class LexicalRetriever:
    """IDF-weighted word overlap (BM25-lite, no dependencies). Returns only docs with a match."""

    def __init__(self, facts, k):
        self.docs = [fact_doc(f) for f in facts]
        self.tokens = [set(words(d)) for d in self.docs]
        n = len(self.docs)
        df = {}
        for t in self.tokens:
            for w in t:
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}
        self.k = k

    def __call__(self, query):
        q = set(words(query))
        scored = [(sum(self.idf.get(w, 0.0) for w in q & t), i) for i, t in enumerate(self.tokens)]
        top = [i for sc, i in sorted(scored, reverse=True)[:self.k] if sc > 0]
        return "Known facts:\n" + "\n".join(self.docs[i] for i in top) if top else ""


def make_context(cond, facts):
    if cond["mode"] == "icl":
        docs = "\n".join(fact_doc(f) for f in facts)
        return lambda query: "Known facts:\n" + docs
    return LexicalRetriever(facts, int(cond["k"]))


@torch.no_grad()
def drift_context(model, tok, texts, context, device):
    """KL on the control tokens with vs. without the retrieved context (the drift a
    retrieval system causes when it fires on unrelated input)."""
    kls, nll0, nll1, n, n_pred = 0.0, 0.0, 0.0, 0, 0
    for text in texts:
        t_ids = tok.encode(text)[:128]
        ctx = context(text)
        c_ids = tok.encode(ctx + "\n\n") if ctx else []
        lp0 = F.log_softmax(model(input_ids=torch.tensor([t_ids], device=device)).logits[0].float(), -1)
        lp1 = F.log_softmax(model(input_ids=torch.tensor([c_ids + t_ids], device=device)).logits[0].float(), -1)
        lp1 = lp1[len(c_ids):]
        kls += (lp0.exp() * (lp0 - lp1)).sum().item()
        tgt = torch.tensor(t_ids[1:], device=device)
        nll0 += F.nll_loss(lp0[:-1], tgt, reduction="sum").item()
        nll1 += F.nll_loss(lp1[:-1], tgt, reduction="sum").item()
        n += len(t_ids)
        n_pred += len(t_ids) - 1
    return {"kl": kls / n, "ppl_base": math.exp(nll0 / n_pred), "ppl_adapted": math.exp(nll1 / n_pred)}


def run_context(model, tok, sessions, base_eval, cond, args, calib_texts, heldout_texts):
    S = len(sessions)
    R_lp = [[None] * S for _ in range(S)]
    R_ex = [[None] * S for _ in range(S)]
    for s in range(S):
        ctx = make_context(cond, [f for j in range(s + 1) for f in sessions[j]])
        for j in range(s + 1):
            r = eval_facts(model, tok, sessions[j], args.device, context=ctx)
            R_lp[s][j], R_ex[s][j] = r["logprob"], r["exact"]
    all_facts = [f for sess in sessions for f in sess]
    ctx = make_context(cond, all_facts)
    final = eval_facts(model, tok, all_facts, args.device, context=ctx)
    para = eval_facts(model, tok, all_facts, args.device, paraphrase=True, context=ctx)
    d_cal = drift_context(model, tok, calib_texts, ctx, args.device)
    d_held = drift_context(model, tok, heldout_texts, ctx, args.device)
    gains = [R_lp[s][s] - base_eval[s]["logprob"] for s in range(S)]
    return {
        "condition": cond["spec"],
        "recall": final["exact"], "recall_logprob": final["logprob"],
        "paraphrase": para["exact"], "paraphrase_logprob": para["logprob"],
        "gain_first": gains[0], "gain_last": gains[-1], "gains": gains,
        "forget_first": R_lp[0][0] - R_lp[S - 1][0],
        "retention_first_exact": R_ex[S - 1][0],
        "R_logprob": R_lp, "R_exact": R_ex,
        "q_fraction_trace": [], "q_fraction": 0.0,
        "kl_calib": d_cal["kl"], "kl_heldout": d_held["kl"],
        "ppl_base": d_held["ppl_base"], "ppl_adapted": d_held["ppl_adapted"],
        "learn_seconds": 0.0,
        "state_mb_per_fact": sum(len(fact_doc(f).encode("utf-8")) for f in all_facts) / 1e6 / len(all_facts),
        "train_passes": 0,
    }


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T95.get(n - 1, 2.0) * sd / math.sqrt(n)


def fmt(xs, spec=".2f"):
    m, ci = mean_ci(xs)
    return f"{m:{spec}}" if math.isnan(ci) else f"{m:{spec}} ± {ci:{spec}}"


def print_summary(runs):
    print("\n## Summary (mean ± 95% CI over seeds)\n")
    cols = ["rank", "condition", "n", "recall", "paraphrase", "gain S1", "gain last", "forget S1",
            "ret S1 exact", "dim(Q)/d_in", "KL held-out", "Δppl held-out", "state MB/fact", "train passes"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---:|" * len(cols))
    keys = []
    for r in runs:
        key = (r["rank"], r["condition"])
        if key not in keys:
            keys.append(key)
    for rank, cond in keys:
        g = [r for r in runs if r["rank"] == rank and r["condition"] == cond]
        get = lambda k: [r[k] for r in g]
        print(f"| {rank} | {cond} | {len(g)} | {fmt(get('recall'))} | {fmt(get('paraphrase'))} | "
              f"{fmt(get('gain_first'))} | {fmt(get('gain_last'))} | {fmt(get('forget_first'))} | "
              f"{fmt(get('retention_first_exact'))} | {fmt(get('q_fraction'), '.3f')} | "
              f"{fmt(get('kl_heldout'), '.1e')} | {fmt([r['ppl_adapted'] - r['ppl_base'] for r in g], '+.3f')} | "
              f"{fmt(get('state_mb_per_fact'), '.4f')} | {fmt(get('train_passes'), '.0f')} |")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--tiny", action="store_true", help="offline smoke test with a random tiny Qwen2")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--targets", default="down_proj", help="comma-separated module names")
    ap.add_argument("--layers", default="all", help="'all' or e.g. '4-15' or '2,5,9'")
    ap.add_argument("--ranks", default="8", help="comma-separated adapter ranks (diagnosis A)")
    ap.add_argument("--max-mass", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=0.1, help="NLMS step size for the projected gradient step")
    ap.add_argument("--conditions", default=DEFAULT_CONDITIONS, help="';'-separated, see docstring")
    ap.add_argument("--lora-lr", type=float, default=1e-2, help="LoRA learning rate (tune per method)")
    ap.add_argument("--lora-opt", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--n-facts", type=int, default=48)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--template-shift", action=argparse.BooleanOptionalAction, default=True,
                    help="last session uses a different surface template")
    ap.add_argument("--epochs", type=int, default=10, help="passes over each session's facts")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--control-file", default=None,
                    help="text file with blank-line-separated blocks; even blocks calibrate, odd blocks are held out")
    ap.add_argument("--out", default="results_gradient_targets_drift.json")
    args = ap.parse_args()

    if args.tiny and args.model == ap.get_default("model"):
        args.model = "tiny-random-qwen2"
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    ranks = [int(r) for r in args.ranks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    conditions = [parse_condition(c) for c in args.conditions.split(";") if c.strip()]

    texts = CONTROL_TEXTS
    if args.control_file:
        texts = [b.strip() for b in Path(args.control_file).read_text(encoding="utf-8").split("\n\n") if b.strip()]
    calib_texts, heldout_texts = texts[0::2], texts[1::2]

    model, tok = load_model(args)
    print(f"# Gradient targets, drift & online Q | model={args.model} | targets={','.join(targets)} | "
          f"ranks={ranks} | lr={args.lr} | facts={args.n_facts} in {args.sessions} sessions "
          f"(template shift: {args.template_shift}) | seeds={seeds} | device={args.device}")

    results = {"config": vars(args), "runs": []}
    families = [f for f in ("plastic", "lora", "context") if any(c["family"] == f for c in conditions)]
    base_cache = {}

    def base_eval_for(seed, sessions):
        if seed not in base_cache:
            base_cache[seed] = [eval_facts(model, tok, s, args.device) for s in sessions]
        return base_cache[seed]

    def record(r, rank, seed, cond):
        r.update({"rank": rank, "seed": seed})
        results["runs"].append(r)
        print(f"  [rank {rank}] {cond['spec']:<32} recall={r['recall']:.2f} gain S1/last={r['gain_first']:+.2f}/"
              f"{r['gain_last']:+.2f} forget S1={r['forget_first']:+.2f} dimQ={r['q_fraction']:.3f} "
              f"KL_held={r['kl_heldout']:.1e} ({r['learn_seconds']:.0f}s)")
        sys.stdout.flush()
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    for family in families:
        fam_conds = [c for c in conditions if c["family"] == family]
        for rank in (ranks if family != "context" else ["-"]):
            if family == "plastic":
                setup = PlasticSetup(model, targets, parse_layers(args.layers), rank, args.max_mass)
            elif family == "lora":
                setup = LoRASetup(model, targets, parse_layers(args.layers), rank)
            else:
                setup = None
            calib_X = None
            if setup is not None:
                calib_X = collect_inputs(model, setup, [ids_of(tok, t, args.device) for t in calib_texts])
                print(f"\n# {family} rank={rank} | {len(setup.adapters)} layers | "
                      f"{next(iter(calib_X.values())).shape[0]} calibration tokens")
            for seed in seeds:
                sessions = build_sessions(seed, args.n_facts, args.sessions, args.template_shift)
                if setup is not None:
                    setup.reset()
                    setup.set_gate(0.0)
                base_eval = base_eval_for(seed, sessions)
                if setup is not None:
                    setup.set_gate(1.0)
                for cond in fam_conds:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    if family == "plastic":
                        r = run_condition(model, tok, setup, sessions, base_eval, cond, args,
                                          calib_X, calib_texts, heldout_texts)
                    elif family == "lora":
                        r = run_lora(model, tok, setup, sessions, base_eval, cond, args,
                                     calib_X, calib_texts, heldout_texts)
                    else:
                        r = run_context(model, tok, sessions, base_eval, cond, args, calib_texts, heldout_texts)
                    record(r, rank, seed, cond)
            if setup is not None:
                setup.remove()

    print_summary(results["runs"])
    print(f"\nFull results incl. retention matrices R[s][j] and Q traces: {args.out}")


if __name__ == "__main__":
    main()
