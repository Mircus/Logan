"""Training data + trainer for the generic SemanticPolicyNet.

Labels come from successful generic refute traces (an MCTS oracle search,
allowed as a baseline). Each example is a candidate-action feature matrix plus
the index of the action taken on the winning trajectory. The final proof runs
with the *trained* neural prior.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch

from ..core.devil import run_devil_bounded
from ..core.obligations import extract_obligation
from ..core.partial_structure import PartialStructure
from .generic_mcts import mcts_semantic_refute, refute_candidates
from .semantic_actions import apply_semantic_edit, semantic_edit_from_json
from .semantic_features import FEATURE_DIM, feature_vector
from .semantic_policy_net import SemanticPolicyNet


def make_semantic_training_examples(theory, claim, n, num_samples, seed=0, rollouts=400):
    examples = []
    attempt = 0
    while len(examples) < num_samples and attempt < num_samples * 5 + 100:
        res = mcts_semantic_refute(theory, claim, n, prior_provider=None,
                                   rollouts=rollouts, seed=seed + attempt)
        attempt += 1
        if res["status"] != "refuted":
            continue
        edits = [semantic_edit_from_json(e["edit"]) for e in res["trace"]]
        A = PartialStructure.empty(theory.signature, n)
        for edit in edits:
            dT = run_devil_bounded(A, theory.clauses)
            cands = refute_candidates(A, theory, claim, dT)
            if edit in cands:
                obl = extract_obligation(A, dT) if dT.status == "unknown" else None
                feats = [feature_vector(c, A, obl) for c in cands]
                examples.append({"features": feats, "target": cands.index(edit)})
                if len(examples) >= num_samples:
                    break
            A = apply_semantic_edit(A, edit)
    return examples


def write_semantic_training_jsonl(examples, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def read_semantic_training_jsonl(path) -> List[dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def train_semantic_policy(data_path, out_path, epochs=50, lr=1e-3, seed=0) -> dict:
    torch.manual_seed(seed)
    examples = read_semantic_training_jsonl(data_path)
    if not examples:
        raise ValueError(f"no training examples in {data_path}")

    model = SemanticPolicyNet(FEATURE_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    final_loss = None
    for _ in range(epochs):
        total = 0.0
        for ex in examples:
            feats = torch.tensor(ex["features"], dtype=torch.float32)   # (m, d)
            target = torch.tensor([ex["target"]])
            logits = model(feats).unsqueeze(0)                          # (1, m)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        final_loss = total / len(examples)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"feature_dim": FEATURE_DIM, "state_dict": model.state_dict(),
                "metadata": {"examples": len(examples), "epochs": epochs, "final_loss": final_loss}},
               out)
    return {"out": str(out), "examples": len(examples), "epochs": epochs, "final_loss": final_loss}


def train_token_policy(data_path, out_path, epochs=50, lr=1e-3, seed=0) -> dict:
    """Train the arity-parametric TokenSemanticPolicyNet.

    Each example is {"edits": [<list of token-rows> per candidate], "target": idx}.
    """
    from .semantic_policy_net import TokenSemanticPolicyNet
    from .semantic_token_features import TOKEN_DIM

    torch.manual_seed(seed)
    examples = read_semantic_training_jsonl(data_path)
    if not examples:
        raise ValueError(f"no training examples in {data_path}")

    model = TokenSemanticPolicyNet(TOKEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    final_loss = None
    for _ in range(epochs):
        total = 0.0
        for ex in examples:
            edits = [torch.tensor(toks, dtype=torch.float32) for toks in ex["edits"]]
            target = torch.tensor([ex["target"]])
            logits = model(edits).unsqueeze(0)              # (1, m)
            loss = loss_fn(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        final_loss = total / len(examples)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": "token", "token_dim": TOKEN_DIM, "state_dict": model.state_dict(),
                "metadata": {"examples": len(examples), "epochs": epochs, "final_loss": final_loss}},
               out)
    return {"out": str(out), "examples": len(examples), "epochs": epochs, "final_loss": final_loss}
