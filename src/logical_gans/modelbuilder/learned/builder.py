"""Neural relation Builder: the net proposes edits; the Devil verifies.

The neural model is NOT trusted. Every step the bounded Devil re-checks the
structure; the model only chooses which legal (UNKNOWN-cell) edit to apply.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..core.devil import run_devil_bounded
from ..core.partial_structure import PartialStructure
from ..core.theory import Theory
from .actions import (
    apply_relation_edit,
    decode_action_index,
    encode_action_index,
    legal_relation_edits,
)
from .encoding import encode_state
from .policy_net import RelationPolicyNet


def load_policy(model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = RelationPolicyNet(checkpoint["n"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["n"]


def _result(status, relation, n, k, budget, structure, trace) -> dict:
    return {
        "status": status,
        "builder": "neural_relation_policy",
        "relation": relation,
        "n": n,
        "k": k,
        "budget": budget,
        "structure": structure.to_json(),
        "trace": trace,
    }


def neural_relation_build(
    theory: Theory,
    relation: str,
    n: int,
    model_path: str,
    seed_structure: Optional[PartialStructure] = None,
    k: Optional[int] = None,
    budget: Optional[int] = None,
    max_steps: int = 50,
) -> dict:
    model, model_n = load_policy(model_path)
    if model_n != n:
        raise ValueError(f"model was trained for n={model_n}, but n={n} was requested")

    structure = seed_structure.copy() if seed_structure is not None else \
        PartialStructure.empty(theory.signature, n)
    trace: list = []

    for _ in range(max_steps):
        result = run_devil_bounded(structure, theory.clauses, k=k, budget=budget)
        if result.status == "ok":
            return _result("satisfied", relation, n, k, budget, structure, trace)
        if result.status == "failed":
            trace.append({"event": "devil_failed", "witness": result.witness.to_json()})
            return _result("failed", relation, n, k, budget, structure, trace)

        legal = legal_relation_edits(structure, relation)
        if not legal:
            return _result("unknown", relation, n, k, budget, structure, trace)

        x = encode_state(structure, relation, result.witness).unsqueeze(0)
        with torch.no_grad():
            logits = model(x).squeeze(0)
        order = torch.argsort(logits, descending=True).tolist()
        legal_index = {encode_action_index(e, n): e for e in legal}

        chosen = None
        chosen_rank = None
        for rank, idx in enumerate(order):
            if idx in legal_index:
                chosen = decode_action_index(idx, n, relation)
                chosen_rank = rank
                break
        if chosen is None:
            return _result("unknown", relation, n, k, budget, structure, trace)

        structure = apply_relation_edit(structure, chosen)
        trace.append({
            "event": "neural_action",
            "chosen_edit": {"relation": chosen.relation, "args": list(chosen.args),
                            "value": chosen.value.value},
            "rank": chosen_rank,
            "devil_witness": result.witness.to_json(),
        })

    return _result("unknown", relation, n, k, budget, structure, trace)
