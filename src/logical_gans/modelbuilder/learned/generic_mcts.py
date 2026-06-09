"""Signature-parametric MCTS over generic semantic edits.

State  = PartialStructure (any signature)
Action = SemanticEdit (SetRelation / SetFunction / SetConstant)
Prior  = a PriorProvider over the candidate edits, or uniform
Reward = symbolic Devil result

At an UNKNOWN node the candidate actions are the edits that discharge the
Devil's current obligation (relation -> 2 truths; function/constant -> domain
values); if no obligation is found it falls back to all legal edits. The
relation-only MCTS in ``mcts.py`` is kept as a prototype; this is the main line.
"""
from __future__ import annotations

import math
from typing import List, Optional

from ..core.devil import run_devil_bounded
from ..core.obligations import extract_obligation
from ..core.partial_structure import PartialStructure
from ..core.theory import Theory
from .semantic_actions import (
    SemanticEdit,
    SetConstant,
    SetFunction,
    SetRelation,
    apply_semantic_edit,
    legal_semantic_edits,
    semantic_edit_to_json,
)


def _reward(result) -> float:
    if result.status == "failed":
        return -1.0
    if result.status == "ok":
        return 0.3 if result.budget_exhausted else 1.0
    return 0.0


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def _obligation_edits(obl) -> List[SemanticEdit]:
    if obl.kind == "relation":
        return [SetRelation(obl.symbol, obl.args, v) for v in obl.suggested_values]
    if obl.kind == "function":
        return [SetFunction(obl.symbol, obl.args, v) for v in obl.suggested_values]
    if obl.kind == "constant":
        return [SetConstant(obl.symbol, v) for v in obl.suggested_values]
    return []


class _Node:
    __slots__ = ("structure", "parent", "action_from_parent", "children", "visits",
                 "total_reward", "devil_status", "terminal", "reward", "child_edits",
                 "priors", "expanded")

    def __init__(self, structure, parent, action_from_parent):
        self.structure = structure
        self.parent = parent
        self.action_from_parent = action_from_parent  # SemanticEdit | None
        self.children = {}                            # idx -> _Node
        self.visits = 0
        self.total_reward = 0.0
        self.devil_status = None
        self.terminal = False
        self.reward = 0.0
        self.child_edits = []                         # List[SemanticEdit]
        self.priors = []                              # List[float] aligned with child_edits
        self.expanded = False


def _evaluate(node, theory, k, budget, prior_hook):
    result = run_devil_bounded(node.structure, theory.clauses, k=k, budget=budget)
    node.devil_status = result.status
    node.reward = _reward(result)
    if result.status in ("ok", "failed"):
        node.terminal = True
        return
    obl = extract_obligation(node.structure, result)
    edits = _obligation_edits(obl) if obl is not None else legal_semantic_edits(node.structure)
    if not edits:
        node.terminal = True
        node.reward = 0.0
        return
    node.child_edits = edits
    if prior_hook is None:
        node.priors = [1.0 / len(edits)] * len(edits)
    else:
        scores = prior_hook.score(node.structure, result, edits)
        node.priors = _softmax([scores.get(e, 0.0) for e in edits])


def _expand(node):
    for idx, edit in enumerate(node.child_edits):
        node.children[idx] = _Node(apply_semantic_edit(node.structure, edit), node, edit)
    node.expanded = True


def _puct(parent, idx, child, c_puct):
    q = (child.total_reward / child.visits) if child.visits > 0 else 0.0
    u = c_puct * parent.priors[idx] * math.sqrt(parent.visits + 1) / (1 + child.visits)
    return q + u


def mcts_semantic_build(
    theory: Theory,
    n: int,
    seed_structure: Optional[PartialStructure] = None,
    k: Optional[int] = None,
    budget: Optional[int] = None,
    rollouts: int = 100,
    c_puct: float = 1.5,
    prior_hook=None,
) -> dict:
    start = seed_structure.copy() if seed_structure is not None \
        else PartialStructure.empty(theory.signature, n)
    root = _Node(start, None, None)
    _evaluate(root, theory, k, budget, prior_hook)
    nodes = 1
    best_node = root if root.devil_status == "ok" else None

    for _ in range(rollouts):
        node = root
        while node.expanded and not node.terminal:
            idx, node = max(node.children.items(),
                            key=lambda kv: _puct(kv[1].parent, kv[0], kv[1], c_puct))
            if node.devil_status is None:
                _evaluate(node, theory, k, budget, prior_hook)
                nodes += 1
                break
        if (not node.terminal) and (not node.expanded) and node.child_edits:
            _expand(node)
        if node.devil_status == "ok" and (best_node is None or node.reward > best_node.reward):
            best_node = node
        value = node.reward
        back = node
        while back is not None:
            back.visits += 1
            back.total_reward += value
            back = back.parent

    trace = []
    if best_node is not None and best_node is not root:
        chain = []
        nd = best_node
        while nd.parent is not None:
            chain.append(nd)
            nd = nd.parent
        chain.reverse()
        for nd in chain:
            parent = nd.parent
            idx = next((i for i, c in parent.children.items() if c is nd), None)
            prior = parent.priors[idx] if (idx is not None and idx < len(parent.priors)) else None
            trace.append({
                "event": "mcts_semantic_action",
                "edit": semantic_edit_to_json(nd.action_from_parent),
                "prior": prior,
                "visits": nd.visits,
                "q": (nd.total_reward / nd.visits) if nd.visits else 0.0,
                "devil_status": nd.devil_status,
            })

    if best_node is not None:
        status, final = "satisfied", best_node.structure
    elif root.devil_status == "failed":
        status, final = "failed", root.structure
    else:
        status, final = "unknown", root.structure

    return {
        "status": status,
        "builder": "mcts_semantic",
        "n": n,
        "k": k,
        "budget": budget,
        "rollouts": rollouts,
        "nodes": nodes,
        "structure": final.to_json(),
        "trace": trace,
    }
