"""Neural-guided tree-policy MCTS Builder over generic semantic edits.

Promoted out of ``experiments/`` so the tool/arena does not depend on a demo.
The Builder plays against the bounded Devil: at each node it proposes semantic
edits (the full legal-edit set for the signature) and the Devil verifies. Random
rollouts are DISABLED -- a win can only be reached by the prior-guided tree
descent (``success_via = "guided_tree_policy"``).

Outcome of a node (mode-aware):
    refute:   won = theory ⊩ and claim ⊭   ;  dead_end = theory ⊩ and claim ⊩
    satisfy:  won = theory ⊩ and goal ⊩     ;  dead_end = theory ⊩ and goal ⊭
    both:     theory_failed = theory ⊭       ;  else open

Verification uses ``run_devil_bounded(k, devil_budget)`` so the engine speaks the
UCMT bound directly; ``k=None, devil_budget=None`` reproduces exhaustive checking.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..core.devil import run_devil_bounded
from ..core.obligations import extract_obligation
from ..core.partial_structure import PartialStructure
from .semantic_actions import (
    SemanticEdit,
    SetConstant,
    SetFunction,
    SetRelation,
    apply_semantic_edit,
    legal_semantic_edits,
    semantic_edit_to_json,
)
from .semantic_features import feature_vector


def obligation_edits(obl) -> List[SemanticEdit]:
    if obl.kind == "relation":
        return [SetRelation(obl.symbol, obl.args, v) for v in obl.suggested_values]
    if obl.kind == "function":
        return [SetFunction(obl.symbol, obl.args, v) for v in obl.suggested_values]
    if obl.kind == "constant":
        return [SetConstant(obl.symbol, v) for v in obl.suggested_values]
    return []


def classify(structure, theory, goal, mode, k=None, b=None):
    """(status, dT, dG) with status in won/theory_failed/dead_end/open."""
    dT = run_devil_bounded(structure, theory.clauses, k=k, budget=b)
    if dT.status == "failed":
        return "theory_failed", dT, None
    dG = run_devil_bounded(structure, goal.clauses, k=k, budget=b)
    if mode == "refute":
        if dT.status == "ok" and dG.status == "failed":
            return "won", dT, dG
        if dT.status == "ok" and dG.status == "ok":
            return "dead_end", dT, dG          # claim holds -> cannot refute here
        return "open", dT, dG
    if mode == "satisfy":
        if dT.status == "ok" and dG.status == "ok":
            return "won", dT, dG
        if dT.status == "ok" and dG.status == "failed":
            return "dead_end", dT, dG          # goal violated by set cells -> unfixable
        return "open", dT, dG
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------
# Refute oracle -> supervision for the (small, problem-specific) neural prior.
# Follows the Devil's theory obligations, then assigns the claim-only constant.
# No algebra/3-cycle knowledge is hard-coded.
# --------------------------------------------------------------------------
def _guided_candidates(structure, theory, claim, k, b) -> List[SemanticEdit]:
    status, dT, _ = classify(structure, theory, claim, "refute", k, b)
    if status == "open" and dT.status == "unknown":
        obl = extract_obligation(structure, dT)
        if obl is not None:
            return obligation_edits(obl)
    cands: List[SemanticEdit] = []
    for cname, value in structure.constants.items():
        if value is None:
            cands.extend(SetConstant(cname, d) for d in structure.domain)
    return cands


def oracle_refuting_paths(theory, claim, n, k=None, b=None, max_paths=24, max_depth=12):
    start = PartialStructure.empty(theory.signature, n)
    paths: List[List[SemanticEdit]] = []

    def dfs(structure, path):
        if len(paths) >= max_paths:
            return
        status, _, _ = classify(structure, theory, claim, "refute", k, b)
        if status == "won":
            paths.append(list(path))
            return
        if status in ("theory_failed", "dead_end") or len(path) >= max_depth:
            return
        for edit in _guided_candidates(structure, theory, claim, k, b):
            dfs(apply_semantic_edit(structure, edit), path + [edit])
            if len(paths) >= max_paths:
                return

    dfs(start, [])
    return paths


def build_training_examples(theory, claim, n, k=None, b=None) -> List[dict]:
    """For each (state, chosen-edit) on a refuting path: a feature matrix over the
    FULL legal-edit set, target = index of the chosen edit."""
    examples: List[dict] = []
    seen: set = set()
    for path in oracle_refuting_paths(theory, claim, n, k=k, b=b):
        structure = PartialStructure.empty(theory.signature, n)
        for edit in path:
            legal = legal_semantic_edits(structure)
            if edit in legal:
                dT = run_devil_bounded(structure, theory.clauses, k=k, budget=b)
                obl = extract_obligation(structure, dT) if dT.status == "unknown" else None
                sj = structure.to_json()
                key = (tuple(sorted(sj["relations"].items())),
                       tuple(sorted(sj["functions"].items())),
                       tuple(sorted(sj["constants"].items())),
                       str(semantic_edit_to_json(edit)))
                if key not in seen:
                    seen.add(key)
                    feats = [feature_vector(e, structure, obl) for e in legal]
                    examples.append({"features": feats, "target": legal.index(edit)})
            structure = apply_semantic_edit(structure, edit)
    return examples


# --------------------------------------------------------------------------
# Tree-policy-only PUCT MCTS (no random rollouts).
# `closed` is propagated purely for termination (no infinite loop when the
# bounded frontier is exhausted); it does NOT prune selection, so search
# dynamics match the original ablation engine.
# --------------------------------------------------------------------------
class _Node:
    __slots__ = ("structure", "parent", "edit", "depth", "children", "visits",
                 "total", "status", "reward", "child_edits", "priors", "expanded", "closed")

    def __init__(self, structure, parent, edit, depth):
        self.structure = structure
        self.parent = parent
        self.edit = edit
        self.depth = depth
        self.children: Dict[int, "_Node"] = {}
        self.visits = 0
        self.total = 0.0
        self.status: Optional[str] = None
        self.reward = 0.0
        self.child_edits: List[SemanticEdit] = []
        self.priors: List[float] = []
        self.expanded = False
        self.closed = False


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es) or 1.0
    return [e / s for e in es]


_REWARD = {"won": 1.0, "theory_failed": -1.0, "dead_end": 0.0, "open": 0.0}


def _evaluate(node, theory, goal, mode, prior, k, b):
    status, dT, _ = classify(node.structure, theory, goal, mode, k, b)
    node.status = status
    node.reward = _REWARD[status]
    if status != "open":
        node.closed = True
        return
    edits = legal_semantic_edits(node.structure)
    if not edits:
        node.status = "dead_end"
        node.reward = 0.0
        node.closed = True
        return
    node.child_edits = edits
    devil_for_prior = dT if dT.status == "unknown" else None
    scores = prior.score(node.structure, devil_for_prior, edits)
    node.priors = _softmax([scores.get(e, 0.0) for e in edits])


def _puct(parent, idx, child, c_puct):
    q = (child.total / child.visits) if child.visits > 0 else 0.0
    u = c_puct * parent.priors[idx] * math.sqrt(parent.visits + 1) / (1 + child.visits)
    return q + u


def _expand(node):
    for idx, edit in enumerate(node.child_edits):
        node.children[idx] = _Node(apply_semantic_edit(node.structure, edit),
                                   node, edit, node.depth + 1)
    node.expanded = True


def _path_edits(node) -> List[SemanticEdit]:
    edits, nd = [], node
    while nd.parent is not None:
        edits.append(nd.edit)
        nd = nd.parent
    edits.reverse()
    return edits


def _propagate_closed(node):
    nd = node
    while nd is not None:
        if (not nd.closed) and nd.status == "open" and nd.expanded and nd.children \
                and all(c.closed for c in nd.children.values()):
            nd.closed = True
            nd = nd.parent
        else:
            break


def tree_policy_search(theory, goal, n, prior, *, mode="refute", k=None,
                       devil_budget=None, search_budget=800, c_puct=2.0, seed=0):
    root = _Node(PartialStructure.empty(theory.signature, n), None, None, 0)
    _evaluate(root, theory, goal, mode, prior, k, devil_budget)
    nodes = 1
    winner = root if root.status == "won" else None

    while winner is None and not root.closed and nodes < search_budget:
        node = root
        while node.expanded and node.status == "open":
            idx, node = max(node.children.items(),
                            key=lambda kv: _puct(node, kv[0], kv[1], c_puct))
            if node.status is None:
                _evaluate(node, theory, goal, mode, prior, k, devil_budget)
                nodes += 1
                break
        if node.status == "open" and not node.expanded and node.child_edits:
            _expand(node)
        if node.status == "won":
            winner = node
        value = node.reward
        back = node
        while back is not None:
            back.visits += 1
            back.total += value
            back = back.parent
        _propagate_closed(node)

    base = {"success": winner is not None,
            "success_via": "guided_tree_policy" if winner is not None else None,
            "nodes_evaluated": nodes, "exhausted": root.closed, "depth": None,
            "structure": None, "witness": None,
            "theory_status": None, "goal_status": None, "trace": []}
    if winner is None:
        return base

    edits = _path_edits(winner)
    dT = run_devil_bounded(winner.structure, theory.clauses, k=k, budget=devil_budget)
    dG = run_devil_bounded(winner.structure, goal.clauses, k=k, budget=devil_budget)
    base.update(
        depth=len(edits),
        structure=winner.structure.to_json(),
        theory_status="satisfied" if dT.status == "ok" else dT.status,
        goal_status=dG.status,
        witness=dG.witness.to_json() if dG.witness else None,
        trace=[{"event": "semantic_action", "edit": semantic_edit_to_json(e)} for e in edits],
    )
    return base
