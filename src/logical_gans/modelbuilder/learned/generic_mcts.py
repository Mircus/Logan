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
from itertools import product
from typing import List, Optional

from ..core.atoms import RelAtom
from ..core.devil import run_devil_bounded
from ..core.eval import eval_term
from ..core.obligations import extract_obligation
from ..core.partial_structure import PartialStructure
from ..core.theory import Claim, Theory
from ..core.types import Truth
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


# --------------------------------------------------------------------------
# Countermodel (refute) search: find A with A |= theory and A |/= claim.
# --------------------------------------------------------------------------

def refute_candidates(structure, theory, claim, dT) -> List[SemanticEdit]:
    """Edits that either make progress toward satisfying the theory, or set a
    claim-conclusion cell to the value that falsifies the claim."""
    cands: List[SemanticEdit] = []
    if dT.status == "unknown":
        obl = extract_obligation(structure, dT)
        if obl is not None:
            cands.extend(_obligation_edits(obl))
    domain = structure.domain
    for clause in claim.clauses:
        for values in product(domain, repeat=len(clause.variables)):
            assignment = dict(zip(clause.variables, values))
            concl = clause.conclusion
            if isinstance(concl, RelAtom):
                argvals = tuple(eval_term(a, assignment, structure) for a in concl.args)
                if all(v is not None for v in argvals) and \
                        structure.get_relation(concl.relation, argvals) is Truth.UNKNOWN:
                    cands.append(SetRelation(concl.relation, argvals, Truth.FALSE))
    # de-duplicate, preserve order
    seen, unique = set(), []
    for e in cands:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique


def _evaluate_refute(node, theory, claim, k, budget, prior_hook):
    dT = run_devil_bounded(node.structure, theory.clauses, k=k, budget=budget)
    if dT.status == "failed":
        node.devil_status = "theory_failed"
        node.terminal = True
        node.reward = -1.0
        return
    dC = run_devil_bounded(node.structure, claim.clauses, k=k, budget=budget)
    if dT.status == "ok" and dC.status == "failed":
        node.devil_status = "refuted"
        node.terminal = True
        node.reward = 1.0
        return
    cands = refute_candidates(node.structure, theory, claim, dT)
    if not cands:
        node.devil_status = "dead_end"
        node.terminal = True
        node.reward = 0.0
        return
    node.devil_status = dT.status
    node.child_edits = cands
    if prior_hook is None:
        node.priors = [1.0 / len(cands)] * len(cands)
    else:
        scores = prior_hook.score(node.structure, dT, cands)
        node.priors = _softmax([scores.get(e, 0.0) for e in cands])


SIMS_PER_EXPANSION = 12


def _path_edits(node) -> List[SemanticEdit]:
    edits = []
    nd = node
    while nd.parent is not None:
        edits.append(nd.action_from_parent)
        nd = nd.parent
    edits.reverse()
    return edits


def _simulate_refute(structure, theory, claim, k, budget, rng, max_depth=30):
    """Random rollout to a terminal; returns (reward, edits, refuted_structure|None)."""
    A = structure
    edits = []
    for _ in range(max_depth):
        dT = run_devil_bounded(A, theory.clauses, k=k, budget=budget)
        if dT.status == "failed":
            return -1.0, edits, None
        dC = run_devil_bounded(A, claim.clauses, k=k, budget=budget)
        if dT.status == "ok" and dC.status == "failed":
            return 1.0, edits, A
        cands = refute_candidates(A, theory, claim, dT)
        if not cands:
            return 0.0, edits, None
        edit = rng.choice(cands)
        edits.append(edit)
        A = apply_semantic_edit(A, edit)
    return 0.0, edits, None


def mcts_semantic_refute(
    theory: Theory,
    claim: Claim,
    n: int,
    prior_provider=None,
    seed_structure: Optional[PartialStructure] = None,
    k: Optional[int] = None,
    budget: Optional[int] = None,
    rollouts: int = 500,
    c_puct: float = 1.5,
    seed: int = 0,
) -> dict:
    import random

    rng = random.Random(seed)
    start = seed_structure.copy() if seed_structure is not None \
        else PartialStructure.empty(theory.signature, n)
    root = _Node(start, None, None)
    _evaluate_refute(root, theory, claim, k, budget, prior_provider)
    nodes = 1

    best_path: Optional[List[SemanticEdit]] = None
    best_struct: Optional[PartialStructure] = None
    if root.devil_status == "refuted":
        best_path, best_struct = [], root.structure

    for _ in range(rollouts):
        if best_struct is not None:
            break
        node = root
        while node.expanded and not node.terminal:
            idx, node = max(node.children.items(),
                            key=lambda kv: _puct(kv[1].parent, kv[0], kv[1], c_puct))
            if node.devil_status is None:
                _evaluate_refute(node, theory, claim, k, budget, prior_provider)
                nodes += 1
                break
        if (not node.terminal) and (not node.expanded) and node.child_edits:
            _expand(node)

        if node.terminal:
            value = node.reward
            if node.devil_status == "refuted":
                best_path, best_struct = _path_edits(node), node.structure
        else:
            # several random simulations per expansion -> robust to prior quality
            total, count = 0.0, 0
            for _ in range(SIMS_PER_EXPANSION):
                r, sim_edits, sim_struct = _simulate_refute(
                    node.structure, theory, claim, k, budget, rng)
                total += r
                count += 1
                if sim_struct is not None:
                    best_path = _path_edits(node) + sim_edits
                    best_struct = sim_struct
                    break
            value = total / count

        back = node
        while back is not None:
            back.visits += 1
            back.total_reward += value
            back = back.parent

    uses_neural = bool(getattr(prior_provider, "is_neural", False))

    if best_struct is None:
        return {
            "status": "not_found", "builder": "neural_semantic_mcts",
            "uses_neural_policy": uses_neural, "n": n, "rollouts": rollouts, "nodes": nodes,
            "theory_status": None, "claim_status": None,
            "structure": root.structure.to_json(), "claim_witness": None, "trace": [],
        }

    # Independent symbolic verification of the found structure.
    from ..core.devil import run_devil
    theory_status = "satisfied" if run_devil(best_struct, theory.clauses).status == "ok" else "unsatisfied"
    claim_devil = run_devil(best_struct, claim.clauses)

    # Trace: replay the winning edit sequence, recording the Devil status after each.
    trace = []
    replay = PartialStructure.empty(theory.signature, n) if seed_structure is None else seed_structure.copy()
    for edit in (best_path or []):
        replay = apply_semantic_edit(replay, edit)
        dT = run_devil_bounded(replay, theory.clauses, k=k, budget=budget)
        trace.append({
            "event": "mcts_semantic_action",
            "edit": semantic_edit_to_json(edit),
            "devil_status": dT.status,
        })

    return {
        "status": "refuted",
        "builder": "neural_semantic_mcts",
        "uses_neural_policy": uses_neural,
        "n": n,
        "rollouts": rollouts,
        "nodes": nodes,
        "theory_status": theory_status,
        "claim_status": "failed" if claim_devil.status == "failed" else claim_devil.status,
        "structure": best_struct.to_json(),
        "claim_witness": claim_devil.witness.to_json() if claim_devil.witness else None,
        "trace": trace,
    }
