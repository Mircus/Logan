"""MCTS over partial binary-relation structures, with optional neural priors.

State   = PartialStructure (one binary relation)
Action  = set the Devil's current obligation cell (an UNKNOWN R(i,j)) to
          TRUE or FALSE -- branching factor 2; known/seed cells are never
          touched.
Prior   = RelationPolicyNet logits (softmax over the obligation cell's two
          truths) if a model is given, else uniform.
Reward  = symbolic Devil result (see _reward).

This is honest imitation+search: the neural policy supplies priors learned
from Devil/search traces; the symbolic Devil supplies reward and verifies
every node. No rollouts (expansion + immediate Devil evaluation), determin-
istic selection.
"""
from __future__ import annotations

import math
from typing import Optional

from ..core.backtracking import _decision_cell
from ..core.devil import run_devil_bounded
from ..core.partial_structure import PartialStructure
from ..core.theory import Theory
from ..core.types import Truth
from .actions import RelationEdit, encode_action_index


def _reward(result) -> float:
    if result.status == "failed":
        return -1.0
    if result.status == "ok":
        return 0.3 if result.budget_exhausted else 1.0
    return 0.0  # unknown


def _action_priors(structure, relation, result, model, n) -> dict:
    if model is None:
        return {"false": 0.5, "true": 0.5}
    import torch

    from .encoding import encode_state

    i, j = result_cell = _decision_cell(structure, result)[2]
    x = encode_state(structure, relation, result.witness).unsqueeze(0)
    with torch.no_grad():
        logits = model(x).squeeze(0)
    lf = logits[encode_action_index(RelationEdit(relation, (i, j), Truth.FALSE), n)].item()
    lt = logits[encode_action_index(RelationEdit(relation, (i, j), Truth.TRUE), n)].item()
    m = max(lf, lt)
    ef, et = math.exp(lf - m), math.exp(lt - m)
    s = ef + et
    return {"false": ef / s, "true": et / s}


class _Node:
    __slots__ = ("structure", "parent", "action_from_parent", "children", "visits",
                 "total_reward", "devil_status", "terminal", "reward", "obligation",
                 "priors", "expanded")

    def __init__(self, structure, parent, action_from_parent):
        self.structure = structure
        self.parent = parent
        self.action_from_parent = action_from_parent  # RelationEdit | None
        self.children = {}                            # "false"/"true" -> _Node
        self.visits = 0
        self.total_reward = 0.0
        self.devil_status = None
        self.terminal = False
        self.reward = 0.0
        self.obligation = None                        # ("relation", name, (i, j))
        self.priors = {}
        self.expanded = False


def _evaluate(node, theory, relation, k, budget, model, n):
    result = run_devil_bounded(node.structure, theory.clauses, k=k, budget=budget)
    node.devil_status = result.status
    node.reward = _reward(result)
    if result.status in ("ok", "failed"):
        node.terminal = True
        return
    cell = _decision_cell(node.structure, result)
    if cell is None or cell[0] != "relation" or cell[1] != relation:
        node.terminal = True   # no legal relation action -> dead leaf
        node.reward = 0.0
        return
    node.obligation = cell
    node.priors = _action_priors(node.structure, relation, result, model, n)


def _expand(node, relation):
    i, j = node.obligation[2]
    for key, truth in (("false", Truth.FALSE), ("true", Truth.TRUE)):
        child_struct = node.structure.copy()
        child_struct.set_relation(relation, (i, j), truth)
        node.children[key] = _Node(child_struct, node, RelationEdit(relation, (i, j), truth))
    node.expanded = True


def _puct(parent, key, child, c_puct):
    q = (child.total_reward / child.visits) if child.visits > 0 else 0.0
    u = c_puct * parent.priors[key] * math.sqrt(parent.visits + 1) / (1 + child.visits)
    return q + u


def mcts_relation_build(
    theory: Theory,
    relation: str,
    n: int,
    model_path: Optional[str] = None,
    seed_structure: Optional[PartialStructure] = None,
    k: Optional[int] = None,
    budget: Optional[int] = None,
    rollouts: int = 100,
    c_puct: float = 1.5,
    seed: int = 0,
) -> dict:
    model = None
    if model_path is not None:
        from .builder import load_policy

        model, model_n = load_policy(model_path)
        if model_n != n:
            raise ValueError(f"model trained for n={model_n}, requested n={n}")

    start = seed_structure.copy() if seed_structure is not None \
        else PartialStructure.empty(theory.signature, n)
    root = _Node(start, None, None)
    _evaluate(root, theory, relation, k, budget, model, n)
    nodes = 1
    best_node = root if root.devil_status == "ok" else None

    for _ in range(rollouts):
        node = root
        # selection
        while node.expanded and not node.terminal:
            key, node = max(node.children.items(),
                            key=lambda kv: _puct(node, kv[0], kv[1], c_puct))
            if node.devil_status is None:
                _evaluate(node, theory, relation, k, budget, model, n)
                nodes += 1
                break
        # expansion (one ply)
        if (not node.terminal) and (not node.expanded) and node.obligation is not None:
            _expand(node, relation)
        # track best OK terminal
        if node.devil_status == "ok" and (best_node is None or node.reward > best_node.reward):
            best_node = node
        # backup
        value = node.reward
        back = node
        while back is not None:
            back.visits += 1
            back.total_reward += value
            back = back.parent

    # assemble result
    trace = []
    if best_node is not None and best_node is not root:
        chain = []
        nd = best_node
        while nd.parent is not None:
            chain.append(nd)
            nd = nd.parent
        chain.reverse()
        for nd in chain:
            edit = nd.action_from_parent
            key = "true" if edit.value is Truth.TRUE else "false"
            trace.append({
                "event": "mcts_action",
                "edit": {"relation": edit.relation, "args": list(edit.args), "value": edit.value.value},
                "prior": nd.parent.priors.get(key),
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
        "builder": "mcts_relation",
        "uses_neural_policy": model is not None,
        "relation": relation,
        "n": n,
        "k": k,
        "budget": budget,
        "rollouts": rollouts,
        "nodes": nodes,
        "structure": final.to_json(),
        "trace": trace,
    }
