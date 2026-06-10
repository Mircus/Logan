"""Gate 2 -- neural semantic ablation on a harder mixed-signature countermodel.

Problem
-------
    Sigma = { E/2, s/1, a }
    T     = { forall x E(x, s(x)) ;  forall x s(s(s(x))) = x }
    C     = { s(a) = a }
    n     = 3

A refuting model must satisfy T while breaking C, i.e. s(a) != a. Since T forces
s to be an order-3 permutation (s^3 = id), refuting C forces s to be a
fixed-point-free 3-cycle (the identity satisfies T but also C). The constant a
does not appear in T and C is an *equation*, so neither the Devil's obligation
mechanism nor a naive claim-cell heuristic surfaces the winning edits for free.
That is what makes a *learned* prior worth something here.

Ablation (identical budget, identical seed, identical Devil verification):
    * uniform MCTS
    * obligation-first MCTS
    * neural-prior MCTS

Search policy -- TREE POLICY ONLY
---------------------------------
Random simulation rollouts are DISABLED. A node's backed-up value is its own
symbolic-Devil reward (terminal +1 refuted / -1 theory-failed / 0 dead-end,
0 for interior nodes); selection is pure PUCT (Q + prior*U). So a refutation
can only be reached by the prior-guided tree descent -- never by a lucky random
rollout. ``random_rollout_success`` is reported as ``disabled`` and the gate
counts a success only when ``success_via == "guided_tree_policy"``.

Candidates at every node are the FULL legal semantic-edit set over the whole
signature (high branching), so the prior -- not the candidate generator -- is
what makes the search efficient.

Run:    PYTHONPATH=src python experiments/neural_semantic_ablation_gate.py
Output: results/neural_semantic_ablation_gate.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_claim, load_theory
from logical_gans.modelbuilder.core.obligations import extract_obligation
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.learned.priors import (
    NeuralSemanticPrior,
    ObligationFirstPrior,
    UniformPrior,
)
from logical_gans.modelbuilder.learned.semantic_actions import (
    SemanticEdit,
    SetConstant,
    SetFunction,
    SetRelation,
    apply_semantic_edit,
    legal_semantic_edits,
    semantic_edit_to_json,
)
from logical_gans.modelbuilder.learned.semantic_features import feature_vector
from logical_gans.modelbuilder.learned.semantic_training import (
    train_semantic_policy,
    write_semantic_training_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "examples" / "theories" / "cycle3_succ.json"
CLAIM = ROOT / "examples" / "claims" / "succ_fixes_a.json"
DATA = ROOT / "results" / "training" / "cycle3_ablation_semantic.jsonl"
MODEL = ROOT / "models" / "cycle3_ablation_semantic_policy.pt"
OUT = ROOT / "results" / "neural_semantic_ablation_gate.json"

N = 3
BUDGET = 800          # max Devil-evaluated nodes per arm (identical across arms)
C_PUCT = 2.0
SEED = 0


# --------------------------------------------------------------------------
# Shared obligation -> edit helper (mirrors generic_mcts, kept local so the
# experiment touches no private kernel API).
# --------------------------------------------------------------------------
def _obligation_edits(obl) -> List[SemanticEdit]:
    if obl.kind == "relation":
        return [SetRelation(obl.symbol, obl.args, v) for v in obl.suggested_values]
    if obl.kind == "function":
        return [SetFunction(obl.symbol, obl.args, v) for v in obl.suggested_values]
    if obl.kind == "constant":
        return [SetConstant(obl.symbol, v) for v in obl.suggested_values]
    return []


def _classify(structure, theory, claim):
    """(status, dT, dC) where status in refuted/theory_failed/dead_end/open."""
    dT = run_devil(structure, theory.clauses)
    if dT.status == "failed":
        return "theory_failed", dT, None
    dC = run_devil(structure, claim.clauses)
    if dT.status == "ok" and dC.status == "failed":
        return "refuted", dT, dC
    if dT.status == "ok" and dC.status == "ok":
        return "dead_end", dT, dC          # claim satisfied -> cannot refute
    return "open", dT, dC


# --------------------------------------------------------------------------
# Oracle: enumerate refuting trajectories to supervise the neural prior.
# Generic -- it follows the Devil's theory obligations, then assigns the
# claim-only constant. No 3-cycle / group knowledge is hard-coded.
# --------------------------------------------------------------------------
def _guided_candidates(structure, theory, claim) -> List[SemanticEdit]:
    status, dT, _ = _classify(structure, theory, claim)
    if status == "open" and dT.status == "unknown":
        obl = extract_obligation(structure, dT)
        if obl is not None:
            return _obligation_edits(obl)
    # theory satisfied but claim still open -> assign undefined constants
    cands: List[SemanticEdit] = []
    for cname, value in structure.constants.items():
        if value is None:
            cands.extend(SetConstant(cname, d) for d in structure.domain)
    return cands


def oracle_refuting_paths(theory, claim, n, max_paths=24, max_depth=12):
    start = PartialStructure.empty(theory.signature, n)
    paths: List[List[SemanticEdit]] = []

    def dfs(structure, path):
        if len(paths) >= max_paths:
            return
        status, _, _ = _classify(structure, theory, claim)
        if status == "refuted":
            paths.append(list(path))
            return
        if status in ("theory_failed", "dead_end") or len(path) >= max_depth:
            return
        for edit in _guided_candidates(structure, theory, claim):
            dfs(apply_semantic_edit(structure, edit), path + [edit])
            if len(paths) >= max_paths:
                return

    dfs(start, [])
    return paths


def build_training_examples(theory, claim, n) -> List[dict]:
    """For each (state, chosen-edit) on a refuting path, a feature matrix over
    the FULL legal-edit set with the chosen edit's index as the target."""
    examples: List[dict] = []
    seen: set = set()
    for path in oracle_refuting_paths(theory, claim, n):
        structure = PartialStructure.empty(theory.signature, n)
        for edit in path:
            legal = legal_semantic_edits(structure)
            if edit in legal:
                dT = run_devil(structure, theory.clauses)
                obl = extract_obligation(structure, dT) if dT.status == "unknown" else None
                key = (tuple(sorted(structure.to_json()["relations"].items())),
                       tuple(sorted((k, v) for k, v in structure.to_json()["functions"].items())),
                       tuple(sorted(structure.to_json()["constants"].items())),
                       semantic_edit_to_json(edit).__str__())
                if key not in seen:
                    seen.add(key)
                    feats = [feature_vector(e, structure, obl) for e in legal]
                    examples.append({"features": feats, "target": legal.index(edit)})
            structure = apply_semantic_edit(structure, edit)
    return examples


# --------------------------------------------------------------------------
# Tree-policy-only PUCT MCTS (no random rollouts).
# --------------------------------------------------------------------------
class _Node:
    __slots__ = ("structure", "parent", "edit", "children", "visits", "total",
                 "status", "reward", "child_edits", "priors", "expanded", "depth")

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


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es) or 1.0
    return [e / s for e in es]


_REWARD = {"refuted": 1.0, "theory_failed": -1.0, "dead_end": 0.0, "open": 0.0}


def _evaluate(node, theory, claim, prior):
    status, dT, _ = _classify(node.structure, theory, claim)
    node.status = status
    node.reward = _REWARD[status]
    if status != "open":
        return
    edits = legal_semantic_edits(node.structure)
    if not edits:
        node.status = "dead_end"
        node.reward = 0.0
        return
    node.child_edits = edits
    devil_for_prior = dT if dT.status == "unknown" else None
    scores = prior.score(node.structure, devil_for_prior, edits)
    node.priors = _softmax([scores.get(e, 0.0) for e in edits])


def _puct(parent, idx, child):
    q = (child.total / child.visits) if child.visits > 0 else 0.0
    u = C_PUCT * parent.priors[idx] * math.sqrt(parent.visits + 1) / (1 + child.visits)
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


def tree_policy_search(theory, claim, n, prior, budget=BUDGET):
    """Returns the arm result. Refutation can only come from the tree descent
    reaching a terminal 'refuted' node -- no random rollouts."""
    root = _Node(PartialStructure.empty(theory.signature, n), None, None, 0)
    _evaluate(root, theory, claim, prior)
    nodes = 1
    winner = root if root.status == "refuted" else None

    while winner is None and nodes < budget:
        node = root
        while node.expanded and node.status == "open":
            idx, node = max(node.children.items(), key=lambda kv: _puct(node, kv[0], kv[1]))
            if node.status is None:
                _evaluate(node, theory, claim, prior)
                nodes += 1
                break
        if node.status == "open" and not node.expanded and node.child_edits:
            _expand(node)
        if node.status == "refuted":
            winner = node
        value = node.reward
        back = node
        while back is not None:
            back.visits += 1
            back.total += value
            back = back.parent

    if winner is None:
        return {"success": False, "success_via": None, "nodes_evaluated": nodes,
                "depth": None, "structure": None, "trace": [],
                "theory_status": None, "claim_status": None, "claim_witness": None}

    # Independent symbolic re-verification of the winning structure.
    edits = _path_edits(winner)
    theory_dev = run_devil(winner.structure, theory.clauses)
    claim_dev = run_devil(winner.structure, claim.clauses)
    trace = []
    replay = PartialStructure.empty(theory.signature, n)
    for edit in edits:
        replay = apply_semantic_edit(replay, edit)
        trace.append({"event": "semantic_action",
                      "edit": semantic_edit_to_json(edit),
                      "theory_status": run_devil(replay, theory.clauses).status})
    return {
        "success": True,
        "success_via": "guided_tree_policy",
        "nodes_evaluated": nodes,
        "depth": len(edits),
        "theory_status": "satisfied" if theory_dev.status == "ok" else "unsatisfied",
        "claim_status": "failed" if claim_dev.status == "failed" else claim_dev.status,
        "structure": winner.structure.to_json(),
        "claim_witness": claim_dev.witness.to_json() if claim_dev.witness else None,
        "trace": trace,
    }


def _structure_has_all_kinds(struct) -> bool:
    if struct is None:
        return False
    has_rel = any(v in ("true", "false") for v in struct["relations"].values())
    has_fn = any(v is not None for v in struct["functions"].values())
    has_const = any(v is not None for v in struct["constants"].values())
    return has_rel and has_fn and has_const


def main(epochs: int = 150) -> int:
    theory = load_theory(THEORY)
    claim = load_claim(CLAIM)

    # 1. oracle traces -> train the neural semantic prior.
    examples = build_training_examples(theory, claim, N)
    write_semantic_training_jsonl(examples, DATA)
    train_meta = train_semantic_policy(str(DATA), str(MODEL), epochs=epochs, seed=SEED)

    # 2. three arms, identical budget / seed / Devil verification, rollouts off.
    arms = {
        "uniform": UniformPrior(),
        "obligation_first": ObligationFirstPrior(),
        "neural": NeuralSemanticPrior(str(MODEL)),
    }
    results = {name: tree_policy_search(theory, claim, N, prior) for name, prior in arms.items()}

    neural = results["neural"]
    uni = results["uniform"]
    obl = results["obligation_first"]

    def worse(base) -> bool:
        return (not base["success"]) or (
            neural["success"] and base["nodes_evaluated"] >= 3 * neural["nodes_evaluated"])

    neural_ok = (
        neural["success"]
        and neural["success_via"] == "guided_tree_policy"
        and neural["theory_status"] == "satisfied"
        and neural["claim_status"] == "failed"
        and neural["claim_witness"] is not None
        and _structure_has_all_kinds(neural["structure"])
    )
    accepted = bool(neural_ok and worse(uni) and worse(obl))

    out = {
        "gate": "neural_semantic_ablation",
        "problem": {
            "signature": "E/2, s/1, a",
            "theory": "forall x E(x,s(x)); forall x s(s(s(x)))=x",
            "claim": "s(a)=a",
            "n": N,
        },
        "protocol": {
            "budget_nodes": BUDGET,
            "c_puct": C_PUCT,
            "seed": SEED,
            "random_rollouts": "disabled",
            "success_definition": "terminal 'refuted' reached by PUCT tree descent",
            "devil_verification": "exhaustive run_devil on theory and claim",
            "training_examples": train_meta["examples"],
            "training_final_loss": train_meta["final_loss"],
        },
        "arms": {
            "uniform": uni,
            "obligation_first": obl,
            "neural": neural,
        },
        "accepted": accepted,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"neural: success={neural['success']} via={neural['success_via']} "
          f"nodes={neural['nodes_evaluated']} theory={neural['theory_status']} "
          f"claim={neural['claim_status']}")
    print(f"uniform: success={uni['success']} nodes={uni['nodes_evaluated']}")
    print(f"obligation_first: success={obl['success']} nodes={obl['nodes_evaluated']}")
    print(f"accepted={accepted} -> {OUT}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
