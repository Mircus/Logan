"""Backtracking search generator.

Unlike the monotone ``generator.generate`` (which commits one cell and never
revises), this performs a deterministic depth-first search that *branches*
over admissible fills for the cell the Devil is currently blocked on, and
backtracks on failure. It is a separate module on purpose; the monotone
generator is left intact.

Branching (P0):
* relation cell UNKNOWN  -> try the two truths in ``policy_order``
                           (default FALSE then TRUE)
* function/constant UNKNOWN -> try every domain element in order

Outcomes:
* satisfied  -> a model was found (with structure + trace)
* unsat      -> the finite search space was exhausted with no model
* unknown    -> the node budget (max_nodes) was exceeded first
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .atoms import EqAtom, RelAtom
from .devil import run_devil, run_devil_bounded
from .eval import eval_atom, eval_term
from .partial_structure import PartialStructure
from .terms import Const, Func, Term
from .theory import Theory
from .types import Truth

_REL_ORDER = {
    ("false", "true"): (Truth.FALSE, Truth.TRUE),
    ("true", "false"): (Truth.TRUE, Truth.FALSE),
}


@dataclass
class BacktrackResult:
    status: str  # "satisfied" | "unsat" | "unknown"
    nodes: int
    structure: PartialStructure
    trace: List[dict] = field(default_factory=list)


# --- finding the cell to branch on ---------------------------------------

def _find_term_cell(structure, term: Term, assignment):
    """First fillable UNKNOWN function/constant cell in `term` (innermost first)."""
    if isinstance(term, Const):
        if structure.get_constant(term.name) is None:
            return ("constant", term.name)
        return None
    if isinstance(term, Func):
        for arg in term.args:
            deep = _find_term_cell(structure, arg, assignment)
            if deep is not None:
                return deep
        argvals = tuple(eval_term(a, assignment, structure) for a in term.args)
        if any(v is None for v in argvals):
            return None
        if structure.get_function(term.name, argvals) is None:
            return ("function", term.name, argvals)
        return None
    return None  # Var


def _first_term_cell(structure, terms, assignment):
    for term in terms:
        cell = _find_term_cell(structure, term, assignment)
        if cell is not None:
            return cell
    return None


def _decision_cell(structure, result):
    """From a Devil UNKNOWN result, the single cell to branch on (or None)."""
    clause = result.clause
    assignment = result.assignment
    witness = result.witness

    if witness.conclusion_value is None:  # an UNKNOWN premise blocked it
        for prem in clause.premises:
            if eval_atom(prem, assignment, structure) is not Truth.UNKNOWN:
                continue
            if isinstance(prem, RelAtom):
                args = tuple(eval_term(a, assignment, structure) for a in prem.args)
                if any(v is None for v in args):
                    cell = _first_term_cell(structure, prem.args, assignment)
                    if cell is not None:
                        return cell
                    continue
                return ("relation", prem.relation, args)
            cell = _first_term_cell(structure, (prem.left, prem.right), assignment)
            if cell is not None:
                return cell
        return None

    concl = clause.conclusion  # premises all TRUE, conclusion UNKNOWN
    if isinstance(concl, RelAtom):
        args = tuple(eval_term(a, assignment, structure) for a in concl.args)
        if any(v is None for v in args):
            return _first_term_cell(structure, concl.args, assignment)
        return ("relation", concl.relation, args)
    return _first_term_cell(structure, (concl.left, concl.right), assignment)


def _admissible_values(cell, structure, rel_order):
    if cell[0] == "relation":
        return list(rel_order)
    return list(structure.domain)  # function / constant


def _apply(structure, cell, value):
    if cell[0] == "relation":
        structure.set_relation(cell[1], cell[2], value)
    elif cell[0] == "function":
        structure.set_function(cell[1], cell[2], value)
    else:
        structure.set_constant(cell[1], value)


def _cell_desc(cell):
    if cell[0] == "relation":
        return {"relation": cell[1], "args": list(cell[2])}
    if cell[0] == "function":
        return {"function": cell[1], "args": list(cell[2])}
    return {"constant": cell[1]}


def _value_desc(value):
    return value.value if isinstance(value, Truth) else value


# --- the search -----------------------------------------------------------

def backtracking_generate(
    theory: Theory,
    n: int,
    *,
    k: Optional[int] = None,
    budget: Optional[int] = None,
    max_nodes: int = 10000,
    policy_order: Tuple[str, str] = ("false", "true"),
) -> BacktrackResult:
    rel_order = _REL_ORDER.get(tuple(policy_order))
    if rel_order is None:
        raise ValueError(f"policy_order must be a permutation of ('false','true'), got {policy_order}")

    bounded = k is not None or budget is not None
    clauses = theory.clauses
    root = PartialStructure.empty(theory.signature, n)
    trace: List[dict] = []
    state = {"nodes": 0}

    def search(structure, depth) -> Tuple[str, Optional[PartialStructure]]:
        state["nodes"] += 1
        if state["nodes"] > max_nodes:
            return ("limit", None)
        node_id = state["nodes"]

        if bounded:
            result = run_devil_bounded(structure, clauses, k=k, budget=budget)
            challenge = {
                "node": node_id, "depth": depth, "event": "challenge",
                "status": result.status,
                "clause": None if result.witness is None else result.witness.clause_name,
                "k": k, "budget": budget,
                "checked_instances": result.checked_instances,
                "budget_exhausted": result.budget_exhausted,
                "skipped_by_depth": result.skipped_by_depth,
            }
        else:
            result = run_devil(structure, clauses)
            challenge = {
                "node": node_id, "depth": depth, "event": "challenge",
                "status": result.status,
                "clause": None if result.witness is None else result.witness.clause_name,
            }
        trace.append(challenge)
        if result.status == "ok":
            return ("sat", structure)
        if result.status == "failed":
            trace.append({"node": node_id, "event": "deadend", "reason": "failed"})
            return ("deadend", None)

        cell = _decision_cell(structure, result)
        if cell is None:
            trace.append({"node": node_id, "event": "deadend", "reason": "no_branch"})
            return ("deadend", None)

        for value in _admissible_values(cell, structure, rel_order):
            child = structure.copy()
            _apply(child, cell, value)
            trace.append({
                "node": node_id, "depth": depth, "event": "branch",
                "cell": _cell_desc(cell), "value": _value_desc(value),
            })
            tag, found = search(child, depth + 1)
            if tag == "sat":
                return ("sat", found)
            if tag == "limit":
                return ("limit", None)
            trace.append({
                "node": node_id, "event": "backtrack",
                "cell": _cell_desc(cell), "value": _value_desc(value),
            })
        return ("deadend", None)

    tag, found = search(root, 0)
    if tag == "sat":
        status, structure = "satisfied", found
    elif tag == "limit":
        status, structure = "unknown", root
    else:
        status, structure = "unsat", root

    trace.append({"event": "result", "status": status, "nodes": state["nodes"]})
    return BacktrackResult(status, state["nodes"], structure, trace)
