"""A deliberately simple, generic, monotone-fill generator.

It repeatedly asks the Devil for the first non-OK clause instance and
fills exactly one UNKNOWN cell to make progress. *What* truth value an
UNKNOWN premise relation gets, and which domain element fills an unknown
function/constant cell, is delegated to a ``BuilderPolicy`` (default
``SparseHornPolicy``: UNKNOWN premise -> FALSE). This keeps the Devil's
three-valued *semantics* separate from the Builder's construction *policy*.

It never revises a known cell. It returns SATISFIED, UNSAT (with the
failing witness), or UNKNOWN if it can make no monotone progress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    from typing import Literal

    Status = Literal["satisfied", "unsat", "unknown"]
except ImportError:  # pragma: no cover
    Status = str  # type: ignore

from .atoms import EqAtom, RelAtom
from .clauses import HornClause
from .devil import DevilResult, run_devil, run_devil_bounded
from .eval import eval_atom, eval_term
from .partial_structure import PartialStructure
from .policy import BuilderPolicy, DEFAULT_POLICY
from .terms import Const, Func, Term
from .types import Truth


@dataclass
class GenerateResult:
    status: str  # "satisfied" | "unsat" | "unknown"
    structure: PartialStructure
    trace: List[dict] = field(default_factory=list)
    policy: str = ""


def _find_unknown_cell_in_term(
    structure: PartialStructure, term: Term, assignment
) -> Optional[Tuple[str, str, Optional[Tuple[int, ...]]]]:
    """Return the first fillable UNKNOWN cell reachable in `term`.

    ("const", name, None) or ("func", name, argvals). Innermost first.
    """
    if isinstance(term, Const):
        if structure.get_constant(term.name) is None:
            return ("const", term.name, None)
        return None
    if isinstance(term, Func):
        for arg in term.args:
            deep = _find_unknown_cell_in_term(structure, arg, assignment)
            if deep is not None:
                return deep
        argvals = tuple(eval_term(a, assignment, structure) for a in term.args)
        if any(v is None for v in argvals):
            return None
        if structure.get_function(term.name, argvals) is None:
            return ("func", term.name, argvals)
        return None
    return None  # Var: nothing to fill


def _fill_cell(structure: PartialStructure, cell, trace: List[dict], fill_value: int) -> bool:
    kind, name, argvals = cell
    if kind == "const":
        structure.set_constant(name, fill_value)
        trace.append({"event": "edit", "action": "set_constant", "constant": name, "value": fill_value})
    else:
        structure.set_function(name, argvals, fill_value)
        trace.append(
            {"event": "edit", "action": "set_function", "function": name,
             "args": list(argvals), "value": fill_value}
        )
    return True


def _fill_term_unknowns(structure, terms, assignment, trace, fill_value) -> bool:
    for term in terms:
        cell = _find_unknown_cell_in_term(structure, term, assignment)
        if cell is not None:
            return _fill_cell(structure, cell, trace, fill_value)
    return False


def _make_progress(
    structure: PartialStructure, result: DevilResult, trace: List[dict], policy: BuilderPolicy
) -> bool:
    clause = result.clause
    assignment = result.assignment
    witness = result.witness
    assert clause is not None and assignment is not None and witness is not None
    fill_value = policy.fill_value(structure)

    # Case A: an UNKNOWN premise blocked the instance (conclusion not reached).
    if witness.conclusion_value is None:
        for prem in clause.premises:
            if eval_atom(prem, assignment, structure) is not Truth.UNKNOWN:
                continue
            if isinstance(prem, RelAtom):
                args = tuple(eval_term(a, assignment, structure) for a in prem.args)
                if any(v is None for v in args):
                    if _fill_term_unknowns(structure, prem.args, assignment, trace, fill_value):
                        return True
                    continue
                value = policy.unknown_premise_value(structure, prem.relation, args, assignment)
                structure.set_relation(prem.relation, args, value)
                trace.append(
                    {"event": "edit", "action": "set_relation", "relation": prem.relation,
                     "args": list(args), "value": value.value, "policy": policy.name}
                )
                return True
            # EqAtom premise: fill an unknown function/constant cell it touches.
            if _fill_term_unknowns(structure, (prem.left, prem.right), assignment, trace, fill_value):
                return True
        return False

    # Case B: premises all TRUE but the conclusion is UNKNOWN.
    concl = clause.conclusion
    if isinstance(concl, RelAtom):
        args = tuple(eval_term(a, assignment, structure) for a in concl.args)
        if any(v is None for v in args):
            return _fill_term_unknowns(structure, concl.args, assignment, trace, fill_value)
        # A TRUE conclusion is forced (anything else fails the clause), so this
        # is not a policy choice.
        structure.set_relation(concl.relation, args, Truth.TRUE)
        trace.append(
            {"event": "edit", "action": "set_relation", "relation": concl.relation,
             "args": list(args), "value": Truth.TRUE.value, "forced": True}
        )
        return True
    # EqAtom conclusion: fill an unknown function/constant cell it touches.
    return _fill_term_unknowns(structure, (concl.left, concl.right), assignment, trace, fill_value)


def generate(
    structure: PartialStructure,
    clauses: List[HornClause],
    max_steps: int = 1000,
    policy: Optional[BuilderPolicy] = None,
    k: Optional[int] = None,
    budget: Optional[int] = None,
) -> GenerateResult:
    policy = policy if policy is not None else DEFAULT_POLICY()
    bounded = k is not None or budget is not None
    structure = structure.copy()
    trace: List[dict] = [{"event": "start", "policy": policy.name}]
    for step in range(max_steps):
        if bounded:
            result = run_devil_bounded(structure, clauses, k=k, budget=budget)
            trace.append(
                {"step": step, "event": "challenge", "status": result.status,
                 "clause": None if result.witness is None else result.witness.clause_name,
                 "k": k, "budget": budget,
                 "checked_instances": result.checked_instances,
                 "budget_exhausted": result.budget_exhausted,
                 "skipped_by_depth": result.skipped_by_depth}
            )
        else:
            result = run_devil(structure, clauses)
            trace.append(
                {"step": step, "event": "challenge", "status": result.status,
                 "clause": None if result.witness is None else result.witness.clause_name}
            )
        if result.status == "ok":
            trace.append({"event": "result", "status": "satisfied"})
            return GenerateResult("satisfied", structure, trace, policy.name)
        if result.status == "failed":
            trace.append({"event": "witness", "witness": result.witness.to_json()})
            trace.append({"event": "result", "status": "unsat"})
            return GenerateResult("unsat", structure, trace, policy.name)
        # UNKNOWN: try to make one monotone fill of progress, per policy.
        trace.append({"event": "obligation", "witness": result.witness.to_json()})
        if not _make_progress(structure, result, trace, policy):
            trace.append({"event": "result", "status": "unknown"})
            return GenerateResult("unknown", structure, trace, policy.name)
    trace.append({"event": "result", "status": "unknown"})
    return GenerateResult("unknown", structure, trace, policy.name)
