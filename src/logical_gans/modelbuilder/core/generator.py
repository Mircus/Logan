"""A deliberately simple, generic, monotone-fill generator.

It repeatedly asks the Devil for the first non-OK clause instance and
fills exactly one UNKNOWN cell to make progress:

* conclusion relation UNKNOWN  -> set that relation cell TRUE
* premise relation UNKNOWN     -> set that relation cell FALSE (discharge
                                  the clause vacuously, monotone)
* an equality/relation term has an UNKNOWN function/constant cell
                               -> fill it with the smallest domain element

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
from .devil import DevilResult, run_devil
from .eval import eval_term
from .partial_structure import PartialStructure
from .terms import Const, Func, Term
from .types import Truth


@dataclass
class GenerateResult:
    status: str  # "satisfied" | "unsat" | "unknown"
    structure: PartialStructure
    trace: List[dict] = field(default_factory=list)


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


def _fill_cell(structure: PartialStructure, cell, trace: List[dict]) -> bool:
    kind, name, argvals = cell
    value = structure.domain[0]  # smallest domain element
    if kind == "const":
        structure.set_constant(name, value)
        trace.append({"event": "edit", "action": "set_constant", "constant": name, "value": value})
    else:
        structure.set_function(name, argvals, value)
        trace.append(
            {"event": "edit", "action": "set_function", "function": name,
             "args": list(argvals), "value": value}
        )
    return True


def _fill_term_unknowns(structure, terms, assignment, trace) -> bool:
    for term in terms:
        cell = _find_unknown_cell_in_term(structure, term, assignment)
        if cell is not None:
            return _fill_cell(structure, cell, trace)
    return False


def _make_progress(structure: PartialStructure, result: DevilResult, trace: List[dict]) -> bool:
    clause = result.clause
    assignment = result.assignment
    witness = result.witness
    assert clause is not None and assignment is not None and witness is not None

    # Case A: an UNKNOWN premise blocked the instance (conclusion not reached).
    if witness.conclusion_value is None:
        for prem in clause.premises:
            from .eval import eval_atom

            if eval_atom(prem, assignment, structure) is not Truth.UNKNOWN:
                continue
            if isinstance(prem, RelAtom):
                args = tuple(eval_term(a, assignment, structure) for a in prem.args)
                if any(v is None for v in args):
                    if _fill_term_unknowns(structure, prem.args, assignment, trace):
                        return True
                    continue
                structure.set_relation(prem.relation, args, Truth.FALSE)
                trace.append(
                    {"event": "edit", "action": "set_relation_false",
                     "relation": prem.relation, "args": list(args)}
                )
                return True
            # EqAtom premise: fill an unknown function/constant cell it touches.
            if _fill_term_unknowns(structure, (prem.left, prem.right), assignment, trace):
                return True
        return False

    # Case B: premises all TRUE but the conclusion is UNKNOWN.
    concl = clause.conclusion
    if isinstance(concl, RelAtom):
        args = tuple(eval_term(a, assignment, structure) for a in concl.args)
        if any(v is None for v in args):
            return _fill_term_unknowns(structure, concl.args, assignment, trace)
        structure.set_relation(concl.relation, args, Truth.TRUE)
        trace.append(
            {"event": "edit", "action": "set_relation_true",
             "relation": concl.relation, "args": list(args)}
        )
        return True
    # EqAtom conclusion: fill an unknown function/constant cell it touches.
    return _fill_term_unknowns(structure, (concl.left, concl.right), assignment, trace)


def generate(
    structure: PartialStructure, clauses: List[HornClause], max_steps: int = 1000
) -> GenerateResult:
    structure = structure.copy()
    trace: List[dict] = []
    for step in range(max_steps):
        result = run_devil(structure, clauses)
        trace.append(
            {"step": step, "event": "challenge", "status": result.status,
             "clause": None if result.witness is None else result.witness.clause_name}
        )
        if result.status == "ok":
            trace.append({"event": "result", "status": "satisfied"})
            return GenerateResult("satisfied", structure, trace)
        if result.status == "failed":
            trace.append({"event": "witness", "witness": result.witness.to_json()})
            trace.append({"event": "result", "status": "unsat"})
            return GenerateResult("unsat", structure, trace)
        # UNKNOWN: try to make one monotone fill of progress.
        trace.append({"event": "obligation", "witness": result.witness.to_json()})
        if not _make_progress(structure, result, trace):
            trace.append({"event": "result", "status": "unknown"})
            return GenerateResult("unknown", structure, trace)
    trace.append({"event": "result", "status": "unknown"})
    return GenerateResult("unknown", structure, trace)
