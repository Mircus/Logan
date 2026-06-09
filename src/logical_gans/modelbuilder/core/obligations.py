"""Generic extraction of the Devil's current obligation.

Turns an UNKNOWN ``DevilResult`` into a signature-agnostic ``DevilObligation``
describing which interpretation cell is blocking and what values could
discharge it -- a relation cell (TRUE/FALSE), a function cell (a domain
value), or a constant (a domain value). No relation/theory is hard-coded.

It reads the offending clause + assignment (carried on the DevilResult) and
the structure; it does not need to rewrite the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    Literal = None  # type: ignore

from .atoms import EqAtom, RelAtom
from .eval import eval_atom, eval_term
from .partial_structure import PartialStructure
from .terms import Const, Func, Term
from .types import Truth
from .witness import Witness


@dataclass(frozen=True)
class DevilObligation:
    kind: str                       # "relation" | "function" | "constant" | "unknown"
    symbol: Optional[str]
    args: Optional[Tuple[int, ...]]
    suggested_values: Tuple[Any, ...]
    source: str                     # "premise" | "conclusion" | "term" | "equality" | "unknown"
    witness: Witness


def _find_term_cell(structure, term: Term, assignment):
    """First fillable UNKNOWN function/constant cell in `term` (innermost-first)."""
    if isinstance(term, Const):
        if structure.get_constant(term.name) is None:
            return ("constant", term.name, None)
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


def _term_obligation(cell, domain, source, witness) -> DevilObligation:
    if cell[0] == "function":
        return DevilObligation("function", cell[1], cell[2], tuple(domain), source, witness)
    return DevilObligation("constant", cell[1], None, tuple(domain), source, witness)


def extract_obligation(structure: PartialStructure, devil_result) -> Optional[DevilObligation]:
    if devil_result.status != "unknown":
        return None
    clause = devil_result.clause
    assignment = devil_result.assignment
    witness = devil_result.witness
    if clause is None or assignment is None:
        return None
    domain = structure.domain

    # premise-block (no conclusion reached) vs conclusion-block
    if witness.conclusion_value is None:
        for prem in clause.premises:
            if eval_atom(prem, assignment, structure) is not Truth.UNKNOWN:
                continue
            if isinstance(prem, RelAtom):
                argvals = tuple(eval_term(a, assignment, structure) for a in prem.args)
                if any(v is None for v in argvals):
                    cell = _first_term_cell(structure, prem.args, assignment)
                    if cell is not None:
                        return _term_obligation(cell, domain, "term", witness)
                    continue
                return DevilObligation("relation", prem.relation, argvals,
                                       (Truth.FALSE, Truth.TRUE), "premise", witness)
            cell = _first_term_cell(structure, (prem.left, prem.right), assignment)
            if cell is not None:
                return _term_obligation(cell, domain, "equality", witness)
        return None

    concl = clause.conclusion
    if isinstance(concl, RelAtom):
        argvals = tuple(eval_term(a, assignment, structure) for a in concl.args)
        if any(v is None for v in argvals):
            cell = _first_term_cell(structure, concl.args, assignment)
            if cell is not None:
                return _term_obligation(cell, domain, "term", witness)
            return None
        # conclusion forced TRUE (premises all TRUE) -> suggest TRUE first
        return DevilObligation("relation", concl.relation, argvals,
                               (Truth.TRUE, Truth.FALSE), "conclusion", witness)

    cell = _first_term_cell(structure, (concl.left, concl.right), assignment)
    if cell is not None:
        return _term_obligation(cell, domain, "equality", witness)
    return None
