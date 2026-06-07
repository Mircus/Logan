"""Three-valued bounded evaluator over partial structures."""
from __future__ import annotations

from typing import Dict, Optional

from .atoms import EqAtom, RelAtom
from .partial_structure import PartialStructure
from .terms import Const, Func, Term, Var, term_str
from .types import Truth

Assignment = Dict[str, int]


def eval_term(term: Term, assignment: Assignment, structure: PartialStructure) -> Optional[int]:
    if isinstance(term, Var):
        return assignment.get(term.name)
    if isinstance(term, Const):
        return structure.get_constant(term.name)
    if isinstance(term, Func):
        argvals = [eval_term(a, assignment, structure) for a in term.args]
        if any(v is None for v in argvals):
            return None
        return structure.get_function(term.name, tuple(argvals))
    raise TypeError(f"unknown term {term!r}")


def eval_atom(atom, assignment: Assignment, structure: PartialStructure) -> Truth:
    if isinstance(atom, EqAtom):
        left = eval_term(atom.left, assignment, structure)
        right = eval_term(atom.right, assignment, structure)
        if left is None or right is None:
            return Truth.UNKNOWN
        return Truth.TRUE if left == right else Truth.FALSE
    if isinstance(atom, RelAtom):
        argvals = [eval_term(a, assignment, structure) for a in atom.args]
        if any(v is None for v in argvals):
            return Truth.UNKNOWN
        return structure.get_relation(atom.relation, tuple(argvals))
    raise TypeError(f"unknown atom {atom!r}")


def ground_atom(atom, assignment: Assignment, structure: PartialStructure) -> dict:
    """A JSON-friendly snapshot of an atom under an assignment, for witnesses."""
    truth = eval_atom(atom, assignment, structure)
    if isinstance(atom, EqAtom):
        return {
            "kind": "eq",
            "left": term_str(atom.left),
            "right": term_str(atom.right),
            "left_value": eval_term(atom.left, assignment, structure),
            "right_value": eval_term(atom.right, assignment, structure),
            "truth": truth.value,
        }
    return {
        "kind": "rel",
        "relation": atom.relation,
        "args": [term_str(a) for a in atom.args],
        "arg_values": [eval_term(a, assignment, structure) for a in atom.args],
        "truth": truth.value,
    }
