"""Logical/term complexity (depth) for the P0 fragment.

    Var / Const     -> 0
    Func            -> 1 + max(arg depths)
    RelAtom         -> max(arg depths)
    EqAtom          -> max(depth(left), depth(right))
    HornClause      -> max depth over premises and conclusion
"""
from __future__ import annotations

from .atoms import EqAtom, RelAtom
from .clauses import HornClause
from .terms import Const, Func, Term, Var


def term_depth(term: Term) -> int:
    if isinstance(term, (Var, Const)):
        return 0
    if isinstance(term, Func):
        return 1 + max((term_depth(a) for a in term.args), default=0)
    raise TypeError(f"unknown term {term!r}")


def atom_depth(atom) -> int:
    if isinstance(atom, EqAtom):
        return max(term_depth(atom.left), term_depth(atom.right))
    if isinstance(atom, RelAtom):
        return max((term_depth(a) for a in atom.args), default=0)
    raise TypeError(f"unknown atom {atom!r}")


def clause_depth(clause: HornClause) -> int:
    depths = [atom_depth(p) for p in clause.premises] + [atom_depth(clause.conclusion)]
    return max(depths) if depths else 0
