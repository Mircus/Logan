"""Semigroup example: signature mul/2, axiom associativity (an equation)."""
from __future__ import annotations

from typing import List

from ..core.atoms import EqAtom
from ..core.clauses import HornClause
from ..core.partial_structure import PartialStructure
from ..core.signature import Signature
from ..core.terms import Func, Var


def semigroup_signature() -> Signature:
    return Signature.build(functions=[("mul", 2)])


def semigroup_clauses() -> List[HornClause]:
    x, y, z = Var("x"), Var("y"), Var("z")
    left = Func("mul", (Func("mul", (x, y)), z))
    right = Func("mul", (x, Func("mul", (y, z))))
    associativity = HornClause(
        name="associativity",
        variables=("x", "y", "z"),
        premises=(),
        conclusion=EqAtom(left, right),
    )
    return [associativity]


def empty_semigroup_structure(n: int) -> PartialStructure:
    return PartialStructure.empty(semigroup_signature(), n)
