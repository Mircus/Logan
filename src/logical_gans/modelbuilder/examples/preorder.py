"""Preorder example: signature R/2, axioms reflexive + transitive.

Also provides the antisymmetry claim and a 2-element countermodel
refuting "all preorders are antisymmetric".
"""
from __future__ import annotations

from typing import List

from ..core.atoms import EqAtom, RelAtom
from ..core.clauses import HornClause
from ..core.devil import run_devil
from ..core.partial_structure import PartialStructure
from ..core.signature import Signature
from ..core.terms import Var
from ..core.types import Truth


def preorder_signature() -> Signature:
    return Signature.build(relations=[("R", 2)])


def preorder_clauses() -> List[HornClause]:
    x, y, z = Var("x"), Var("y"), Var("z")
    reflexive = HornClause(
        name="reflexive",
        variables=("x",),
        premises=(),
        conclusion=RelAtom("R", (x, x)),
    )
    transitive = HornClause(
        name="transitive",
        variables=("x", "y", "z"),
        premises=(RelAtom("R", (x, y)), RelAtom("R", (y, z))),
        conclusion=RelAtom("R", (x, z)),
    )
    return [reflexive, transitive]


def antisymmetry_clause() -> HornClause:
    x, y = Var("x"), Var("y")
    return HornClause(
        name="antisymmetric",
        variables=("x", "y"),
        premises=(RelAtom("R", (x, y)), RelAtom("R", (y, x))),
        conclusion=EqAtom(x, y),
    )


def empty_preorder_structure(n: int) -> PartialStructure:
    return PartialStructure.empty(preorder_signature(), n)


def total_preorder_on_two() -> PartialStructure:
    """The 2-element preorder where every pair is related (not antisymmetric)."""
    A = empty_preorder_structure(2)
    for a in (0, 1):
        for b in (0, 1):
            A.set_relation("R", (a, b), Truth.TRUE)
    return A


def antisymmetry_refutation() -> dict:
    """Build a 2-element preorder and refute antisymmetry on it."""
    A = total_preorder_on_two()
    preorder_check = run_devil(A, preorder_clauses())
    refute = run_devil(A, [antisymmetry_clause()])
    return {
        "claim": "all_preorders_are_antisymmetric",
        "preorder_status": preorder_check.status,
        "status": "refuted" if refute.status == "failed" else refute.status,
        "structure": A.to_json(),
        "witness": None if refute.witness is None else refute.witness.to_json(),
    }
