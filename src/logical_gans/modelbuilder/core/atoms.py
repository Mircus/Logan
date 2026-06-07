"""Atomic formulas: relation atoms and equalities."""
from __future__ import annotations

from dataclasses import dataclass

from .terms import Term


@dataclass(frozen=True)
class RelAtom:
    relation: str
    args: tuple[Term, ...]


@dataclass(frozen=True)
class EqAtom:
    left: Term
    right: Term
