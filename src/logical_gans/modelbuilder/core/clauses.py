"""Universal clauses for the P0 fragments.

Fragment A: universal relational Horn clauses (premises -> conclusion).
Fragment B: universal equations (empty premises, EqAtom conclusion).
Both are represented by a single HornClause type; premises may be empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .atoms import EqAtom, RelAtom

Atom = Union[RelAtom, EqAtom]


@dataclass(frozen=True)
class HornClause:
    name: str
    variables: tuple[str, ...]
    premises: tuple[Atom, ...]
    conclusion: Atom
