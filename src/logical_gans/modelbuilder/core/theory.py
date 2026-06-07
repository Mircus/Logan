"""Containers for a loaded theory pack and a loaded claim."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .clauses import HornClause
from .signature import Signature


@dataclass
class Theory:
    name: str
    signature: Signature
    clauses: List[HornClause] = field(default_factory=list)


@dataclass
class Claim:
    name: str
    clauses: List[HornClause] = field(default_factory=list)
