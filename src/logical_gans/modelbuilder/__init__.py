"""LOGAN ModelBuilder: a bounded partial finite-model generation kernel.

Given a signature, a finite domain bound, and axioms in a restricted
fragment (universal Horn clauses + universal equations), the Generator
fills unknown interpretation tables while the Devil checks bounded axiom
instances and returns witnesses/obligations.

This package is stdlib-only and does not import torch.
"""
from .core.clauses import HornClause
from .core.devil import DevilResult, run_devil
from .core.generator import GenerateResult, generate
from .core.partial_structure import PartialStructure
from .core.signature import Signature
from .core.types import Truth
from .core.witness import Witness

__all__ = [
    "Truth",
    "Signature",
    "PartialStructure",
    "HornClause",
    "Witness",
    "DevilResult",
    "run_devil",
    "GenerateResult",
    "generate",
]
