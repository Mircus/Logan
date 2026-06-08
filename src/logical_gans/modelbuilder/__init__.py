"""LOGAN ModelBuilder: a bounded partial finite-model generation kernel.

Given a signature, a finite domain bound, and axioms in a restricted
fragment (universal Horn clauses + universal equations), the Generator
fills unknown interpretation tables while the Devil checks bounded axiom
instances and returns witnesses/obligations.

This package is stdlib-only and does not import torch.
"""
from .core.backtracking import BacktrackResult, backtracking_generate
from .core.clauses import HornClause
from .core.depth import atom_depth, clause_depth, term_depth
from .core.devil import DevilResult, run_devil, run_devil_bounded
from .core.generator import GenerateResult, generate
from .core.partial_structure import PartialStructure
from .core.loader import (
    TheoryLoadError,
    load_claim,
    load_structure,
    load_theory,
)
from .core.policy import (
    BuilderPolicy,
    MaximalHornPolicy,
    SparseHornPolicy,
    get_policy,
)
from .core.runner import check, refute, synthesize
from .core.signature import Signature
from .core.theory import Claim, Theory
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
    "run_devil_bounded",
    "term_depth",
    "atom_depth",
    "clause_depth",
    "GenerateResult",
    "generate",
    "BacktrackResult",
    "backtracking_generate",
    "BuilderPolicy",
    "SparseHornPolicy",
    "MaximalHornPolicy",
    "get_policy",
    "Theory",
    "Claim",
    "load_theory",
    "load_claim",
    "load_structure",
    "TheoryLoadError",
    "synthesize",
    "check",
    "refute",
]
