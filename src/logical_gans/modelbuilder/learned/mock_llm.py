"""Deterministic mock LLM adapter (no network).

Toy strategies that emit the same JSON shape a real LLM would, so the
protocol/validation pipeline and demos run without any external API.
"""
from __future__ import annotations

import json
from typing import Callable, List, Optional, Sequence

from ..core.partial_structure import PartialStructure
from ..core.types import Truth
from .actions import RelationEdit, legal_relation_edits
from .llm_protocol import (
    edit_to_dict,
    parse_llm_output,
    validate_llm_actions,
)

STRATEGIES = ("first_true", "first_false", "witness_match")


def _first_with_value(allowed: Sequence[RelationEdit], value: Truth) -> Optional[RelationEdit]:
    for e in allowed:
        if e.value is value:
            return e
    return None


def _witness_conclusion_cell(witness):
    """The (i,j) of the witness's conclusion relation atom, if any."""
    atoms = getattr(witness, "touched_atoms", None)
    if atoms is None and isinstance(witness, dict):
        atoms = witness.get("touched_atoms")
    if not atoms:
        return None
    concl = atoms[-1]  # conclusion is appended last
    if concl.get("kind") == "rel":
        vals = concl.get("arg_values")
        if vals and len(vals) == 2 and all(v is not None for v in vals):
            return tuple(vals)
    return None


def _choose(strategy: str, allowed: Sequence[RelationEdit], witness=None) -> Optional[RelationEdit]:
    if strategy == "first_true":
        return _first_with_value(allowed, Truth.TRUE)
    if strategy == "first_false":
        return _first_with_value(allowed, Truth.FALSE)
    if strategy == "witness_match":
        cell = _witness_conclusion_cell(witness)
        if cell is not None:
            for e in allowed:
                if e.args == cell and e.value is Truth.TRUE:
                    return e
        return _first_with_value(allowed, Truth.TRUE)  # fallback
    raise ValueError(f"unknown mock strategy {strategy!r}; choices: {STRATEGIES}")


def mock_llm_json(strategy: str, allowed: Sequence[RelationEdit], witness=None) -> str:
    """Return the JSON text a mock LLM would emit for this state."""
    chosen = _choose(strategy, allowed, witness)
    actions = [edit_to_dict(chosen)] if chosen is not None else []
    return json.dumps({
        "proposed_actions": actions,
        "explanation": f"mock:{strategy} (ignored by the verifier)",
    })


def mock_llm_prior(strategy: str) -> Callable:
    """Build an MCTS llm_prior hook: (structure, relation, devil_result) -> RelationEdit|None.

    Runs the full untrusted pipeline (emit JSON -> parse -> validate against
    the legal actions) and returns the first VALID suggested edit, or None.
    """
    def hook(structure: PartialStructure, relation: str, result) -> Optional[RelationEdit]:
        allowed = legal_relation_edits(structure, relation)
        text = mock_llm_json(strategy, allowed, getattr(result, "witness", None))
        plan = validate_llm_actions(parse_llm_output(text), allowed)
        return plan.validated[0] if plan.validated else None

    return hook
