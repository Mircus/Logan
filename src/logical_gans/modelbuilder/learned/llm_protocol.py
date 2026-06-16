"""Strict protocol for LLM-proposed semantic edits.

The LLM is NOT trusted. It proposes ``set_relation`` actions as JSON; this
module parses and validates them against the legal action set, and only
validated edits may be applied. The Devil verifies the result afterwards.
The LLM's free-text ``explanation`` is ignored by the verifier.

This module is stdlib-only (no torch, no network).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.partial_structure import PartialStructure
from ..core.types import Truth
from .actions import RelationEdit, legal_relation_edits


class LLMProtocolError(ValueError):
    """Raised when LLM output is malformed (not when an action is merely illegal)."""


@dataclass(frozen=True)
class LLMProposedAction:
    kind: str
    relation: Optional[str]
    args: Optional[tuple]
    value: Any  # raw; coerced/validated later


@dataclass
class LLMBuilderOutput:
    proposed_actions: List[LLMProposedAction]
    explanation: str = ""


@dataclass
class LLMBuilderInput:
    theory_name: str
    signature_summary: Dict[str, Any]
    current_structure: Dict[str, Any]
    last_witness: Optional[Dict[str, Any]]
    allowed_actions: List[Dict[str, Any]]
    goal: str  # "satisfy_theory" | "refute_claim"
    k: Optional[int]
    budget: Optional[int]

    def to_json(self) -> dict:
        return {
            "theory_name": self.theory_name,
            "signature_summary": self.signature_summary,
            "current_structure": self.current_structure,
            "last_witness": self.last_witness,
            "allowed_actions": self.allowed_actions,
            "goal": self.goal,
            "k": self.k,
            "budget": self.budget,
        }


@dataclass
class ValidatedLLMPlan:
    validated: List[RelationEdit]
    rejected: List[Dict[str, Any]] = field(default_factory=list)


def edit_to_dict(edit: RelationEdit) -> dict:
    return {"kind": "set_relation", "relation": edit.relation,
            "args": list(edit.args), "value": edit.value.value}


def _action_to_dict(a: LLMProposedAction) -> dict:
    return {"kind": a.kind, "relation": a.relation,
            "args": list(a.args) if a.args is not None else None, "value": a.value}


def _coerce_truth(value: Any) -> Optional[Truth]:
    if value is True:
        return Truth.TRUE
    if value is False:
        return Truth.FALSE
    if value == "true":
        return Truth.TRUE
    if value == "false":
        return Truth.FALSE
    return None


def parse_llm_output(json_text: str) -> LLMBuilderOutput:
    try:
        obj = json.loads(json_text)
    except (json.JSONDecodeError, TypeError) as e:
        raise LLMProtocolError(f"malformed JSON: {e}")
    if not isinstance(obj, dict):
        raise LLMProtocolError("LLM output must be a JSON object")
    raw = obj.get("proposed_actions")
    if not isinstance(raw, list):
        raise LLMProtocolError("'proposed_actions' must be a list")
    actions: List[LLMProposedAction] = []
    for a in raw:
        if not isinstance(a, dict):
            raise LLMProtocolError(f"each action must be an object: {a!r}")
        args = a.get("args")
        args_t = tuple(args) if isinstance(args, list) else None
        actions.append(LLMProposedAction(
            kind=a.get("kind"), relation=a.get("relation"), args=args_t, value=a.get("value"),
        ))
    explanation = obj.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = ""
    return LLMBuilderOutput(proposed_actions=actions, explanation=explanation)


def validate_llm_actions(
    output: LLMBuilderOutput, allowed_actions: Sequence[RelationEdit]
) -> ValidatedLLMPlan:
    """Validate proposed actions against the legal action set. Explanation ignored."""
    allowed = set(allowed_actions)
    plan = ValidatedLLMPlan(validated=[], rejected=[])
    for a in output.proposed_actions:
        if a.kind != "set_relation":
            plan.rejected.append({"action": _action_to_dict(a), "reason": "unknown_action_kind"})
            continue
        truth = _coerce_truth(a.value)
        if truth is None:
            plan.rejected.append({"action": _action_to_dict(a), "reason": "invalid_truth_value"})
            continue
        if not (isinstance(a.relation, str) and a.args is not None and len(a.args) == 2
                and all(isinstance(x, int) and not isinstance(x, bool) for x in a.args)):
            plan.rejected.append({"action": _action_to_dict(a), "reason": "malformed_action"})
            continue
        edit = RelationEdit(a.relation, tuple(a.args), truth)
        if edit not in allowed:
            # not legal: out-of-range, a known/seed cell, or simply not offered
            plan.rejected.append({"action": _action_to_dict(a), "reason": "not_in_allowed_actions"})
            continue
        plan.validated.append(edit)
    return plan


def apply_validated_llm_action(structure: PartialStructure, action: RelationEdit) -> PartialStructure:
    """Apply a *validated* edit to a COPY (the original is untouched)."""
    s = structure.copy()
    s.set_relation(action.relation, action.args, action.value)
    return s


def build_llm_input(theory, relation, structure, witness, goal="satisfy_theory",
                    k=None, budget=None) -> LLMBuilderInput:
    sig = theory.signature
    summary = {
        "relations": {n: r.arity for n, r in sig.relations.items()},
        "functions": {n: f.arity for n, f in sig.functions.items()},
        "constants": list(sig.constants),
    }
    allowed = [edit_to_dict(e) for e in legal_relation_edits(structure, relation)]
    witness_json = None
    if witness is not None:
        witness_json = witness.to_json() if hasattr(witness, "to_json") else witness
    return LLMBuilderInput(
        theory_name=theory.name,
        signature_summary=summary,
        current_structure=structure.to_json(),
        last_witness=witness_json,
        allowed_actions=allowed,
        goal=goal,
        k=k,
        budget=budget,
    )
