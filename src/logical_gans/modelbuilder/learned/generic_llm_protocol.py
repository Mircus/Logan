"""Strict LLM protocol over GENERIC semantic edits (no live API).

Supported JSON actions:
    {"kind": "set_relation", "symbol": "R", "args": [0,1], "value": "true"}
    {"kind": "set_function", "symbol": "f", "args": [0,1], "value": 1}
    {"kind": "set_constant", "symbol": "c", "value": 0}

The LLM is untrusted: proposals are parsed and validated against the legal
semantic-edit set. The free-text explanation is ignored. stdlib-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from ..core.types import Truth
from .semantic_actions import SemanticEdit, SetConstant, SetFunction, SetRelation


class GenericLLMProtocolError(ValueError):
    """Raised when the LLM output envelope is malformed (not for illegal actions)."""


@dataclass(frozen=True)
class GenericProposedAction:
    kind: Any
    symbol: Any
    args: Any
    value: Any


@dataclass
class GenericLLMOutput:
    proposed_actions: List[GenericProposedAction]
    explanation: str = ""


@dataclass
class GenericValidatedPlan:
    validated: List[SemanticEdit]
    rejected: List[Dict[str, Any]] = field(default_factory=list)


def _coerce_truth(value):
    if value is True or value == "true":
        return Truth.TRUE
    if value is False or value == "false":
        return Truth.FALSE
    return None


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def parse_generic_llm_output(json_text: str) -> GenericLLMOutput:
    try:
        obj = json.loads(json_text)
    except (json.JSONDecodeError, TypeError) as e:
        raise GenericLLMProtocolError(f"malformed JSON: {e}")
    if not isinstance(obj, dict):
        raise GenericLLMProtocolError("output must be a JSON object")
    raw = obj.get("proposed_actions")
    if not isinstance(raw, list):
        raise GenericLLMProtocolError("'proposed_actions' must be a list")
    actions = []
    for a in raw:
        if not isinstance(a, dict):
            raise GenericLLMProtocolError(f"each action must be an object: {a!r}")
        args = a.get("args")
        actions.append(GenericProposedAction(
            kind=a.get("kind"), symbol=a.get("symbol"),
            args=tuple(args) if isinstance(args, list) else None, value=a.get("value")))
    explanation = obj.get("explanation", "")
    return GenericLLMOutput(actions, explanation if isinstance(explanation, str) else "")


def _action_dict(a: GenericProposedAction) -> dict:
    return {"kind": a.kind, "symbol": a.symbol,
            "args": list(a.args) if a.args is not None else None, "value": a.value}


def validate_generic_llm_actions(
    output: GenericLLMOutput, allowed_semantic_edits: Sequence[SemanticEdit]
) -> GenericValidatedPlan:
    allowed = set(allowed_semantic_edits)
    plan = GenericValidatedPlan(validated=[], rejected=[])
    for a in output.proposed_actions:
        reject = lambda reason: plan.rejected.append({"action": _action_dict(a), "reason": reason})

        if a.kind == "set_relation":
            truth = _coerce_truth(a.value)
            if truth is None:
                reject("invalid_truth_value"); continue
            if not (isinstance(a.symbol, str) and a.args is not None
                    and all(_is_int(x) for x in a.args)):
                reject("malformed_action"); continue
            edit = SetRelation(a.symbol, tuple(a.args), truth)
        elif a.kind == "set_function":
            if not _is_int(a.value):
                reject("invalid_value_for_function"); continue
            if not (isinstance(a.symbol, str) and a.args is not None
                    and all(_is_int(x) for x in a.args)):
                reject("malformed_action"); continue
            edit = SetFunction(a.symbol, tuple(a.args), a.value)
        elif a.kind == "set_constant":
            if not _is_int(a.value):
                reject("invalid_value_for_constant"); continue
            if not isinstance(a.symbol, str):
                reject("malformed_action"); continue
            edit = SetConstant(a.symbol, a.value)
        else:
            reject("unknown_action_kind"); continue

        if edit not in allowed:
            # wrong arity, out-of-domain value, known/seed cell, or simply not offered
            reject("not_in_allowed_actions"); continue
        plan.validated.append(edit)
    return plan
