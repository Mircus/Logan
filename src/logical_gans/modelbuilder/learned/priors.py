"""Prior providers over semantic edits.

Architectural seam: generic MCTS consumes a ``PriorProvider`` that scores
candidate ``SemanticEdit``s, instead of relation-only logits. Three small
providers ship here; a learned net can be wrapped as a provider later.
"""
from __future__ import annotations

import json
from typing import Dict, List, Sequence

from ..core.obligations import extract_obligation
from .generic_llm_protocol import parse_generic_llm_output, validate_generic_llm_actions
from .semantic_actions import (
    SemanticEdit,
    SetConstant,
    SetFunction,
    SetRelation,
    semantic_edit_to_json,
)


class PriorProvider:
    def score(self, structure, devil_result, allowed_edits: Sequence[SemanticEdit]) -> Dict[SemanticEdit, float]:
        raise NotImplementedError


class UniformPrior(PriorProvider):
    def score(self, structure, devil_result, allowed_edits):
        return {e: 1.0 for e in allowed_edits}


def _matches_cell(edit: SemanticEdit, obl) -> bool:
    if obl.kind == "relation":
        return isinstance(edit, SetRelation) and edit.symbol == obl.symbol and edit.args == obl.args
    if obl.kind == "function":
        return isinstance(edit, SetFunction) and edit.symbol == obl.symbol and edit.args == obl.args
    if obl.kind == "constant":
        return isinstance(edit, SetConstant) and edit.symbol == obl.symbol
    return False


def _edit_value(edit: SemanticEdit):
    return edit.value


class ObligationFirstPrior(PriorProvider):
    """Boost edits that discharge the Devil's current obligation (preferred value highest)."""

    def score(self, structure, devil_result, allowed_edits):
        obl = extract_obligation(structure, devil_result)
        if obl is None:
            return {e: 1.0 for e in allowed_edits}
        preferred = obl.suggested_values[0] if obl.suggested_values else None
        scores = {}
        for e in allowed_edits:
            if _matches_cell(e, obl):
                scores[e] = 2.0 if _edit_value(e) == preferred else 1.0
            else:
                scores[e] = 0.1
        return scores


class MockLLMPrior(PriorProvider):
    """Propose edits via a toy strategy, validate them through the untrusted
    protocol, and boost only the VALIDATED ones."""

    def __init__(self, strategy: str = "first"):
        self.strategy = strategy

    def _choose(self, allowed_edits: Sequence[SemanticEdit]):
        if not allowed_edits:
            return None
        if self.strategy == "first_true":
            for e in allowed_edits:
                if isinstance(e, SetRelation) and e.value.value == "true":
                    return e
        return allowed_edits[0]

    def score(self, structure, devil_result, allowed_edits):
        chosen = self._choose(allowed_edits)
        proposed = [semantic_edit_to_json(chosen)] if chosen is not None else []
        text = json.dumps({"proposed_actions": proposed, "explanation": "mock (ignored)"})
        plan = validate_generic_llm_actions(parse_generic_llm_output(text), allowed_edits)
        valid = set(plan.validated)
        return {e: (2.0 if e in valid else 0.5) for e in allowed_edits}


class NeuralSemanticPrior(PriorProvider):
    """A trained SemanticPolicyNet scoring generic semantic edits by features."""

    is_neural = True

    def __init__(self, model_path: str):
        import torch

        from .semantic_features import FEATURE_DIM
        from .semantic_policy_net import SemanticPolicyNet

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            feature_dim = checkpoint.get("feature_dim", FEATURE_DIM)
            model = SemanticPolicyNet(feature_dim)
            model.load_state_dict(checkpoint["state_dict"])
        except Exception as e:  # malformed/incompatible checkpoint
            raise ValueError(f"invalid SemanticPolicyNet checkpoint {model_path!r}: {e}")
        model.eval()
        self._model = model

    def score(self, structure, devil_result, allowed_edits):
        import torch

        from .semantic_features import feature_vector

        obligation = None
        if devil_result is not None and getattr(devil_result, "status", None) == "unknown":
            obligation = extract_obligation(structure, devil_result)
        feats = torch.tensor(
            [feature_vector(e, structure, obligation) for e in allowed_edits],
            dtype=torch.float32)
        with torch.no_grad():
            logits = self._model(feats)
        return {e: float(v) for e, v in zip(allowed_edits, logits.tolist() if logits.dim() else [float(logits)])}
