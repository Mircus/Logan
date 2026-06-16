"""Numeric features for candidate SemanticEdit actions (signature-parametric).

Fixed-length feature vector per candidate action, so a single neural policy can
score relation / function / constant edits over any signature.
"""
from __future__ import annotations

from typing import List, Optional

from ..core.partial_structure import PartialStructure
from ..core.types import Truth
from .semantic_actions import SemanticEdit, SetConstant, SetFunction, SetRelation

MAX_ARITY = 2
FEATURE_DIM = 13  # see feature_vector layout below


def symbol_ids(signature):
    names = list(signature.relations) + list(signature.functions) + list(signature.constants)
    return {name: i for i, name in enumerate(names)}, len(names)


def _matches_obligation(edit: SemanticEdit, obligation) -> bool:
    if obligation is None:
        return False
    if obligation.kind == "relation":
        return isinstance(edit, SetRelation) and edit.symbol == obligation.symbol and edit.args == obligation.args
    if obligation.kind == "function":
        return isinstance(edit, SetFunction) and edit.symbol == obligation.symbol and edit.args == obligation.args
    if obligation.kind == "constant":
        return isinstance(edit, SetConstant) and edit.symbol == obligation.symbol
    return False


def feature_vector(edit: SemanticEdit, structure: PartialStructure, obligation=None) -> List[float]:
    """Layout (len = FEATURE_DIM = 13):

        [0:3]  is_relation, is_function, is_constant
        [3]    symbol id (normalized)
        [4]    arity
        [5:7]  args (normalized, padded with -1 to MAX_ARITY)
        [7]    value (truth 1/0 for relation; domain value normalized for f/c)
        [8]    matches_current_obligation (0/1)
        [9:13] source-kind one-hot: relation / function / constant / none
    """
    ids, num = symbol_ids(structure.signature)
    n = len(structure.domain)
    denom = (n - 1) if n > 1 else 1

    is_rel = isinstance(edit, SetRelation)
    is_fn = isinstance(edit, SetFunction)
    is_const = isinstance(edit, SetConstant)

    sym_id = (ids.get(edit.symbol, -1) / (num - 1)) if num > 1 else 0.0
    args = () if is_const else edit.args
    arity = float(len(args))
    args_norm = [a / denom for a in args][:MAX_ARITY]
    while len(args_norm) < MAX_ARITY:
        args_norm.append(-1.0)

    if is_rel:
        value = 1.0 if edit.value is Truth.TRUE else 0.0
    else:
        value = edit.value / denom

    matches = 1.0 if _matches_obligation(edit, obligation) else 0.0
    src_kind = obligation.kind if obligation is not None else "none"
    one_hot = [1.0 if src_kind == kind else 0.0 for kind in ("relation", "function", "constant", "none")]

    return [float(is_rel), float(is_fn), float(is_const), sym_id, arity,
            *args_norm, value, matches, *one_hot]
