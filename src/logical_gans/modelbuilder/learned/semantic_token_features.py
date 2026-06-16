"""Arity-parametric featurizer: a SemanticEdit -> variable-length typed tokens.

Unlike ``semantic_features.feature_vector`` (fixed 13-dim with MAX_ARITY=2 and
``args[:MAX_ARITY]`` truncation), this encoder emits **one token per argument
position** with a positional encoding, so signatures of ANY arity are
represented without truncation. Each token is a fixed-width vector; the number of
tokens varies with the edit's arity.

Token layout (TOKEN_DIM): [0:5] type one-hot (kind/symbol/arg/value/context),
[5:9] four type-specific scalars, [9:15] sinusoidal positional encoding (used by
argument tokens; zeros otherwise).
"""
from __future__ import annotations

import math
from typing import List, Optional

from ..core.types import Truth
from .semantic_actions import SemanticEdit, SetConstant, SetFunction, SetRelation
from .semantic_features import _matches_obligation

_TYPES = ("kind", "symbol", "arg", "value", "context")
POS_DIM = 6
TOKEN_DIM = len(_TYPES) + 4 + POS_DIM  # 5 + 4 + 6 = 15


def _posenc(i: int) -> List[float]:
    """Sinusoidal positional encoding for an UNBOUNDED argument index i."""
    out = []
    for d in range(POS_DIM):
        freq = 1.0 / (10000 ** (2 * (d // 2) / POS_DIM))
        out.append(math.sin(i * freq) if d % 2 == 0 else math.cos(i * freq))
    return out


def _tok(kind: str, scalars: List[float], pos: Optional[List[float]] = None) -> List[float]:
    type_onehot = [1.0 if kind == t else 0.0 for t in _TYPES]
    posv = pos if pos is not None else [0.0] * POS_DIM
    return type_onehot + scalars + posv


def _known_fraction(structure, edit: SemanticEdit) -> float:
    if isinstance(edit, SetRelation):
        cells = [v for (name, _), v in structure.relations.items() if name == edit.symbol]
        known = sum(1 for v in cells if v is not Truth.UNKNOWN)
    elif isinstance(edit, SetFunction):
        cells = [v for (name, _), v in structure.functions.items() if name == edit.symbol]
        known = sum(1 for v in cells if v is not None)
    else:
        return 1.0 if structure.constants.get(edit.symbol) is not None else 0.0
    return known / len(cells) if cells else 0.0


def edit_tokens(edit: SemanticEdit, structure, obligation=None) -> List[List[float]]:
    """Variable-length list of TOKEN_DIM-vectors describing `edit` in context."""
    sig = structure.signature
    n = len(structure.domain)
    denom = (n - 1) if n > 1 else 1

    is_rel = isinstance(edit, SetRelation)
    is_fn = isinstance(edit, SetFunction)
    is_const = isinstance(edit, SetConstant)

    if is_rel:
        arity = sig.relations[edit.symbol].arity
    elif is_fn:
        arity = sig.functions[edit.symbol].arity
    else:
        arity = 0

    tokens: List[List[float]] = []
    # kind
    tokens.append(_tok("kind", [float(is_rel), float(is_fn), float(is_const), 0.0]))
    # symbol: structural (kind, arity, how filled-in this symbol is)
    tokens.append(_tok("symbol", [float(arity), float(is_rel), _known_fraction(structure, edit), 0.0]))
    # arguments: ONE token per position, with positional encoding (no MAX_ARITY)
    args = () if is_const else edit.args
    for i, av in enumerate(args):
        tokens.append(_tok("arg", [av / denom, 0.0, float(av), float(i)], pos=_posenc(i)))
    # value
    if is_rel:
        tokens.append(_tok("value", [1.0 if edit.value is Truth.TRUE else 0.0, 1.0, 0.0, 0.0]))
    else:
        tokens.append(_tok("value", [edit.value / denom, 0.0, float(edit.value), 0.0]))
    # context
    matches = 1.0 if (obligation is not None and _matches_obligation(edit, obligation)) else 0.0
    tokens.append(_tok("context", [matches, float(n), 0.0, 0.0]))
    return tokens
