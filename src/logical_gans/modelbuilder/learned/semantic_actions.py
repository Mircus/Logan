"""Signature-parametric semantic edits.

Generic actions over ANY first-order signature -- not pinned to one binary
relation. An edit fixes one currently-UNKNOWN interpretation cell:

    SetRelation(symbol, args, TRUE/FALSE)
    SetFunction(symbol, args, value in domain)
    SetConstant(symbol, value in domain)

Known cells are never revisable. stdlib-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

from ..core.partial_structure import PartialStructure
from ..core.types import Truth


@dataclass(frozen=True)
class SetRelation:
    symbol: str
    args: Tuple[int, ...]
    value: Truth


@dataclass(frozen=True)
class SetFunction:
    symbol: str
    args: Tuple[int, ...]
    value: int


@dataclass(frozen=True)
class SetConstant:
    symbol: str
    value: int


SemanticEdit = Union[SetRelation, SetFunction, SetConstant]


def legal_semantic_edits(structure: PartialStructure) -> List[SemanticEdit]:
    """All edits to currently-UNKNOWN cells over the whole signature."""
    edits: List[SemanticEdit] = []
    for (name, args), value in structure.relations.items():
        if value is Truth.UNKNOWN:
            edits.append(SetRelation(name, args, Truth.FALSE))
            edits.append(SetRelation(name, args, Truth.TRUE))
    for (name, args), value in structure.functions.items():
        if value is None:
            for d in structure.domain:
                edits.append(SetFunction(name, args, d))
    for name, value in structure.constants.items():
        if value is None:
            for d in structure.domain:
                edits.append(SetConstant(name, d))
    return edits


def apply_semantic_edit(structure: PartialStructure, edit: SemanticEdit) -> PartialStructure:
    """Return a COPY with `edit` applied; the original is untouched."""
    s = structure.copy()
    if isinstance(edit, SetRelation):
        s.set_relation(edit.symbol, edit.args, edit.value)
    elif isinstance(edit, SetFunction):
        s.set_function(edit.symbol, edit.args, edit.value)
    elif isinstance(edit, SetConstant):
        s.set_constant(edit.symbol, edit.value)
    else:
        raise TypeError(f"unknown semantic edit {edit!r}")
    return s


def semantic_edit_to_json(edit: SemanticEdit) -> dict:
    if isinstance(edit, SetRelation):
        return {"kind": "set_relation", "symbol": edit.symbol,
                "args": list(edit.args), "value": edit.value.value}
    if isinstance(edit, SetFunction):
        return {"kind": "set_function", "symbol": edit.symbol,
                "args": list(edit.args), "value": edit.value}
    if isinstance(edit, SetConstant):
        return {"kind": "set_constant", "symbol": edit.symbol, "value": edit.value}
    raise TypeError(f"unknown semantic edit {edit!r}")


def semantic_edit_from_json(obj: dict) -> SemanticEdit:
    kind = obj.get("kind")
    if kind == "set_relation":
        value = obj["value"]
        truth = Truth.TRUE if value in (True, "true") else (
            Truth.FALSE if value in (False, "false") else None)
        if truth is None:
            raise ValueError(f"bad truth value {value!r}")
        return SetRelation(obj["symbol"], tuple(obj["args"]), truth)
    if kind == "set_function":
        return SetFunction(obj["symbol"], tuple(obj["args"]), int(obj["value"]))
    if kind == "set_constant":
        return SetConstant(obj["symbol"], int(obj["value"]))
    raise ValueError(f"unknown action kind {kind!r}")
