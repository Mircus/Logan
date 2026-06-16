"""Relation edits and the (n*n*2) action index space for one binary relation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ..core.partial_structure import PartialStructure
from ..core.types import Truth


@dataclass(frozen=True)
class RelationEdit:
    relation: str
    args: Tuple[int, int]
    value: Truth


def legal_relation_edits(structure: PartialStructure, relation: str) -> List[RelationEdit]:
    """Both truth values for each UNKNOWN cell. Known cells are not revisable."""
    n = len(structure.domain)
    edits: List[RelationEdit] = []
    for i in range(n):
        for j in range(n):
            if structure.get_relation(relation, (i, j)) is Truth.UNKNOWN:
                edits.append(RelationEdit(relation, (i, j), Truth.FALSE))
                edits.append(RelationEdit(relation, (i, j), Truth.TRUE))
    return edits


def apply_relation_edit(structure: PartialStructure, edit: RelationEdit) -> PartialStructure:
    """Return a COPY with the edit applied (original untouched)."""
    s = structure.copy()
    s.set_relation(edit.relation, edit.args, edit.value)
    return s


def encode_action_index(edit: RelationEdit, n: int) -> int:
    i, j = edit.args
    tbit = 1 if edit.value is Truth.TRUE else 0
    return (i * n + j) * 2 + tbit


def decode_action_index(index: int, n: int, relation: str) -> RelationEdit:
    cell, tbit = divmod(index, 2)
    i, j = divmod(cell, n)
    value = Truth.TRUE if tbit == 1 else Truth.FALSE
    return RelationEdit(relation, (i, j), value)


def num_actions(n: int) -> int:
    return n * n * 2
