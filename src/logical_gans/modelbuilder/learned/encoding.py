"""Encode a partial binary relation (and a Devil witness) as torch tensors."""
from __future__ import annotations

from typing import Optional

import torch

from ..core.partial_structure import PartialStructure
from ..core.types import Truth

_CHANNEL = {Truth.FALSE: 0, Truth.TRUE: 1, Truth.UNKNOWN: 2}


def relation_tensor(structure: PartialStructure, relation: str) -> torch.Tensor:
    """(3, n, n): channel 0=FALSE, 1=TRUE, 2=UNKNOWN."""
    n = len(structure.domain)
    t = torch.zeros(3, n, n, dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            t[_CHANNEL[structure.get_relation(relation, (i, j))], i, j] = 1.0
    return t


def _touched_atoms(witness):
    if witness is None:
        return []
    if hasattr(witness, "touched_atoms"):
        return witness.touched_atoms
    if isinstance(witness, dict):
        return witness.get("touched_atoms", [])
    return []


def witness_mask(witness, relation: str, n: int) -> torch.Tensor:
    """(1, n, n): 1.0 at relation cells touched by the witness, else 0.0."""
    mask = torch.zeros(1, n, n, dtype=torch.float32)
    for atom in _touched_atoms(witness):
        if atom.get("kind") == "rel" and atom.get("relation") == relation:
            vals = atom.get("arg_values")
            if vals and len(vals) == 2 and all(v is not None for v in vals):
                i, j = vals
                if 0 <= i < n and 0 <= j < n:
                    mask[0, i, j] = 1.0
    return mask


def encode_state(structure: PartialStructure, relation: str, witness=None) -> torch.Tensor:
    """(4, n, n): FALSE / TRUE / UNKNOWN / witness_touched."""
    rel = relation_tensor(structure, relation)
    n = rel.shape[-1]
    mask = witness_mask(witness, relation, n)
    return torch.cat([rel, mask], dim=0)
