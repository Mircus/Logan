from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..structures import FiniteMagma
from ..theories.groups import build_cyclic_group, build_dihedral_group, build_klein_four_group

GroupKind = Literal["cyclic", "klein", "dihedral"]


@dataclass
class GroupBuilder:
    """Simple deterministic Builder for first LOGAN group experiments."""

    kind: GroupKind = "cyclic"
    n: int = 4

    def build(self) -> FiniteMagma:
        if self.kind == "cyclic":
            return build_cyclic_group(self.n)
        if self.kind == "klein":
            return build_klein_four_group()
        if self.kind == "dihedral":
            return build_dihedral_group(self.n)
        raise ValueError(f"unknown group kind: {self.kind}")
