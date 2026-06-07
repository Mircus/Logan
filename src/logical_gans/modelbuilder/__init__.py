"""LOGAN ModelBuilder: adversarial finite model and countermodel synthesis."""

from .structures import FiniteMagma
from .theories.groups import (
    build_cyclic_group,
    build_klein_four_group,
    build_dihedral_group,
    verify_group,
    find_counterexample_all_groups_abelian,
)

__all__ = [
    "FiniteMagma",
    "build_cyclic_group",
    "build_klein_four_group",
    "build_dihedral_group",
    "verify_group",
    "find_counterexample_all_groups_abelian",
]
