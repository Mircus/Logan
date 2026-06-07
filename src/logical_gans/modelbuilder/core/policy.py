"""Builder policies: how the Generator *discharges* UNKNOWN cells.

This is a construction strategy, NOT logical semantics. The Devil's
evaluation stays strictly three-valued (TRUE / FALSE / UNKNOWN). When the
Generator faces an UNKNOWN premise relation cell, the policy decides what
truth value to commit to (and which domain element to use when filling an
unknown function/constant cell).

P0 ships two concrete monotone policies:

* ``SparseHornPolicy`` (default): UNKNOWN premise -> FALSE. Discharges Horn
  clauses vacuously and converges to the sparse/identity model (e.g. the
  identity preorder). Conceptually the "vacuous minimal" policy.
* ``MaximalHornPolicy``: UNKNOWN premise -> TRUE. Converges to the dense/
  maximal model (e.g. the total preorder).

Future, non-monotone policies (need search/backtracking, not yet built):
``BranchingHornPolicy`` (try both polarities) and ``BacktrackingPolicy``
(choose values that preserve future satisfiability).
"""
from __future__ import annotations

from typing import Dict

from .partial_structure import PartialStructure
from .types import Truth


class BuilderPolicy:
    """Strategy object the Generator consults to discharge UNKNOWNs."""

    name = "builder-policy"

    def unknown_premise_value(
        self,
        structure: PartialStructure,
        relation: str,
        args: tuple,
        assignment: Dict[str, int],
    ) -> Truth:
        """Truth value to commit for an UNKNOWN premise relation cell."""
        raise NotImplementedError

    def fill_value(self, structure: PartialStructure) -> int:
        """Domain element to use when filling an UNKNOWN function/constant cell."""
        return structure.domain[0]  # smallest, monotone default


class SparseHornPolicy(BuilderPolicy):
    """UNKNOWN premise -> FALSE (vacuous discharge; sparse/identity model)."""

    name = "sparse_horn"

    def unknown_premise_value(self, structure, relation, args, assignment) -> Truth:
        return Truth.FALSE


class MaximalHornPolicy(BuilderPolicy):
    """UNKNOWN premise -> TRUE (dense/maximal model)."""

    name = "maximal_horn"

    def unknown_premise_value(self, structure, relation, args, assignment) -> Truth:
        return Truth.TRUE


# Registry for CLI / config selection.
POLICIES: Dict[str, type] = {
    SparseHornPolicy.name: SparseHornPolicy,
    MaximalHornPolicy.name: MaximalHornPolicy,
}

DEFAULT_POLICY = SparseHornPolicy


def get_policy(name: str) -> BuilderPolicy:
    try:
        return POLICIES[name]()
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; choices: {sorted(POLICIES)}")
