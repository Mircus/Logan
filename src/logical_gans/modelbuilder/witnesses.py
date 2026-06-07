from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class Witness:
    """Small, checkable LOGAN witness.

    A witness is not a philosophical explanation. It is data small enough that
    another checker can replay it deterministically.
    """

    kind: str
    payload: Dict[str, Any]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AxiomWitness(Witness):
    """Witness that an axiom instance fails."""


@dataclass(frozen=True)
class CounterexampleCertificate:
    """A model of assumptions plus a witness refuting a target claim."""

    claim: str
    structure_name: str
    structure_order: int
    assumptions_verified: bool
    refuting_witness: Witness

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "structure_name": self.structure_name,
            "structure_order": self.structure_order,
            "assumptions_verified": self.assumptions_verified,
            "refuting_witness": self.refuting_witness.to_dict(),
        }
