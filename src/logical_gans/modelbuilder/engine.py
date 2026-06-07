from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from .builders.group_builder import GroupBuilder
from .opponents.horn_group import HornGroupOpponent
from .witnesses import Witness


@dataclass
class RunResult:
    status: str
    structure: str
    order: int
    witnesses: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_group_episode(kind: str, n: int, k: int = 2, budget: int = 100, include_commutativity: bool = False) -> RunResult:
    builder = GroupBuilder(kind=kind, n=n)
    A = builder.build()
    opponent = HornGroupOpponent(k=k, budget=budget, include_commutativity=include_commutativity)
    witnesses = opponent.challenge(A)
    return RunResult(
        status="survived" if not witnesses else "failed",
        structure=A.name,
        order=A.n,
        witnesses=[w.to_dict() for w in witnesses],
    )
