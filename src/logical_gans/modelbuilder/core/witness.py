"""Replayable witness produced by the Devil for a non-OK clause instance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Witness:
    clause_name: str
    assignment: Dict[str, int]
    premise_values: List[str]
    conclusion_value: Optional[str]
    status: str  # "unknown" | "failed"
    touched_atoms: List[Dict[str, Any]]
    message: str

    def to_json(self) -> dict:
        return {
            "clause_name": self.clause_name,
            "assignment": dict(self.assignment),
            "premise_values": list(self.premise_values),
            "conclusion_value": self.conclusion_value,
            "status": self.status,
            "touched_atoms": list(self.touched_atoms),
            "message": self.message,
        }
