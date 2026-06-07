"""The Devil: a deterministic exhaustive bounded checker.

It enumerates all variable assignments over the finite domain in
lexicographic order and returns the first non-OK clause instance as a
witness (either UNKNOWN, i.e. an obligation, or FAILED, a violation).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Optional

try:  # Python 3.8+ has typing.Literal; fall back gracefully
    from typing import Literal

    Status = Literal["ok", "unknown", "failed"]
except ImportError:  # pragma: no cover
    Status = str  # type: ignore

from .clauses import HornClause
from .eval import Assignment, eval_atom, ground_atom
from .partial_structure import PartialStructure
from .types import Truth
from .witness import Witness


@dataclass
class DevilResult:
    status: str  # "ok" | "unknown" | "failed"
    witness: Optional[Witness] = None
    # The offending instance, so the Builder/Generator can act precisely.
    clause: Optional[HornClause] = None
    assignment: Optional[Assignment] = None


def _make_witness(
    status: str,
    clause: HornClause,
    assignment: Assignment,
    premise_truths: List[Truth],
    conclusion: Optional[Truth],
    structure: PartialStructure,
) -> Witness:
    touched = [ground_atom(p, assignment, structure) for p in clause.premises]
    touched.append(ground_atom(clause.conclusion, assignment, structure))
    if conclusion is None:
        msg = (
            f"clause {clause.name!r} has an UNKNOWN premise under {assignment} "
            f"(premises={[t.value for t in premise_truths]})"
        )
    elif status == "failed":
        msg = (
            f"clause {clause.name!r} FAILED under {assignment}: premises all TRUE "
            f"but conclusion is {conclusion.value}"
        )
    else:
        msg = (
            f"clause {clause.name!r} UNKNOWN under {assignment}: premises all TRUE "
            f"but conclusion is UNKNOWN"
        )
    return Witness(
        clause_name=clause.name,
        assignment=dict(assignment),
        premise_values=[t.value for t in premise_truths],
        conclusion_value=None if conclusion is None else conclusion.value,
        status=status,
        touched_atoms=touched,
        message=msg,
    )


def check_clause(clause: HornClause, structure: PartialStructure) -> DevilResult:
    domain = structure.domain
    for values in product(domain, repeat=len(clause.variables)):
        assignment: Assignment = dict(zip(clause.variables, values))
        premise_truths = [eval_atom(p, assignment, structure) for p in clause.premises]

        if any(t is Truth.FALSE for t in premise_truths):
            continue  # premise false -> clause vacuously OK
        if any(t is Truth.UNKNOWN for t in premise_truths):
            w = _make_witness("unknown", clause, assignment, premise_truths, None, structure)
            return DevilResult("unknown", w, clause, assignment)

        conclusion = eval_atom(clause.conclusion, assignment, structure)
        if conclusion is Truth.TRUE:
            continue
        status = "unknown" if conclusion is Truth.UNKNOWN else "failed"
        w = _make_witness(status, clause, assignment, premise_truths, conclusion, structure)
        return DevilResult(status, w, clause, assignment)

    return DevilResult("ok", None)


def run_devil(structure: PartialStructure, clauses: Iterable[HornClause]) -> DevilResult:
    """Return the first non-OK clause instance in deterministic order."""
    for clause in clauses:
        result = check_clause(clause, structure)
        if result.status != "ok":
            return result
    return DevilResult("ok", None)
