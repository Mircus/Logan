"""The Devil: a deterministic bounded checker.

``run_devil`` is the exhaustive checker (unchanged behaviour): it enumerates
all variable assignments over the finite domain in lexicographic order and
returns the first non-OK clause instance.

``run_devil_bounded`` is the bounded logical challenger. It applies a depth
bound ``k`` (clauses with ``clause_depth > k`` are skipped) and a challenge
budget ``b`` (at most ``b`` grounded instances are checked). When the budget
is exhausted without finding a problem, the result is OK with
``budget_exhausted=True`` --- meaning "survived the bounded attack", NOT
"globally verified".
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, List, Optional, Tuple

from .clauses import HornClause
from .depth import clause_depth
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
    # Bounded-run metadata (defaults make exhaustive runs unaffected).
    checked_instances: int = 0
    budget_exhausted: bool = False
    skipped_by_depth: int = 0


def _instance_status(
    clause: HornClause, assignment: Assignment, structure: PartialStructure
) -> Tuple[str, List[Truth], Optional[Truth]]:
    """Status of one grounded clause instance.

    Returns (status, premise_truths, conclusion) where status is
    "ok"/"unknown"/"failed" and conclusion is None when an UNKNOWN premise
    blocked the instance (so no conclusion was reached).
    """
    premise_truths = [eval_atom(p, assignment, structure) for p in clause.premises]
    if any(t is Truth.FALSE for t in premise_truths):
        return ("ok", premise_truths, None)  # vacuously OK
    if any(t is Truth.UNKNOWN for t in premise_truths):
        return ("unknown", premise_truths, None)  # premise-blocked
    conclusion = eval_atom(clause.conclusion, assignment, structure)
    if conclusion is Truth.TRUE:
        return ("ok", premise_truths, conclusion)
    if conclusion is Truth.UNKNOWN:
        return ("unknown", premise_truths, conclusion)
    return ("failed", premise_truths, conclusion)


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
    for values in product(structure.domain, repeat=len(clause.variables)):
        assignment: Assignment = dict(zip(clause.variables, values))
        status, premise_truths, conclusion = _instance_status(clause, assignment, structure)
        if status == "ok":
            continue
        w = _make_witness(status, clause, assignment, premise_truths, conclusion, structure)
        return DevilResult(status, w, clause, assignment)
    return DevilResult("ok", None)


def run_devil(structure: PartialStructure, clauses: Iterable[HornClause]) -> DevilResult:
    """Exhaustive: first non-OK clause instance in deterministic order."""
    for clause in clauses:
        result = check_clause(clause, structure)
        if result.status != "ok":
            return result
    return DevilResult("ok", None)


def run_devil_bounded(
    structure: PartialStructure,
    clauses: Iterable[HornClause],
    *,
    k: Optional[int] = None,
    budget: Optional[int] = None,
) -> DevilResult:
    """Bounded logical challenger with depth bound ``k`` and instance budget ``b``."""
    skipped_by_depth = 0
    active: List[HornClause] = []
    for clause in clauses:
        if k is not None and clause_depth(clause) > k:
            skipped_by_depth += 1
            continue
        active.append(clause)

    checked = 0
    for clause in active:
        for values in product(structure.domain, repeat=len(clause.variables)):
            if budget is not None and checked >= budget:
                return DevilResult(
                    "ok", checked_instances=checked, budget_exhausted=True,
                    skipped_by_depth=skipped_by_depth,
                )
            checked += 1
            assignment: Assignment = dict(zip(clause.variables, values))
            status, premise_truths, conclusion = _instance_status(clause, assignment, structure)
            if status == "ok":
                continue
            w = _make_witness(status, clause, assignment, premise_truths, conclusion, structure)
            return DevilResult(
                status, w, clause, assignment,
                checked_instances=checked, budget_exhausted=False,
                skipped_by_depth=skipped_by_depth,
            )

    return DevilResult(
        "ok", checked_instances=checked, budget_exhausted=False,
        skipped_by_depth=skipped_by_depth,
    )
