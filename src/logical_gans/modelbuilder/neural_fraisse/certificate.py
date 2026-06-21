"""Bounded obstruction certificate for DEVIL_WINS (Gate R2).

The symbolic Judge (`core.devil.run_devil`) independently re-verifies that NO total
completion of the signature over the finite domain satisfies the theory while also
achieving the goal. When the win-relevant completion space is small enough to
enumerate exhaustively and contains zero winning completions, the certificate is
`judge_verified` --- a genuine bounded obstruction, not a search timeout.

This is intentionally independent of the game's DFS: the game's `"lost"` status
only signals that the obligation-ordered search closed without a win; this module
re-confirms it by brute force under the Judge.
"""
from __future__ import annotations

from itertools import product
from typing import Optional

from ..core.devil import run_devil
from ..core.partial_structure import PartialStructure
from ..core.types import Truth


def build_obstruction_certificate(task, hint: Optional[dict] = None,
                                  max_completions: int = 200000) -> dict:
    sig = task.signature
    n = task.domain_size
    dom = list(range(n))

    rel_cells = [(name, args) for name, s in sig.relations.items()
                 for args in product(dom, repeat=s.arity)]
    fn_cells = [(name, args) for name, s in sig.functions.items()
                for args in product(dom, repeat=s.arity)]
    consts = list(sig.constants)

    total = (2 ** len(rel_cells)) * (n ** len(fn_cells)) * (n ** len(consts))
    cert = {
        "domain_size": n,
        "method": "exhaustive bounded completion check (symbolic Judge = run_devil)",
        "completion_space": total,
        "budget_exhausted": False,
    }
    if total > max_completions:
        cert.update(judge_verified=False, completions_checked=0, winning_completions=None,
                    note=f"completion space {total} exceeds max {max_completions}; not certified")
        return cert

    checked = 0
    winning = 0
    for rel_vals in product([Truth.FALSE, Truth.TRUE], repeat=len(rel_cells)):
        for fn_vals in product(dom, repeat=len(fn_cells)):
            for c_vals in product(dom, repeat=len(consts)):
                A = PartialStructure.empty(sig, n)
                for (name, args), v in zip(rel_cells, rel_vals):
                    A.set_relation(name, args, v)
                for (name, args), v in zip(fn_cells, fn_vals):
                    A.set_function(name, args, v)
                for name, v in zip(consts, c_vals):
                    A.set_constant(name, v)
                checked += 1
                if run_devil(A, task.theory.clauses).status != "ok":
                    continue  # not a model of the theory
                goal_status = run_devil(A, task.goal.clauses).status
                if task.mode == "refute":
                    if goal_status == "failed":      # theory ok AND claim refuted -> a Builder win
                        winning += 1
                else:                                 # satisfy
                    if goal_status == "ok":
                        winning += 1

    cert.update(completions_checked=checked, winning_completions=winning,
                judge_verified=(winning == 0))
    if hint:
        if hint.get("reason"):
            cert["reason"] = hint["reason"]
        if hint.get("consequence"):
            cert["consequence"] = hint["consequence"]
    return cert
