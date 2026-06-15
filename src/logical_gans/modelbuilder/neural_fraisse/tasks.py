"""Generated task family for the Neural Fraïssé benchmark.

Signature Σ = {E/2, s/1, a}. Theories T = {∀x E(x,s(x)); ∀x s^m(x)=x}.
Two kinds:
  model        : satisfy T            (mode "satisfy", empty goal)
  countermodel : satisfy T, refute C  (mode "refute", goal = claim)

Tasks are generated over domain sizes and filtered to FEASIBLE ones (a solution
provably exists by brute force) so the benchmark measures search/learning, not
luck of solvability. No single example is hand-picked.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import List, Optional

from ..core.atoms import EqAtom, RelAtom
from ..core.clauses import HornClause
from ..core.signature import Signature
from ..core.terms import Const, Func, Var
from ..core.theory import Claim, Theory

SIG = Signature.build(relations=[("E", 2)], functions=[("s", 1)], constants=["a"])


def _s_pow(term, m: int):
    t = term
    for _ in range(m):
        t = Func("s", (t,))
    return t


def _theory(m: int) -> Theory:
    x = Var("x")
    clauses = [
        HornClause("edge_to_succ", ("x",), (), RelAtom("E", (x, Func("s", (x,))))),
        HornClause(f"order_{m}", ("x",), (), EqAtom(_s_pow(x, m), x)),
    ]
    return Theory(name=f"cyc_m{m}", signature=SIG, clauses=clauses)


def _claim(kind: str, r: int) -> Claim:
    a = Const("a")
    x = Var("x")
    if kind == "fix_a":            # s(a) = a
        return Claim("fix_a", [HornClause("fix_a", (), (), EqAtom(Func("s", (a,)), a))])
    if kind == "loop_a":           # E(a,a)
        return Claim("loop_a", [HornClause("loop_a", (), (), RelAtom("E", (a, a)))])
    if kind == "reflexive":        # forall x E(x,x)
        return Claim("reflexive", [HornClause("reflexive", ("x",), (), RelAtom("E", (x, x)))])
    if kind == "spow_a":           # s^r(a) = a
        return Claim(f"spow{r}_a", [HornClause(f"spow{r}_a", (), (), EqAtom(_s_pow(a, r), a))])
    raise ValueError(kind)


@dataclass(frozen=True)
class Task:
    name: str
    signature: Signature
    theory: Theory
    goal: Claim                 # empty Claim for model tasks
    mode: str                   # "satisfy" | "refute"
    kind: str                   # "model" | "countermodel"
    domain_size: int
    depth: int                  # k
    m: int


# --------------------------------------------------------------------------
# Feasibility oracle (brute force over total structures).  Used to filter tasks
# and (in training.py) to find a winning trajectory.
# --------------------------------------------------------------------------
def _permutations_order_div(n: int, m: int):
    """All maps s: [n]->[n] that are permutations with s^m = identity."""
    for s in itertools.permutations(range(n)):
        ok = True
        for x in range(n):
            y = x
            for _ in range(m):
                y = s[y]
            if y != x:
                ok = False
                break
        if ok:
            yield s


def has_solution(task: Task) -> bool:
    return solution_structures(task, limit=1) != []


def solution_structures(task: Task, limit: int = 1):
    """Return up to `limit` full solutions as (s_tuple, a_value or None).

    A solution sets s to a valid permutation, E(x,s(x))=true (rest false), and for
    countermodels picks a constant a making the claim fail while T holds.
    """
    from ..core.devil import run_devil
    from ..core.partial_structure import PartialStructure
    from ..core.types import Truth

    n, m = task.domain_size, task.m
    out = []
    for s in _permutations_order_div(n, m):
        a_choices = range(n) if task.kind == "countermodel" else [None]
        for a_val in a_choices:
            A = PartialStructure.empty(task.signature, n)
            for x in range(n):
                A.set_function("s", (x,), s[x])
            for x in range(n):
                for y in range(n):
                    A.set_relation("E", (x, y), Truth.TRUE if y == s[x] else Truth.FALSE)
            if a_val is not None:
                A.set_constant("a", a_val)
            if run_devil(A, task.theory.clauses).status != "ok":
                continue
            if task.kind == "countermodel":
                if run_devil(A, task.goal.clauses).status != "failed":
                    continue
            out.append((s, a_val))
            if len(out) >= limit:
                return out
    return out


_CLAIMS = ["fix_a", "loop_a", "reflexive", "spow_a"]


def generate_tasks(sizes, count: int, seed: int = 0,
                   kinds=("model", "countermodel")) -> List[Task]:
    """Generate `count` feasible tasks spread over sizes / m / claim / kind."""
    rng = random.Random(seed)
    tasks: List[Task] = []
    attempts = 0
    seen = set()
    while len(tasks) < count and attempts < count * 60 + 200:
        attempts += 1
        n = rng.choice(list(sizes))
        m = rng.choice([2, 3, 4])
        kind = rng.choice(list(kinds))
        if kind == "model":
            goal, mode = _claim("fix_a", 0), "satisfy"
            goal = Claim("true", [])          # empty goal: satisfy T only
            r = 0
            cname = f"model_n{n}_m{m}"
        else:
            ckind = rng.choice(_CLAIMS)
            r = rng.randint(1, m - 1) if ckind == "spow_a" and m > 1 else 1
            goal, mode = _claim(ckind, r), "refute"
            cname = f"cm_{ckind}{r if ckind=='spow_a' else ''}_n{n}_m{m}"
        key = cname
        if key in seen:
            continue
        task = Task(name=cname, signature=SIG, theory=_theory(m), goal=goal,
                    mode=mode, kind=kind, domain_size=n, depth=max(m, 1), m=m)
        if not has_solution(task):
            continue
        seen.add(key)
        tasks.append(task)
    return tasks
