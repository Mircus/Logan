"""Known finite-group Cayley tables, kept ONLY as regression fixtures.

These are NOT the model builder. The real engine fills unknown
interpretation tables (see ``modelbuilder.core``). These hand-written
tables are convenient sanity fixtures for tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class CayleyTable:
    table: Sequence[Sequence[int]]
    name: str
    identity: int = 0

    @property
    def n(self) -> int:
        return len(self.table)

    def mul(self, a: int, b: int) -> int:
        return self.table[a][b]


def cyclic_group(n: int) -> CayleyTable:
    return CayleyTable([[(a + b) % n for b in range(n)] for a in range(n)], f"C_{n}")


def klein_four_group() -> CayleyTable:
    elems = [(0, 0), (1, 0), (0, 1), (1, 1)]
    index = {x: i for i, x in enumerate(elems)}
    table = [
        [index[((a[0] + b[0]) % 2, (a[1] + b[1]) % 2)] for b in elems] for a in elems
    ]
    return CayleyTable(table, "V_4")


def dihedral_group(m: int) -> CayleyTable:
    if m < 3:
        raise ValueError("m must be >= 3")
    elems = [(i, j) for j in (0, 1) for i in range(m)]
    index = {x: p for p, x in enumerate(elems)}
    table = [
        [index[(((i + ((-1) ** j) * k) % m), (j + l) % 2)] for (k, l) in elems]
        for (i, j) in elems
    ]
    return CayleyTable(table, f"D_{m}", identity=index[(0, 0)])


def is_group(g: CayleyTable) -> bool:
    """Minimal self-contained group check (assoc + identity + inverses)."""
    rng = range(g.n)
    for x in rng:
        for y in rng:
            for z in rng:
                if g.mul(g.mul(x, y), z) != g.mul(x, g.mul(y, z)):
                    return False
    e = g.identity
    if any(g.mul(e, x) != x or g.mul(x, e) != x for x in rng):
        return False
    for x in rng:
        if not any(g.mul(x, y) == e and g.mul(y, x) == e for y in rng):
            return False
    return True


def is_abelian(g: CayleyTable) -> bool:
    return all(g.mul(x, y) == g.mul(y, x) for x in range(g.n) for y in range(g.n))
