from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class FiniteMagma:
    """Finite binary algebra represented by a total Cayley table.

    Elements are integers 0..n-1.  For group examples, element 0 is usually
    the identity.  The representation is intentionally simple so that LOGAN
    witnesses are replayable without hidden libraries.
    """

    table: Sequence[Sequence[int]]
    name: str = "anonymous"
    identity: Optional[int] = 0

    def __post_init__(self) -> None:
        n = len(self.table)
        if n == 0:
            raise ValueError("table must be nonempty")
        for row in self.table:
            if len(row) != n:
                raise ValueError("Cayley table must be square")
            for value in row:
                if not isinstance(value, int) or value < 0 or value >= n:
                    raise ValueError(f"table entry {value!r} is not an element 0..{n-1}")

    @property
    def n(self) -> int:
        return len(self.table)

    @property
    def elements(self) -> range:
        return range(self.n)

    def mul(self, a: int, b: int) -> int:
        self._check_element(a)
        self._check_element(b)
        return self.table[a][b]

    def _check_element(self, x: int) -> None:
        if not isinstance(x, int) or x < 0 or x >= self.n:
            raise ValueError(f"{x!r} is not an element of {self.name}")

    def row(self, a: int) -> List[int]:
        self._check_element(a)
        return list(self.table[a])

    def as_nested_lists(self) -> List[List[int]]:
        return [list(row) for row in self.table]

    def format_table(self) -> str:
        rows = [" ".join(f"{x:2d}" for x in row) for row in self.table]
        return "\n".join(rows)
