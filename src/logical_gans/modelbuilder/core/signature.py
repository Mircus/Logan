"""First-order signatures: relation, function, and constant symbols."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple


@dataclass(frozen=True)
class RelationSymbol:
    name: str
    arity: int


@dataclass(frozen=True)
class FunctionSymbol:
    name: str
    arity: int


@dataclass(frozen=True)
class ConstantSymbol:
    name: str


@dataclass
class Signature:
    relations: dict[str, RelationSymbol] = field(default_factory=dict)
    functions: dict[str, FunctionSymbol] = field(default_factory=dict)
    constants: dict[str, ConstantSymbol] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        relations: Iterable[Tuple[str, int]] = (),
        functions: Iterable[Tuple[str, int]] = (),
        constants: Iterable[str] = (),
    ) -> "Signature":
        """Convenience constructor from (name, arity) pairs and constant names."""
        return cls(
            relations={n: RelationSymbol(n, a) for n, a in relations},
            functions={n: FunctionSymbol(n, a) for n, a in functions},
            constants={n: ConstantSymbol(n) for n in constants},
        )
