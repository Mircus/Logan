"""A finite partial Sigma-structure with UNKNOWN interpretations."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import List, Optional, Tuple

from .signature import Signature
from .types import Truth

RelCell = Tuple[str, Tuple[int, ...]]
FuncCell = Tuple[str, Tuple[int, ...]]


@dataclass
class PartialStructure:
    signature: Signature
    domain: tuple[int, ...]
    constants: dict[str, Optional[int]] = field(default_factory=dict)
    relations: dict[RelCell, Truth] = field(default_factory=dict)
    functions: dict[FuncCell, Optional[int]] = field(default_factory=dict)

    @classmethod
    def empty(cls, signature: Signature, n: int) -> "PartialStructure":
        """Every relation tuple starts UNKNOWN; every function/constant starts None."""
        domain = tuple(range(n))
        constants: dict[str, Optional[int]] = {name: None for name in signature.constants}
        relations: dict[RelCell, Truth] = {}
        for name, rel in signature.relations.items():
            for args in product(domain, repeat=rel.arity):
                relations[(name, args)] = Truth.UNKNOWN
        functions: dict[FuncCell, Optional[int]] = {}
        for name, fn in signature.functions.items():
            for args in product(domain, repeat=fn.arity):
                functions[(name, args)] = None
        return cls(signature, domain, constants, relations, functions)

    # -- relations -------------------------------------------------------
    def get_relation(self, name: str, args: Tuple[int, ...]) -> Truth:
        return self.relations[(name, tuple(args))]

    def set_relation(self, name: str, args: Tuple[int, ...], value: Truth) -> None:
        self.relations[(name, tuple(args))] = value

    # -- functions -------------------------------------------------------
    def get_function(self, name: str, args: Tuple[int, ...]) -> Optional[int]:
        return self.functions[(name, tuple(args))]

    def set_function(self, name: str, args: Tuple[int, ...], value: int) -> None:
        self.functions[(name, tuple(args))] = value

    # -- constants -------------------------------------------------------
    def get_constant(self, name: str) -> Optional[int]:
        return self.constants[name]

    def set_constant(self, name: str, value: int) -> None:
        self.constants[name] = value

    # -- introspection ---------------------------------------------------
    def unknown_relation_cells(self) -> List[RelCell]:
        return [cell for cell, v in self.relations.items() if v is Truth.UNKNOWN]

    def unknown_function_cells(self) -> List[FuncCell]:
        return [cell for cell, v in self.functions.items() if v is None]

    def copy(self) -> "PartialStructure":
        return PartialStructure(
            self.signature,
            self.domain,
            dict(self.constants),
            dict(self.relations),
            dict(self.functions),
        )

    def to_json(self) -> dict:
        def cell_key(name: str, args: Tuple[int, ...]) -> str:
            return f"{name}({','.join(map(str, args))})"

        return {
            "domain": list(self.domain),
            "constants": dict(self.constants),
            "relations": {
                cell_key(name, args): self.relations[(name, args)].value
                for (name, args) in sorted(self.relations)
            },
            "functions": {
                cell_key(name, args): self.functions[(name, args)]
                for (name, args) in sorted(self.functions)
            },
        }
