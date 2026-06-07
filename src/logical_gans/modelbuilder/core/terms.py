"""Term AST: variables, constants, and function applications."""
from __future__ import annotations

from dataclasses import dataclass


class Term:
    pass


@dataclass(frozen=True)
class Var(Term):
    name: str


@dataclass(frozen=True)
class Const(Term):
    name: str


@dataclass(frozen=True)
class Func(Term):
    name: str
    args: tuple[Term, ...]


def term_str(term: Term) -> str:
    if isinstance(term, Var):
        return term.name
    if isinstance(term, Const):
        return term.name
    if isinstance(term, Func):
        return f"{term.name}({','.join(term_str(a) for a in term.args)})"
    raise TypeError(f"unknown term {term!r}")
