"""JSON loaders for the P0 fragment: signatures, theories, claims, structures.

JSON shapes
-----------
Term:    {"var": "x"} | {"const": "e"} | {"func": "mul", "args": [<term>, ...]}
Atom:    {"rel": "R", "args": [<term>, ...]} | {"eq": [<term>, <term>]}
Clause:  {"name": str, "variables": [str, ...],
          "premises": [<atom>, ...], "conclusion": <atom>}
Theory:  {"name": str, "signature": {"relations": [{"name","arity"}, ...],
          "functions": [...], "constants": ["e", ...]}, "clauses": [<clause>, ...]}
Claim:   {"name": str, "clauses": [<clause>, ...]}
Struct:  {"domain": [0,...,n-1], "constants": {name: int},
          "relations": {R: [[a,b], ...]},   # listed tuples are TRUE; rest FALSE
          "functions": {f: [[[a,b], value], ...]}}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from .atoms import EqAtom, RelAtom
from .clauses import HornClause
from .partial_structure import PartialStructure
from .signature import Signature
from .terms import Const, Func, Term, Var
from .theory import Claim, Theory
from .types import Truth

PathLike = Union[str, Path]


class TheoryLoadError(ValueError):
    """Raised when a JSON theory/claim/structure pack cannot be parsed."""


# --- term / atom / clause -------------------------------------------------

def parse_term(obj: Any) -> Term:
    if not isinstance(obj, dict):
        raise TheoryLoadError(f"term must be an object, got {obj!r}")
    if "var" in obj:
        return Var(obj["var"])
    if "const" in obj:
        return Const(obj["const"])
    if "func" in obj:
        args = obj.get("args", [])
        if not isinstance(args, list):
            raise TheoryLoadError(f"func args must be a list in {obj!r}")
        return Func(obj["func"], tuple(parse_term(a) for a in args))
    raise TheoryLoadError(f"term must have one of 'var'/'const'/'func': {obj!r}")


def parse_atom(obj: Any):
    if not isinstance(obj, dict):
        raise TheoryLoadError(f"atom must be an object, got {obj!r}")
    if "rel" in obj:
        args = obj.get("args", [])
        if not isinstance(args, list):
            raise TheoryLoadError(f"rel args must be a list in {obj!r}")
        return RelAtom(obj["rel"], tuple(parse_term(a) for a in args))
    if "eq" in obj:
        pair = obj["eq"]
        if not isinstance(pair, list) or len(pair) != 2:
            raise TheoryLoadError(f"'eq' must be a 2-element list in {obj!r}")
        return EqAtom(parse_term(pair[0]), parse_term(pair[1]))
    raise TheoryLoadError(f"atom must have 'rel' or 'eq': {obj!r}")


def parse_clause(obj: Any) -> HornClause:
    if not isinstance(obj, dict):
        raise TheoryLoadError(f"clause must be an object, got {obj!r}")
    try:
        name = obj["name"]
        variables = tuple(obj["variables"])
        premises = tuple(parse_atom(a) for a in obj.get("premises", []))
        conclusion = parse_atom(obj["conclusion"])
    except KeyError as e:
        raise TheoryLoadError(f"clause missing field {e} in {obj!r}")
    return HornClause(name=name, variables=variables, premises=premises, conclusion=conclusion)


def parse_signature(obj: Any) -> Signature:
    if not isinstance(obj, dict):
        raise TheoryLoadError(f"signature must be an object, got {obj!r}")
    try:
        relations = [(r["name"], r["arity"]) for r in obj.get("relations", [])]
        functions = [(f["name"], f["arity"]) for f in obj.get("functions", [])]
        constants = [c if isinstance(c, str) else c["name"] for c in obj.get("constants", [])]
    except (KeyError, TypeError) as e:
        raise TheoryLoadError(f"malformed signature {obj!r}: {e}")
    return Signature.build(relations=relations, functions=functions, constants=constants)


# --- file loading ---------------------------------------------------------

def _read_json(path: PathLike) -> Any:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise TheoryLoadError(f"cannot read {p}: {e}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise TheoryLoadError(f"invalid JSON in {p}: {e}")


def load_theory(path: PathLike) -> Theory:
    obj = _read_json(path)
    if not isinstance(obj, dict) or "signature" not in obj or "clauses" not in obj:
        raise TheoryLoadError(f"theory {path} needs 'signature' and 'clauses'")
    signature = parse_signature(obj["signature"])
    clauses = [parse_clause(c) for c in obj["clauses"]]
    return Theory(name=obj.get("name", "theory"), signature=signature, clauses=clauses)


def load_claim(path: PathLike) -> Claim:
    obj = _read_json(path)
    if not isinstance(obj, dict) or "clauses" not in obj:
        raise TheoryLoadError(f"claim {path} needs 'clauses'")
    clauses = [parse_clause(c) for c in obj["clauses"]]
    return Claim(name=obj.get("name", "claim"), clauses=clauses)


def load_structure(path: PathLike, signature: Signature) -> PartialStructure:
    obj = _read_json(path)
    if not isinstance(obj, dict) or "domain" not in obj:
        raise TheoryLoadError(f"structure {path} needs 'domain'")
    domain = obj["domain"]
    n = len(domain)
    if list(domain) != list(range(n)):
        raise TheoryLoadError(f"domain must be 0..n-1, got {domain}")

    structure = PartialStructure.empty(signature, n)

    for cname, value in obj.get("constants", {}).items():
        if cname not in signature.constants:
            raise TheoryLoadError(f"constant {cname!r} not in signature")
        if value is not None:
            structure.set_constant(cname, int(value))

    for rname, tuples in obj.get("relations", {}).items():
        if rname not in signature.relations:
            raise TheoryLoadError(f"relation {rname!r} not in signature")
        for t in tuples:
            structure.set_relation(rname, tuple(t), Truth.TRUE)

    # Closed-world: any relation cell still UNKNOWN becomes FALSE.
    for cell in structure.unknown_relation_cells():
        structure.relations[cell] = Truth.FALSE

    for fname, entries in obj.get("functions", {}).items():
        if fname not in signature.functions:
            raise TheoryLoadError(f"function {fname!r} not in signature")
        for args, value in entries:
            structure.set_function(fname, tuple(args), int(value))

    return structure
