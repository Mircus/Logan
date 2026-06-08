"""Generic synthesize / check / refute over loaded theory packs."""
from __future__ import annotations

from typing import Optional

from .devil import run_devil
from .generator import GenerateResult, generate
from .partial_structure import PartialStructure
from .policy import BuilderPolicy
from .theory import Claim, Theory


def synthesize(
    theory: Theory,
    n: int,
    policy: Optional[BuilderPolicy] = None,
    k: Optional[int] = None,
    budget: Optional[int] = None,
) -> GenerateResult:
    return generate(
        PartialStructure.empty(theory.signature, n),
        theory.clauses,
        policy=policy,
        k=k,
        budget=budget,
    )


def check(theory: Theory, structure: PartialStructure) -> dict:
    res = run_devil(structure, theory.clauses)
    status = "satisfied" if res.status == "ok" else res.status
    return {
        "status": status,
        "structure": structure.to_json(),
        "witness": None if res.witness is None else res.witness.to_json(),
    }


def refute(
    theory: Theory,
    claim: Claim,
    n: Optional[int] = None,
    structure: Optional[PartialStructure] = None,
    policy: Optional[BuilderPolicy] = None,
) -> dict:
    """Obtain a model of `theory`, then attack `claim` on it.

    A model is either supplied (`structure`) or synthesized (`n` + `policy`).
    If the claim has a failing instance on the model, it is refuted.
    """
    if structure is None:
        if n is None:
            raise ValueError("refute needs either a structure or a domain size n")
        gen = synthesize(theory, n, policy)
        if gen.status != "satisfied":
            return {
                "status": f"model_{gen.status}",
                "claim": claim.name,
                "structure": gen.structure.to_json(),
                "witness": None,
            }
        structure = gen.structure

    res = run_devil(structure, claim.clauses)
    if res.status == "failed":
        status = "refuted"
    elif res.status == "ok":
        status = "not_refuted"
    else:
        status = "unknown"
    return {
        "status": status,
        "claim": claim.name,
        "structure": structure.to_json(),
        "witness": None if res.witness is None else res.witness.to_json(),
    }
