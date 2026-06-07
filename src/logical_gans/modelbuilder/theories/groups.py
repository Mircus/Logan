from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from ..structures import FiniteMagma
from ..witnesses import CounterexampleCertificate, Witness


def build_cyclic_group(n: int) -> FiniteMagma:
    """Return the cyclic group C_n under addition modulo n."""
    if n <= 0:
        raise ValueError("n must be positive")
    table = [[(a + b) % n for b in range(n)] for a in range(n)]
    return FiniteMagma(table=table, name=f"C_{n}", identity=0)


def build_klein_four_group() -> FiniteMagma:
    """Return V_4 represented as Z_2 x Z_2."""
    elems = [(0, 0), (1, 0), (0, 1), (1, 1)]
    index = {x: i for i, x in enumerate(elems)}
    table = []
    for a in elems:
        row = []
        for b in elems:
            row.append(index[((a[0] + b[0]) % 2, (a[1] + b[1]) % 2)])
        table.append(row)
    return FiniteMagma(table=table, name="V_4", identity=0)


def build_dihedral_group(m: int) -> FiniteMagma:
    """Return D_m, the symmetries of an m-gon, of order 2m.

    Elements are encoded as pairs (i,j), where i is rotation exponent and
    j=0/1 records whether a reflection is present. Multiplication is
    (i,j)(k,l) = (i + (-1)^j k, j+l).
    """
    if m < 3:
        raise ValueError("m must be >= 3 for the nontrivial dihedral examples")
    elems = [(i, j) for j in (0, 1) for i in range(m)]
    index = {x: p for p, x in enumerate(elems)}
    table = []
    for i, j in elems:
        row = []
        for k, l in elems:
            prod = ((i + ((-1) ** j) * k) % m, (j + l) % 2)
            row.append(index[prod])
        table.append(row)
    return FiniteMagma(table=table, name=f"D_{m}", identity=index[(0, 0)])


def associativity_witness(A: FiniteMagma) -> Optional[Witness]:
    for x in A.elements:
        for y in A.elements:
            for z in A.elements:
                left = A.mul(A.mul(x, y), z)
                right = A.mul(x, A.mul(y, z))
                if left != right:
                    return Witness(
                        kind="associativity_failure",
                        payload={"x": x, "y": y, "z": z, "left": left, "right": right},
                        message=f"({x}*{y})*{z}={left}, but {x}*({y}*{z})={right}",
                    )
    return None


def identity_witness(A: FiniteMagma, e: Optional[int] = None) -> Optional[Witness]:
    identity = A.identity if e is None else e
    if identity is None:
        return Witness(kind="identity_missing", payload={}, message="No identity candidate supplied")
    for x in A.elements:
        left = A.mul(identity, x)
        right = A.mul(x, identity)
        if left != x or right != x:
            return Witness(
                kind="identity_failure",
                payload={"e": identity, "x": x, "e*x": left, "x*e": right},
                message=f"identity candidate {identity} fails on {x}: e*x={left}, x*e={right}",
            )
    return None


def inverse_witness(A: FiniteMagma, e: Optional[int] = None) -> Optional[Witness]:
    identity = A.identity if e is None else e
    if identity is None:
        return Witness(kind="inverse_no_identity", payload={}, message="Cannot check inverses without identity")
    for x in A.elements:
        found = False
        for y in A.elements:
            if A.mul(x, y) == identity and A.mul(y, x) == identity:
                found = True
                break
        if not found:
            return Witness(
                kind="inverse_failure",
                payload={"e": identity, "x": x},
                message=f"element {x} has no two-sided inverse for identity {identity}",
            )
    return None


def commutativity_witness(A: FiniteMagma) -> Optional[Witness]:
    for x in A.elements:
        for y in A.elements:
            xy = A.mul(x, y)
            yx = A.mul(y, x)
            if xy != yx:
                return Witness(
                    kind="commutativity_failure",
                    payload={"x": x, "y": y, "x*y": xy, "y*x": yx},
                    message=f"{x}*{y}={xy}, but {y}*{x}={yx}",
                )
    return None


def verify_group(A: FiniteMagma) -> Tuple[bool, List[Witness]]:
    """Return whether A is a group, plus any first-order axiom witnesses."""
    witnesses: List[Witness] = []
    for checker in (associativity_witness, identity_witness, inverse_witness):
        w = checker(A)
        if w is not None:
            witnesses.append(w)
    return (len(witnesses) == 0, witnesses)


def verify_abelian_group(A: FiniteMagma) -> Tuple[bool, List[Witness]]:
    ok, witnesses = verify_group(A)
    if not ok:
        return False, witnesses
    w = commutativity_witness(A)
    if w is not None:
        return False, [w]
    return True, []


def find_counterexample_all_groups_abelian(max_order: int = 8) -> CounterexampleCertificate:
    """Find a small group refuting `all finite groups are abelian`.

    The current deterministic search checks dihedral groups D_m. D_3 has order
    6 and is the smallest non-abelian group.
    """
    for m in range(3, max_order + 1):
        A = build_dihedral_group(m)
        ok, group_witnesses = verify_group(A)
        if not ok:
            continue
        w = commutativity_witness(A)
        if w is not None:
            return CounterexampleCertificate(
                claim="all_groups_abelian",
                structure_name=A.name,
                structure_order=A.n,
                assumptions_verified=True,
                refuting_witness=w,
            )
    raise ValueError(f"no counterexample found up to dihedral parameter {max_order}")
