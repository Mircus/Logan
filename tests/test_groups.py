"""Regression tests for the demoted known-group fixtures.

Known groups are NOT the model builder; they are sanity fixtures.
The real kernel lives in tests/test_core_*.py.
"""
from logical_gans.modelbuilder.fixtures.known_groups import (
    cyclic_group,
    dihedral_group,
    is_abelian,
    is_group,
    klein_four_group,
)


def test_cyclic_groups_are_abelian_groups():
    for n in [1, 2, 5, 8]:
        g = cyclic_group(n)
        assert is_group(g)
        assert is_abelian(g)


def test_klein_four_group_is_abelian_group():
    g = klein_four_group()
    assert is_group(g)
    assert is_abelian(g)


def test_dihedral_three_is_a_nonabelian_group():
    g = dihedral_group(3)
    assert is_group(g)
    assert not is_abelian(g)
