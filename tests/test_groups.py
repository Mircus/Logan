from logical_gans.modelbuilder.theories.groups import (
    build_cyclic_group,
    build_dihedral_group,
    build_klein_four_group,
    commutativity_witness,
    find_counterexample_all_groups_abelian,
    verify_group,
)


def test_cyclic_groups_are_groups_and_abelian():
    for n in [1, 2, 5, 8]:
        A = build_cyclic_group(n)
        ok, witnesses = verify_group(A)
        assert ok, witnesses
        assert commutativity_witness(A) is None


def test_klein_four_group_is_group_and_abelian():
    A = build_klein_four_group()
    ok, witnesses = verify_group(A)
    assert ok, witnesses
    assert commutativity_witness(A) is None


def test_dihedral_three_is_group_but_not_abelian():
    A = build_dihedral_group(3)
    ok, witnesses = verify_group(A)
    assert ok, witnesses
    w = commutativity_witness(A)
    assert w is not None
    assert w.kind == "commutativity_failure"


def test_counterexample_all_groups_abelian():
    cert = find_counterexample_all_groups_abelian(max_order=8)
    assert cert.assumptions_verified is True
    assert cert.structure_name == "D_3"
    assert cert.refuting_witness.kind == "commutativity_failure"
