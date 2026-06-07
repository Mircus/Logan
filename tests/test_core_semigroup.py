from logical_gans.modelbuilder.core.generator import generate
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.examples.semigroup import (
    empty_semigroup_structure,
    semigroup_clauses,
)


def test_semigroup_n1_succeeds():
    result = generate(empty_semigroup_structure(1), semigroup_clauses())
    assert result.status == "satisfied"
    # the single function cell got filled
    assert result.structure.get_function("mul", (0, 0)) == 0
    assert result.structure.unknown_function_cells() == []
