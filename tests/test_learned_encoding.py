from pathlib import Path

import torch

from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.encoding import (
    encode_state,
    relation_tensor,
    witness_mask,
)

THEORIES = Path(__file__).resolve().parents[1] / "examples" / "theories"
SIG = load_theory(THEORIES / "preorder.json").signature


def _structure(cells):
    s = PartialStructure.empty(SIG, 3)
    for (i, j), v in cells.items():
        s.set_relation("R", (i, j), v)
    return s


def _all(value):
    return {(i, j): value for i in range(3) for j in range(3)}


def test_identity_preorder_encodes_correctly():
    cells = {(i, j): (Truth.TRUE if i == j else Truth.FALSE) for i in range(3) for j in range(3)}
    t = relation_tensor(_structure(cells), "R")
    assert t.shape == (3, 3, 3)
    for i in range(3):
        for j in range(3):
            ch = 1 if i == j else 0  # TRUE on diagonal, FALSE off
            assert t[ch, i, j] == 1.0
            assert t[:, i, j].sum() == 1.0


def test_total_preorder_encodes_correctly():
    t = relation_tensor(_structure(_all(Truth.TRUE)), "R")
    assert torch.equal(t[1], torch.ones(3, 3))
    assert t[0].sum() == 0 and t[2].sum() == 0


def test_unknown_cells_encode_correctly():
    t = relation_tensor(_structure(_all(Truth.UNKNOWN)), "R")
    assert torch.equal(t[2], torch.ones(3, 3))


def test_witness_mask_marks_touched_cells():
    cells = {(i, j): Truth.FALSE for i in range(3) for j in range(3)}
    cells.update({(0, 0): Truth.TRUE, (1, 1): Truth.TRUE, (2, 2): Truth.TRUE,
                  (0, 1): Truth.TRUE, (1, 2): Truth.TRUE, (0, 2): Truth.UNKNOWN})
    theory = load_theory(THEORIES / "preorder.json")
    structure = _structure(cells)
    result = run_devil(structure, theory.clauses)
    assert result.status == "unknown"
    mask = witness_mask(result.witness, "R", 3)
    assert mask.shape == (1, 3, 3)

    # the mask must mark exactly the relation cells the witness touched
    expected = {
        tuple(a["arg_values"]) for a in result.witness.touched_atoms
        if a["kind"] == "rel" and a["relation"] == "R"
    }
    assert expected, "expected the witness to touch at least one R cell"
    marked = {(i, j) for i in range(3) for j in range(3) if mask[0, i, j] == 1.0}
    assert marked == expected
    # the blocking UNKNOWN cell (0,2) is among them
    assert (0, 2) in marked


def test_encode_state_has_four_channels():
    t = encode_state(_structure(_all(Truth.UNKNOWN)), "R", witness=None)
    assert t.shape == (4, 3, 3)
    assert t[3].sum() == 0.0  # no witness -> empty mask
