from pathlib import Path

from logical_gans.modelbuilder.core.loader import load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.actions import (
    RelationEdit,
    apply_relation_edit,
    decode_action_index,
    encode_action_index,
    legal_relation_edits,
)

SIG = load_theory(Path(__file__).resolve().parents[1] / "examples" / "theories" / "preorder.json").signature


def test_legal_edits_only_for_unknown_cells():
    s = PartialStructure.empty(SIG, 3)
    s.set_relation("R", (0, 0), Truth.TRUE)
    s.set_relation("R", (1, 1), Truth.FALSE)
    edits = legal_relation_edits(s, "R")
    cells = {e.args for e in edits}
    assert (0, 0) not in cells and (1, 1) not in cells
    # 7 unknown cells * 2 truths
    assert len(edits) == 7 * 2
    for e in edits:
        assert s.get_relation("R", e.args) is Truth.UNKNOWN


def test_apply_edit_copies():
    s = PartialStructure.empty(SIG, 3)
    edit = RelationEdit("R", (0, 1), Truth.TRUE)
    s2 = apply_relation_edit(s, edit)
    assert s.get_relation("R", (0, 1)) is Truth.UNKNOWN  # original unchanged
    assert s2.get_relation("R", (0, 1)) is Truth.TRUE


def test_action_index_roundtrips():
    n = 3
    for i in range(n):
        for j in range(n):
            for value in (Truth.FALSE, Truth.TRUE):
                edit = RelationEdit("R", (i, j), value)
                idx = encode_action_index(edit, n)
                assert 0 <= idx < n * n * 2
                assert decode_action_index(idx, n, "R") == edit
