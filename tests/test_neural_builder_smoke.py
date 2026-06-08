from pathlib import Path

import torch

from logical_gans.modelbuilder.core.loader import load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.builder import neural_relation_build
from logical_gans.modelbuilder.learned.policy_net import RelationPolicyNet

THEORIES = Path(__file__).resolve().parents[1] / "examples" / "theories"


def _tiny_model(tmp_path, n=3):
    torch.manual_seed(0)
    model = RelationPolicyNet(n)
    path = tmp_path / "tiny.pt"
    torch.save({"n": n, "state_dict": model.state_dict(),
                "metadata": {"relation": "R"}}, path)
    return str(path)


def test_builder_runs_and_returns_valid_shape(tmp_path):
    theory = load_theory(THEORIES / "preorder.json")
    out = neural_relation_build(theory, "R", 3, _tiny_model(tmp_path),
                                k=1, budget=20, max_steps=50)
    assert out["status"] in {"satisfied", "failed", "unknown"}
    assert out["builder"] == "neural_relation_policy"
    assert out["relation"] == "R" and out["n"] == 3
    assert isinstance(out["structure"], dict)
    assert isinstance(out["trace"], list)


def test_builder_only_edits_unknown_cells(tmp_path):
    theory = load_theory(THEORIES / "preorder.json")
    out = neural_relation_build(theory, "R", 3, _tiny_model(tmp_path), max_steps=50)
    # replay: every neural action must target a then-UNKNOWN cell
    s = PartialStructure.empty(theory.signature, 3)
    for ev in out["trace"]:
        if ev.get("event") != "neural_action":
            continue
        i, j = ev["chosen_edit"]["args"]
        assert s.get_relation("R", (i, j)) is Truth.UNKNOWN
        s.set_relation("R", (i, j), Truth(ev["chosen_edit"]["value"]))


def test_builder_never_revises_known_cells(tmp_path):
    theory = load_theory(THEORIES / "preorder.json")
    seed = PartialStructure.empty(theory.signature, 3)
    seed.set_relation("R", (0, 0), Truth.TRUE)  # a known cell
    out = neural_relation_build(theory, "R", 3, _tiny_model(tmp_path),
                                seed_structure=seed, max_steps=50)
    assert out["structure"]["relations"]["R(0,0)"] == "true"
