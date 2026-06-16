from pathlib import Path

import pytest

from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_seed_open_world, load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.builder import neural_relation_build
from logical_gans.modelbuilder.learned.data import (
    make_relation_training_examples,
    write_training_examples_jsonl,
)
from logical_gans.modelbuilder.learned.policy_net import RelationPolicyNet
from logical_gans.modelbuilder.learned.train import train_relation_policy

ROOT = Path(__file__).resolve().parents[1]
THEORIES = ROOT / "examples" / "theories"
SEEDS = ROOT / "examples" / "seeds"
SEED_FILE = SEEDS / "preorder_chain_3.json"


def _theory():
    return load_theory(THEORIES / "preorder.json")


def _tiny_model(path, n=3):
    import torch
    torch.manual_seed(0)
    m = RelationPolicyNet(n)
    torch.save({"n": n, "state_dict": m.state_dict(), "metadata": {"relation": "R"}}, path)
    return str(path)


def _build_from_seed(theory, model_path):
    seed = load_seed_open_world(SEED_FILE, theory.signature)
    return neural_relation_build(theory, "R", 3, model_path, seed_structure=seed, max_steps=50)


def test_open_world_seed_leaves_unspecified_cells_unknown():
    sig = _theory().signature
    s = load_seed_open_world(SEED_FILE, sig)
    assert s.get_relation("R", (0, 1)) is Truth.TRUE
    assert s.get_relation("R", (1, 2)) is Truth.TRUE
    for i in range(3):
        for j in range(3):
            if (i, j) not in {(0, 1), (1, 2)}:
                assert s.get_relation("R", (i, j)) is Truth.UNKNOWN


def test_seed_facts_preserved_by_builder(tmp_path):
    theory = _theory()
    out = _build_from_seed(theory, _tiny_model(tmp_path / "m.pt"))
    assert out["structure"]["relations"]["R(0,1)"] == "true"
    assert out["structure"]["relations"]["R(1,2)"] == "true"


def test_builder_never_revises_seed_known_cells(tmp_path):
    theory = _theory()
    out = _build_from_seed(theory, _tiny_model(tmp_path / "m.pt"))
    for ev in out["trace"]:
        if ev.get("event") == "neural_action":
            assert tuple(ev["chosen_edit"]["args"]) not in {(0, 1), (1, 2)}


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    theory = _theory()
    d = tmp_path_factory.mktemp("seeded")
    data, model = d / "data.jsonl", d / "model.pt"
    examples = make_relation_training_examples(theory, "R", 3, 800, seed=0)
    write_training_examples_jsonl(examples, data)
    train_relation_policy(str(data), str(model), 3, epochs=300, lr=1e-3, seed=0)
    return str(model)


def test_seeded_builder_produces_forced_fact(trained_model):
    out = _build_from_seed(_theory(), trained_model)
    assert out["status"] == "satisfied"
    assert out["structure"]["relations"]["R(0,2)"] == "true"


def test_seeded_final_structure_passes_exhaustive_devil(trained_model):
    theory = _theory()
    out = _build_from_seed(theory, trained_model)
    final = PartialStructure.empty(theory.signature, 3)
    for key, val in out["structure"]["relations"].items():
        inside = key[key.index("(") + 1: key.index(")")]
        i, j = (int(x) for x in inside.split(","))
        final.set_relation("R", (i, j), Truth(val))
    assert run_devil(final, theory.clauses).status == "ok"
