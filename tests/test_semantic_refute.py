from pathlib import Path

import pytest
import pytest
torch = pytest.importorskip("torch")

from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_claim, load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.generic_mcts import mcts_semantic_refute
from logical_gans.modelbuilder.learned.priors import NeuralSemanticPrior
from logical_gans.modelbuilder.learned.semantic_actions import (
    SetConstant,
    SetFunction,
    SetRelation,
)
from logical_gans.modelbuilder.learned.semantic_features import FEATURE_DIM
from logical_gans.modelbuilder.learned.semantic_policy_net import SemanticPolicyNet
from logical_gans.modelbuilder.learned.semantic_training import (
    make_semantic_training_examples,
    train_semantic_policy,
    write_semantic_training_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "examples" / "theories" / "toy_mixed_signature.json"
CLAIM = ROOT / "examples" / "claims" / "toy_reflexive_E.json"


def _theory_claim():
    return load_theory(THEORY), load_claim(CLAIM)


def _rebuild(theory, sj, n=2):
    A = PartialStructure.empty(theory.signature, n)
    for key, val in sj["relations"].items():
        inside = key[key.index("(") + 1: key.index(")")]
        A.set_relation(key[: key.index("(")], tuple(int(x) for x in inside.split(",")), Truth(val))
    for key, val in sj["functions"].items():
        if val is not None:
            inside = key[key.index("(") + 1: key.index(")")]
            A.set_function(key[: key.index("(")], tuple(int(x) for x in inside.split(",")), val)
    for name, val in sj["constants"].items():
        if val is not None:
            A.set_constant(name, val)
    return A


def _tiny_model(path):
    torch.manual_seed(0)
    m = SemanticPolicyNet(FEATURE_DIM)
    torch.save({"feature_dim": FEATURE_DIM, "state_dict": m.state_dict(), "metadata": {}}, path)
    return str(path)


def test_semantic_training_data_produced():
    theory, claim = _theory_claim()
    examples = make_semantic_training_examples(theory, claim, 2, num_samples=30, seed=0)
    assert len(examples) >= 1
    ex = examples[0]
    assert all(len(row) == FEATURE_DIM for row in ex["features"])
    assert 0 <= ex["target"] < len(ex["features"])


def test_train_produces_model_file(tmp_path):
    theory, claim = _theory_claim()
    examples = make_semantic_training_examples(theory, claim, 2, num_samples=40, seed=0)
    data = tmp_path / "d.jsonl"
    write_semantic_training_jsonl(examples, data)
    info = train_semantic_policy(str(data), str(tmp_path / "m.pt"), epochs=20)
    assert (tmp_path / "m.pt").is_file()
    ckpt = torch.load(tmp_path / "m.pt", weights_only=False)
    assert ckpt["feature_dim"] == FEATURE_DIM and "state_dict" in ckpt


def test_neural_semantic_prior_scores_generic_edits(tmp_path):
    theory, _ = _theory_claim()
    prior = NeuralSemanticPrior(_tiny_model(tmp_path / "m.pt"))
    s = PartialStructure.empty(theory.signature, 2)
    edits = [SetRelation("E", (0, 0), Truth.TRUE), SetFunction("s", (0,), 1), SetConstant("a", 0)]
    scores = prior.score(s, None, edits)
    assert set(scores) == set(edits)
    assert all(isinstance(v, float) for v in scores.values())


def test_invalid_checkpoint_rejected(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"not": "a model"}, bad)
    with pytest.raises(ValueError):
        NeuralSemanticPrior(str(bad))


def test_mcts_semantic_refute_returns_refuted():
    theory, claim = _theory_claim()
    res = mcts_semantic_refute(theory, claim, 2, prior_provider=None, rollouts=500, seed=0)
    assert res["status"] == "refuted"
    assert res["theory_status"] == "satisfied"
    assert res["claim_status"] == "failed"
    sj = res["structure"]
    # final structure contains E (relation), s (function), and a (constant)
    assert any(v != "unknown" for v in sj["relations"].values())
    assert any(v is not None for v in sj["functions"].values())
    assert any(v is not None for v in sj["constants"].values())
    # claim witness verifies the claim failure independently
    assert res["claim_witness"] is not None
    rebuilt = _rebuild(theory, sj)
    assert run_devil(rebuilt, theory.clauses).status == "ok"            # theory verifies
    claim_devil = run_devil(rebuilt, claim.clauses)
    assert claim_devil.status == "failed"                               # claim fails
    assert claim_devil.witness.to_json()["assignment"] == res["claim_witness"]["assignment"]


def test_mcts_semantic_refute_with_neural_prior(tmp_path):
    theory, claim = _theory_claim()
    prior = NeuralSemanticPrior(_tiny_model(tmp_path / "m.pt"))
    res = mcts_semantic_refute(theory, claim, 2, prior_provider=prior, rollouts=500, seed=0)
    assert res["status"] == "refuted"
    assert res["uses_neural_policy"] is True
    assert res["theory_status"] == "satisfied" and res["claim_status"] == "failed"


def test_no_hardcoded_model_constructor_used():
    # the signature-parametric refute path must not call named-domain constructors
    forbidden = ("known_groups", "cyclic_group", "dihedral_group", "klein_four", "build_cyclic")
    learned = ROOT / "src" / "logical_gans" / "modelbuilder" / "learned"
    for mod in ("generic_mcts.py", "semantic_actions.py", "semantic_training.py",
                "priors.py", "semantic_features.py", "semantic_policy_net.py"):
        text = (learned / mod).read_text(encoding="utf-8")
        for bad in forbidden:
            assert bad not in text, f"{mod} references {bad}"
