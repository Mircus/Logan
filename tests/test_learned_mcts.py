from pathlib import Path

from logical_gans.modelbuilder.core.loader import load_seed_open_world, load_theory
from logical_gans.modelbuilder.learned.mcts import mcts_relation_build
from logical_gans.modelbuilder.learned.policy_net import RelationPolicyNet

ROOT = Path(__file__).resolve().parents[1]
THEORIES = ROOT / "examples" / "theories"
SEED_FILE = ROOT / "examples" / "seeds" / "preorder_chain_3.json"


def _theory():
    return load_theory(THEORIES / "preorder.json")


def _tiny_model(path, n=3):
    import torch
    torch.manual_seed(0)
    m = RelationPolicyNet(n)
    torch.save({"n": n, "state_dict": m.state_dict(), "metadata": {"relation": "R"}}, path)
    return str(path)


def _seed(theory):
    return load_seed_open_world(SEED_FILE, theory.signature)


def test_mcts_uniform_runs_on_preorder():
    out = mcts_relation_build(_theory(), "R", 3, model_path=None, rollouts=100)
    assert out["status"] in {"satisfied", "failed", "unknown"}
    assert out["uses_neural_policy"] is False
    assert out["builder"] == "mcts_relation"


def test_mcts_neural_runs_on_preorder(tmp_path):
    out = mcts_relation_build(_theory(), "R", 3, model_path=_tiny_model(tmp_path / "m.pt"),
                              rollouts=100)
    assert out["status"] in {"satisfied", "failed", "unknown"}
    assert out["uses_neural_policy"] is True


def test_mcts_returns_json_with_trace_actions_rewards():
    theory = _theory()
    out = mcts_relation_build(theory, "R", 3, seed_structure=_seed(theory), rollouts=100)
    for key in ("status", "builder", "uses_neural_policy", "relation", "n",
                "rollouts", "nodes", "structure", "trace"):
        assert key in out
    assert isinstance(out["trace"], list)
    for ev in out["trace"]:
        assert ev["event"] == "mcts_action"
        assert {"edit", "prior", "visits", "q", "devil_status"} <= set(ev)


def test_mcts_preserves_seed_facts():
    theory = _theory()
    out = mcts_relation_build(theory, "R", 3, seed_structure=_seed(theory), rollouts=100)
    rels = out["structure"]["relations"]
    assert rels["R(0,1)"] == "true"
    assert rels["R(1,2)"] == "true"


def test_mcts_never_revises_known_cells():
    theory = _theory()
    out = mcts_relation_build(theory, "R", 3, seed_structure=_seed(theory), rollouts=100)
    for ev in out["trace"]:
        assert tuple(ev["edit"]["args"]) not in {(0, 1), (1, 2)}


def test_mcts_seeded_includes_forced_fact():
    # uniform priors suffice: any complete preorder extending the chain forces R(0,2)
    theory = _theory()
    out = mcts_relation_build(theory, "R", 3, seed_structure=_seed(theory), rollouts=200)
    assert out["status"] == "satisfied"
    assert out["structure"]["relations"]["R(0,2)"] == "true"


def test_demo_script_writes_result_json():
    import experiments.neural_mcts_preorder_demo as demo
    rc = demo.main(samples=300, epochs=120, rollouts=100)
    assert demo.OUT.is_file()
    data = __import__("json").loads(demo.OUT.read_text(encoding="utf-8"))
    assert "neural" in data and "uniform" in data
    assert data["neural"]["forced_fact_present"] is True
    assert data["neural"]["exhaustive_devil"] == "ok"
    assert rc == 0
