"""Gate R6: minimal God/Devil adversarial training loop.

Proves both neural players participate, the symbolic Judge produces outcomes /
certificates, rewards come from outcomes, and the neural Devil's parameters
actually change after a training step (no fake learning).
"""
import pytest
import torch

from logical_gans.modelbuilder.neural_fraisse import adversarial_train_demo
from logical_gans.modelbuilder.neural_fraisse.adversarial_train_demo import (
    EXAMPLES, build_neural_god, rewards_for, train_episode)
from logical_gans.modelbuilder.neural_fraisse.neural_devil import NeuralDevilPolicy

CYCLE3 = EXAMPLES / "cycle3_fight.json"
CYCLE2 = EXAMPLES / "cycle2_impossible_fight.json"


@pytest.fixture(scope="module")
def god():
    return build_neural_god(epochs=8, seed=0)


@pytest.fixture(scope="module")
def demo_output():
    return adversarial_train_demo.run(epochs=2, seed=0, god_epochs=8)


def test_demo_runs_without_error(demo_output):
    assert isinstance(demo_output, str) and len(demo_output) > 0


def test_output_includes_outcome_counters(demo_output):
    assert "GOD_WINS count:" in demo_output
    assert "DEVIL_WINS count:" in demo_output
    assert "DRAW count:" in demo_output


def test_output_includes_rewards(demo_output):
    assert "God reward:" in demo_output
    assert "Devil reward:" in demo_output


def test_output_includes_losses_or_god_fixed(demo_output):
    assert "Devil loss:" in demo_output
    # God is fixed in this gate -- must be stated explicitly.
    assert "NeuralGod fixed in Gate R6" in demo_output
    assert "God loss:" in demo_output


def test_symbolic_judge_is_used_for_outcomes(demo_output, god):
    assert "Judge outcome:" in demo_output
    # the Judge decides each task's outcome deterministically here
    policy = NeuralDevilPolicy(seed=0)
    opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    m3 = train_episode(policy, opt, god, "Demo A", CYCLE3)
    m2 = train_episode(policy, opt, god, "Demo B", CYCLE2)
    assert m3["outcome"] == "GOD_WINS"
    assert m2["outcome"] == "DEVIL_WINS"


def test_devil_wins_certificate_path_intact(god):
    policy = NeuralDevilPolicy(seed=0)
    opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    m = train_episode(policy, opt, god, "Demo B", CYCLE2)
    cert = m["certificate"]
    assert cert is not None
    assert cert["judge_verified"] is True
    assert cert["budget_exhausted"] is False


def test_rewards_follow_the_scheme():
    assert rewards_for("GOD_WINS") == (1.0, -1.0)
    assert rewards_for("DEVIL_WINS") == (-1.0, 1.0)
    assert rewards_for("DRAW") == (-0.2, -0.2)


def test_neural_devil_parameters_change_after_a_step(god):
    policy = NeuralDevilPolicy(seed=0)
    opt = torch.optim.SGD(policy.parameters(), lr=0.1)
    before = [p.detach().clone() for p in policy.parameters()]
    train_episode(policy, opt, god, "Demo A", CYCLE3)
    after = [p.detach().clone() for p in policy.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))
