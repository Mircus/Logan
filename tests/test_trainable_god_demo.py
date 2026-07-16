"""Gate R8: minimal judged God/Devil co-training.

Proves BOTH neural policies are trained (parameters actually change), the symbolic
Judge decides outcomes, and the DEVIL_WINS certificate / GOD_WINS paths still work.
No fake God loss; no no_grad around the trainable God path.
"""
import pytest
import pytest
torch = pytest.importorskip("torch")

from logical_gans.modelbuilder.neural_fraisse import cotraining_demo
from logical_gans.modelbuilder.neural_fraisse.cotraining_demo import (
    EXAMPLES, cotrain_episode)
from logical_gans.modelbuilder.neural_fraisse.neural_devil import NeuralDevilPolicy
from logical_gans.modelbuilder.neural_fraisse.neural_god import NeuralGodPolicy

CYCLE3 = EXAMPLES / "cycle3_fight.json"
CYCLE2 = EXAMPLES / "cycle2_impossible_fight.json"


def _fresh():
    dp = NeuralDevilPolicy(seed=0)
    gp = NeuralGodPolicy(seed=0)
    return dp, gp, torch.optim.SGD(dp.parameters(), lr=0.1), torch.optim.SGD(gp.parameters(), lr=0.1)


@pytest.fixture(scope="module")
def demo_output():
    return cotraining_demo.run(epochs=2, seed=0)


def test_demo_runs_and_reports_metrics(demo_output):
    for token in ("GOD_WINS count:", "DEVIL_WINS count:", "DRAW count:",
                  "God reward:", "Devil reward:", "God loss:", "Devil loss:",
                  "Judge outcome:"):
        assert token in demo_output


def test_god_loss_is_real_not_fake(demo_output):
    # R6 said "God loss: n/a (NeuralGod fixed)"; R8 must report a real number.
    assert "n/a" not in demo_output
    assert "NeuralGodPolicy" in demo_output      # trainable wrapper clearly named


def test_devil_parameters_change_after_a_step():
    dp, gp, od, og = _fresh()
    before = [p.detach().clone() for p in dp.parameters()]
    cotrain_episode(dp, gp, od, og, "Demo A", CYCLE3)
    after = [p.detach().clone() for p in dp.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_god_parameters_change_after_a_step():
    dp, gp, od, og = _fresh()
    before = [p.detach().clone() for p in gp.parameters()]
    cotrain_episode(dp, gp, od, og, "Demo A", CYCLE3)
    after = [p.detach().clone() for p in gp.parameters()]
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_symbolic_judge_decides_outcomes():
    dp, gp, od, og = _fresh()
    m3 = cotrain_episode(dp, gp, od, og, "Demo A", CYCLE3)
    m2 = cotrain_episode(dp, gp, od, og, "Demo B", CYCLE2)
    assert m3["outcome"] == "GOD_WINS"
    assert m2["outcome"] == "DEVIL_WINS"


def test_devil_wins_certificate_still_works():
    dp, gp, od, og = _fresh()
    m = cotrain_episode(dp, gp, od, og, "Demo B", CYCLE2)
    cert = m["certificate"]
    assert cert is not None
    assert cert["judge_verified"] is True
    assert cert["budget_exhausted"] is False


def test_god_wins_still_works():
    dp, gp, od, og = _fresh()
    m = cotrain_episode(dp, gp, od, og, "Demo A", CYCLE3)
    assert m["outcome"] == "GOD_WINS"
    assert m["god_reward"] == 1.0 and m["devil_reward"] == -1.0
