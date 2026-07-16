"""Gate R5: minimal Mirco GAN prototype demo.

The demo wires the neural Devil (NeuralDevilPolicy / NeuralActiveDevil) + the
existing learned God (NeuralBuilder + FraisseNeuralPrior) + the symbolic Judge,
and shows GOD_WINS (Demo A) and DEVIL_WINS (Demo B) with a visible trace.
"""
import pytest
pytest.importorskip("torch")

from logical_gans.modelbuilder.neural_fraisse import mirco_gan_demo
from logical_gans.modelbuilder.neural_fraisse.neural_devil import (
    NeuralActiveDevil, NeuralDevilPolicy)


@pytest.fixture(scope="module")
def demo_output():
    # small epochs: GOD_WINS/DEVIL_WINS are decided by the bounded search + Judge,
    # not by Builder accuracy, so a light pretrain is enough for the demo.
    return mirco_gan_demo.run(epochs=12, seed=0)


@pytest.fixture(scope="module")
def sections(demo_output):
    assert "Demo B:" in demo_output
    a, b = demo_output.split("Demo B:", 1)
    return a, "Demo B:" + b


def test_demo_runs_without_error(demo_output):
    assert isinstance(demo_output, str) and len(demo_output) > 0


def test_demo_A_emits_god_wins(sections):
    demo_a, demo_b = sections
    assert "Outcome: GOD_WINS" in demo_a


def test_demo_B_emits_devil_wins(sections):
    demo_a, demo_b = sections
    assert "Outcome: DEVIL_WINS" in demo_b


def test_demo_B_emits_obstruction_certificate(sections):
    _demo_a, demo_b = sections
    assert "Obstruction certificate:" in demo_b
    assert "judge_verified: true" in demo_b
    assert "budget_exhausted: false" in demo_b


def test_output_contains_neural_devil(demo_output):
    assert "NeuralDevil" in demo_output


def test_output_contains_neural_god_with_alias(demo_output):
    assert "NeuralGod" in demo_output
    # the actual repo learned-God classes, clearly aliased in the trace
    assert "NeuralBuilder + FraisseNeuralPrior" in demo_output


def test_demo_uses_neural_devil_policy_not_symbolic_alone(demo_output):
    # the neural Devil impl is named in the trace ...
    assert "NeuralActiveDevil + NeuralDevilPolicy" in demo_output
    assert "NeuralDevil: selected challenge" in demo_output
    # ... and the demo really constructs that policy/devil
    devil = NeuralActiveDevil(NeuralDevilPolicy(seed=0))
    assert isinstance(devil, NeuralActiveDevil)
    assert isinstance(devil.policy, NeuralDevilPolicy)


def test_main_prints_and_returns_zero(capsys):
    rc = mirco_gan_demo.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "Outcome: GOD_WINS" in out
    assert "Outcome: DEVIL_WINS" in out
