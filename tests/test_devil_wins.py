"""Gate R2: bounded DEVIL_WINS obstruction for the n=2 impossible cyclic demo.

Proves DEVIL_WINS is a Judge-verified bounded obstruction, not a search timeout.
"""
import copy
from pathlib import Path

from logical_gans.modelbuilder.neural_fraisse.certificate import build_obstruction_certificate
from logical_gans.modelbuilder.neural_fraisse.fight import build_task, load_problem, run_fight

ROOT = Path(__file__).resolve().parents[1]
CYCLE2 = ROOT / "examples" / "problems" / "cycle2_impossible_fight.json"
CYCLE3 = ROOT / "examples" / "problems" / "cycle3_fight.json"


def test_cycle2_impossible_is_devil_wins():
    result = run_fight(load_problem(CYCLE2), "active_symbolic")
    assert result["outcome"] == "DEVIL_WINS"


def test_devil_wins_is_not_timeout():
    result = run_fight(load_problem(CYCLE2), "active_symbolic")
    cert = result["certificate"]
    assert cert is not None
    assert cert["budget_exhausted"] is False          # frontier closed, NOT a budget timeout


def test_obstruction_certificate_present_and_judge_verified():
    result = run_fight(load_problem(CYCLE2), "active_symbolic")
    cert = result["certificate"]
    assert cert is not None
    assert cert["judge_verified"] is True
    # the symbolic Judge (run_devil) found zero winning completions over the bounded space
    assert cert["winning_completions"] == 0
    assert cert["completions_checked"] > 0


def test_certificate_independently_reverified_by_judge():
    # Re-run the exhaustive Judge check directly (independent of the game DFS).
    task = build_task(load_problem(CYCLE2))
    cert = build_obstruction_certificate(task)
    assert cert["judge_verified"] is True
    assert cert["winning_completions"] == 0
    # sanity: it really enumerated the n=2 completion space (16 E * 4 s * 2 a)
    assert cert["completions_checked"] == 128


def test_tiny_budget_is_draw_not_devil_wins():
    problem = copy.deepcopy(load_problem(CYCLE2))
    problem["resources"]["budget"] = 1            # force a budget timeout
    result = run_fight(problem, "active_symbolic")
    assert result["outcome"] == "DRAW"            # NOT DEVIL_WINS
    assert result["certificate"] is None


def test_god_wins_still_works():
    result = run_fight(load_problem(CYCLE3), "active_symbolic")
    assert result["outcome"] == "GOD_WINS"
    assert result["witness"] is not None
