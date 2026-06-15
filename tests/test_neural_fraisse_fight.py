"""Tests for the user-facing Neural Fraïssé fight backend."""
from pathlib import Path

from logical_gans.modelbuilder.neural_fraisse.fight import build_task, load_problem, run_fight

PROBLEM = Path(__file__).resolve().parents[1] / "examples" / "problems" / "cycle3_fight.json"


def test_problem_parses():
    problem = load_problem(PROBLEM)
    task = build_task(problem)
    assert task.mode == "refute" and task.kind == "countermodel"
    assert task.domain_size == 3 and task.depth == 3


def test_fight_produces_alternating_trace_and_outcome():
    problem = load_problem(PROBLEM)
    result = run_fight(problem, "active_symbolic")

    assert result["outcome"] in ("GOD_WINS", "DRAW")
    events = result["events"]
    assert events, "fight produced no trace"

    actors = [e.actor for e in events]
    assert actors[0] == "Devil"                       # Devil moves first
    assert {"Devil", "Builder", "Judge"} <= set(actors)

    # alternating Devil -> Builder -> Judge per round
    for i in range(0, len(events) - 1, 3):
        assert events[i].actor == "Devil"
        assert events[i + 1].actor == "Builder"
        assert events[i + 2].actor == "Judge"

    # a Devil challenge names a clause; a Builder reply is a set_* edit
    assert any(e.actor == "Devil" and e.payload.get("clause") for e in events)
    assert any(e.actor == "Builder" and str(e.payload.get("reply", "")).startswith("set_")
               for e in events)


def test_god_wins_refute_has_witness():
    problem = load_problem(PROBLEM)
    result = run_fight(problem, "active_symbolic")
    if result["outcome"] == "GOD_WINS":
        assert result["witness"] is not None
        assert result["structure"] is not None
        s = result["structure"]
        assert any(v is not None for v in s["functions"].values())
        assert any(v is not None for v in s["constants"].values())
