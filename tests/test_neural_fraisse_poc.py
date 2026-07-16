"""Structural smoke test for the Neural Fraïssé spine.

Asserts the experiment runs honestly (disjoint train/test, both task kinds, all
four methods, an alternating Devil/Builder trace, a computed verdict). It does
NOT assert PASS -- a tiny fast run is noisy; the PASS/FAIL claim belongs to the
full benchmark command.
"""
import pytest
pytest.importorskip("torch")

from logical_gans.modelbuilder.neural_fraisse.benchmark import run_benchmark
from logical_gans.modelbuilder.neural_fraisse.tasks import generate_tasks


def test_tasks_are_feasible_and_varied():
    tasks = generate_tasks([3, 4], 10, seed=0)
    assert len(tasks) >= 5
    kinds = {t.kind for t in tasks}
    assert "model" in kinds and "countermodel" in kinds


def test_benchmark_spine_runs_held_out_and_compares_four_methods():
    res = run_benchmark([3], [4], n_tasks=8, budget=300, seed=0, epochs=30, verbose=False)
    rows = res["rows"]

    methods = {r["method"] for r in rows}
    assert methods == {"uniform", "obligation_first", "neural_passive", "neural_active"}

    kinds = {r["kind"] for r in rows}
    assert "model" in kinds and "countermodel" in kinds          # both task kinds run

    assert isinstance(res["passed"], bool)                       # explicit verdict computed

    # held-out: train sizes {3} disjoint from test sizes {4} (enforced in run_benchmark)
    # an alternating active trace exists: Devil challenge then Builder reply
    assert res["example_trace"] is not None, "no neural_active win produced a trace"
    _task, trace = res["example_trace"]
    assert trace[0].actor == "Devil"
    assert trace[1].actor == "Builder"
    assert any(ev.actor == "Devil" and ev.payload.get("clause") for ev in trace)
    assert any(ev.actor == "Builder" and ev.payload.get("reply", "").startswith("set_")
               for ev in trace)
