"""Neural Fraïssé benchmark.

Compares four methods on HELD-OUT tasks (train sizes disjoint from test sizes):
  1. uniform            : active Devil + random Builder replies
  2. obligation_first   : active Devil + symbolic suggested-value replies
  3. neural_passive     : neural policy + free search (Builder-first, Devil verifies)
  4. neural_active      : neural policy + active Devil game (Devil challenges, Builder replies)

Prints a table + one neural_active trace and a PASS/FAIL verdict.

    python -m logical_gans.modelbuilder.neural_fraisse.benchmark \
      --train-sizes 3,4 --test-sizes 5,6 --tasks 50 --budget 800 --seed 0
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ..learned.semantic_search import tree_policy_search
from . import results as R
from .game import play_game
from .players import (
    FraisseNeuralPrior,
    NeuralBuilder,
    ObligationFirstBuilder,
    SymbolicActiveDevil,
    UniformBuilder,
)
from .tasks import generate_tasks
from .training import train_builder


def _run_passive(task, prior, budget):
    res = tree_policy_search(task.theory, task.goal, task.domain_size, prior,
                             mode=task.mode, k=task.depth, devil_budget=None,
                             search_budget=budget)
    if res["success"]:
        return "won", res["nodes_evaluated"]
    if res.get("exhausted"):
        return "lost", res["nodes_evaluated"]
    return "draw", res["nodes_evaluated"]


def run_benchmark(train_sizes, test_sizes, n_tasks, budget, seed, epochs=120, verbose=True):
    # disjoint train / held-out test task families
    train_tasks = generate_tasks(train_sizes, n_tasks, seed=seed)
    test_tasks = generate_tasks(test_sizes, n_tasks, seed=seed + 1000)
    assert set(train_sizes).isdisjoint(set(test_sizes)), "train/test sizes must be disjoint"

    tmp = Path(tempfile.gettempdir()) / "logan_nf"
    tmp.mkdir(parents=True, exist_ok=True)
    data_path, model_path = tmp / "builder_train.jsonl", tmp / "builder.pt"
    train_meta = train_builder(train_tasks, data_path, model_path, epochs=epochs, seed=seed)
    prior = FraisseNeuralPrior(str(model_path))

    devil = SymbolicActiveDevil()
    rows = []
    example_trace = None
    for task in test_tasks:
        # 1 uniform (active)
        o = play_game(task, devil, UniformBuilder(seed=seed), budget)
        rows.append({"method": "uniform", "kind": task.kind, "outcome": o.outcome, "nodes": o.nodes})
        # 2 obligation_first (active)
        o = play_game(task, devil, ObligationFirstBuilder(), budget)
        rows.append({"method": "obligation_first", "kind": task.kind, "outcome": o.outcome, "nodes": o.nodes})
        # 3 neural_passive (free search)
        outcome, nodes = _run_passive(task, prior, budget)
        rows.append({"method": "neural_passive", "kind": task.kind, "outcome": outcome, "nodes": nodes})
        # 4 neural_active (active game)
        o = play_game(task, devil, NeuralBuilder(prior), budget)
        rows.append({"method": "neural_active", "kind": task.kind, "outcome": o.outcome, "nodes": o.nodes})
        if example_trace is None and o.outcome == "won" and o.trace:
            example_trace = (task, o.trace)

    agg = R.aggregate(rows)
    ov = {m: R.overall(rows, m) for m in
          ("uniform", "obligation_first", "neural_passive", "neural_active")}

    na, un, ob, pa = (ov["neural_active"], ov["uniform"],
                      ov["obligation_first"], ov["neural_passive"])
    # uniform & obligation_first are the SAME-active-Devil non-neural baselines.
    active_base_succ = max(un["success_rate"], ob["success_rate"])
    active_base_nodes = min(un["median_nodes"], ob["median_nodes"])
    beats_active_success = na["success_rate"] > active_base_succ
    ties_active_cheaper = (na["success_rate"] >= active_base_succ
                           and na["success_rate"] > 0
                           and na["median_nodes"] * 2 <= active_base_nodes)
    beats_active = beats_active_success or ties_active_cheaper
    beats_passive = (na["success_rate"] > pa["success_rate"]
                     or (na["success_rate"] >= pa["success_rate"] and na["success_rate"] > 0
                         and na["median_nodes"] * 2 <= pa["median_nodes"]))
    verdict = "PASS" if beats_active else ("PARTIAL" if beats_passive else "FAIL")
    passed = verdict == "PASS"

    if verbose:
        print("baselines (all share ONE active symbolic Devil except neural_passive):")
        print("  uniform           : active Devil + random matching Builder reply")
        print("  obligation_first  : active Devil + symbolic suggested-value Builder reply")
        print("  neural_passive    : learned policy + FREE search, NO active Devil")
        print("  neural_active     : active Devil + learned Builder (trained on active-game traces)")
        print()
        print(f"train tasks: {len(train_tasks)} (sizes {sorted(set(train_sizes))}); "
              f"held-out test tasks: {len(test_tasks)} (sizes {sorted(set(test_sizes))}); "
              f"training examples: {train_meta['examples']}")
        print()
        print(R.format_table(agg))
        print()
        print("overall held-out (success / median_nodes / draws):")
        for m in ("uniform", "obligation_first", "neural_passive", "neural_active"):
            s = ov[m]
            print(f"  {m:<20} success={s['success_rate']:.2f}  median_nodes={s['median_nodes']}  draws={s['draws']}")
        print()
        if example_trace is not None:
            task, trace = example_trace
            print(f"example neural_active trace on held-out task '{task.name}':")
            for ev in trace[:10]:
                print(f"  {ev.actor:<7} {json.dumps(ev.payload)}")
        print()
        print(f"NEURAL_FRAISSE_POC = {verdict}")
        print(f"  vs same-active-Devil non-neural Builders: neural_active success "
              f"{na['success_rate']:.2f} (nodes {na['median_nodes']}) | uniform "
              f"{un['success_rate']:.2f} ({un['median_nodes']}) | obligation_first "
              f"{ob['success_rate']:.2f} ({ob['median_nodes']})")
        print(f"  vs passive (no active Devil): neural_passive success "
              f"{pa['success_rate']:.2f} (nodes {pa['median_nodes']})")
        if verdict == "PASS" and beats_active_success:
            print("  -> beats the active non-neural baselines on SUCCESS RATE: neural Fraïssé learns")
        elif verdict == "PASS":
            print("  -> ties success but >=2x cheaper than active non-neural baselines")
        elif verdict == "PARTIAL":
            print("  -> beats passive search but NOT the active non-neural Builder baseline")
        else:
            print("  -> does not beat the active non-neural baselines")

    return {"aggregate": agg, "overall": ov, "passed": passed, "verdict": verdict,
            "example_trace": example_trace, "rows": rows}


def _sizes(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="neural-fraisse-benchmark")
    p.add_argument("--train-sizes", default="3,4")
    p.add_argument("--test-sizes", default="5,6")
    p.add_argument("--tasks", type=int, default=50)
    p.add_argument("--budget", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=120)
    a = p.parse_args(argv)
    res = run_benchmark(_sizes(a.train_sizes), _sizes(a.test_sizes),
                        a.tasks, a.budget, a.seed, epochs=a.epochs)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
