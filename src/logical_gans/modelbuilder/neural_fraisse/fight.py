"""User-facing God-vs-Devil fight: the Neural Fraïssé front door.

    python -m logical_gans.modelbuilder.neural_fraisse.fight examples/problems/cycle3_fight.json
    python -m logical_gans.modelbuilder.neural_fraisse.fight <problem>.json --builder neural_active

Reads a fight problem (signature, assumptions, goal, resources, builder mode),
runs the active Devil vs the Builder (God), and prints a readable alternating
trace plus the final outcome / structure / witness.

Modes:
  active_symbolic : active Devil + symbolic Builder (fast, default; no training)
  neural_active   : active Devil + learned Builder (auto-trained on the cyclic
                    task family; slower)
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from ..capsule import _formula_to_clause
from ..core.devil import run_devil_bounded
from ..core.loader import parse_clause, parse_signature
from ..core.partial_structure import PartialStructure
from ..core.theory import Claim, Theory
from ..learned.semantic_actions import apply_semantic_edit, semantic_edit_to_json
from .game import play_game
from .players import (
    FraisseNeuralPrior,
    NeuralBuilder,
    ObligationFirstBuilder,
    SymbolicActiveDevil,
)
from .tasks import Task


def load_problem(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_task(problem: dict) -> Task:
    sig = parse_signature(problem["signature"])
    theory = Theory(name=problem.get("name", "theory"), signature=sig,
                    clauses=[parse_clause(_formula_to_clause(f, f"axiom_{i}", sig))
                             for i, f in enumerate(problem["theory"])])
    goal = problem["goal"]
    gmode = goal["mode"]
    r = problem["resources"]
    if gmode == "refute":
        claim = Claim("claim", [parse_clause(_formula_to_clause(goal["claim"], "claim", sig))])
        mode, kind = "refute", "countermodel"
    elif gmode in ("model", "satisfy"):
        claim = Claim("true", [])
        mode, kind = "satisfy", "model"
    else:
        raise ValueError(f"goal.mode must be 'refute' or 'model' (got {gmode!r})")
    return Task(name=problem.get("name", "fight"), signature=sig, theory=theory, goal=claim,
                mode=mode, kind=kind, domain_size=int(r["domain_size"]),
                depth=int(r["depth"]), m=int(r["depth"]))


def _final_structure_and_witness(task, decisions):
    A = PartialStructure.empty(task.signature, task.domain_size)
    for (_s, _move, edit) in decisions:
        A = apply_semantic_edit(A, edit)
    structure = A.to_json()
    witness = None
    if task.mode == "refute":
        dC = run_devil_bounded(A, task.goal.clauses, k=task.depth, budget=None)
        if dC.witness is not None:
            witness = dC.witness.to_json()
    return structure, witness


def run_fight(problem: dict, builder_mode: str, epochs: int = 120):
    task = build_task(problem)
    budget = int(problem["resources"]["budget"])
    devil = SymbolicActiveDevil()

    if builder_mode == "neural_active":
        from .tasks import generate_tasks
        from .training import train_builder
        tmp = Path(tempfile.gettempdir()) / "logan_nf_fight"
        tmp.mkdir(parents=True, exist_ok=True)
        train_tasks = generate_tasks([3, 4], 30, seed=0)
        train_builder(train_tasks, tmp / "fight_builder.jsonl", tmp / "fight_builder.pt",
                      epochs=epochs, seed=0)
        builder = NeuralBuilder(FraisseNeuralPrior(str(tmp / "fight_builder.pt")))
    elif builder_mode == "active_symbolic":
        builder = ObligationFirstBuilder()
    else:
        raise ValueError(f"unknown builder mode {builder_mode!r}")

    out = play_game(task, devil, builder, budget)
    outcome = "GOD_WINS" if out.outcome == "won" else "DRAW"
    structure, witness = (None, None)
    if out.outcome == "won":
        structure, witness = _final_structure_and_witness(task, out.decisions)
    return {"task": task, "builder_mode": builder_mode, "outcome": outcome,
            "decisions": out.decisions, "events": out.trace, "nodes": out.nodes,
            "structure": structure, "witness": witness}


# --------------------------------------------------------------------------
# Readable rendering
# --------------------------------------------------------------------------
def _fmt_cell(cell):
    if not cell:
        return "?"
    args = "" if cell.get("args") is None else "(" + ",".join(map(str, cell["args"])) + ")"
    return f"{cell['symbol']}{args}"


def _fmt_edit(edit):
    e = semantic_edit_to_json(edit)
    args = "(" + ",".join(map(str, e.get("args", []))) + ")" if "args" in e and e.get("args") is not None else ""
    return f"{e['kind']} {e['symbol']}{args}={e['value']}"


def render(result, problem) -> str:
    task = result["task"]
    sig = problem["signature"]
    lines = ["LOGAN Neural Fraïssé — God vs Devil", "=" * 38, ""]
    lines.append(f"Problem: {problem.get('name', 'fight')}")
    rel = ", ".join(f"{r['name']}/{r['arity']}" for r in sig.get("relations", []))
    fn = ", ".join(f"{f['name']}/{f['arity']}" for f in sig.get("functions", []))
    consts = ", ".join(sig.get("constants", []))
    lines.append(f"Signature: relations [{rel}]  functions [{fn}]  constants [{consts}]")
    lines.append("Assumptions (theory):")
    for f in problem["theory"]:
        lines.append(f"  - {f}")
    g = problem["goal"]
    if g["mode"] == "refute":
        lines.append(f"Goal: refute the claim  {g['claim']}  (build A with A ⊨ T and A ⊭ C)")
    else:
        lines.append("Goal: build a model of the assumptions")
    r = problem["resources"]
    lines.append(f"Resources: domain_size={r['domain_size']}  depth k={r['depth']}  "
                 f"budget={r['budget']}  max_rounds={r['max_rounds']}")
    lines.append(f"Mode: {result['builder_mode']}  (God=Builder, Devil=active symbolic challenger)")
    lines.append("")
    lines.append("Fight:")
    if result["decisions"]:
        for i, (_s, move, edit) in enumerate(result["decisions"], 1):
            asg = "" if not move.assignment else " " + ",".join(f"{k}={v}" for k, v in move.assignment.items())
            lines.append(f"  Round {i}")
            lines.append(f"    Devil: {move.kind} clause={move.clause_name}{asg} target={_fmt_cell(move.target_cell)}")
            lines.append(f"    God:   {_fmt_edit(edit)}")
            lines.append(f"    Judge: progress")
    else:
        lines.append("  (no verified winning line within the bounded resources)")
    lines.append("")
    lines.append(f"Outcome: {result['outcome']}")
    if result["outcome"] == "GOD_WINS":
        lines.append(f"  (God survived the Devil and achieved the goal in {len(result['decisions'])} rounds, "
                     f"{result['nodes']} nodes)")
        s = result["structure"]
        if s:
            consts2 = ", ".join(f"{k}={v}" for k, v in s["constants"].items() if v is not None)
            fns = ", ".join(f"{k}={v}" for k, v in s["functions"].items() if v is not None)
            rels = ", ".join(f"{k}=true" for k, v in s["relations"].items() if v == "true")
            lines.append("Final structure:")
            lines.append(f"  constants: {consts2}")
            lines.append(f"  functions: {fns}")
            lines.append(f"  relations: {rels}")
        if result["witness"]:
            lines.append("Witness:")
            lines.append(f"  {result['witness'].get('message', '(witness)')}")
    else:
        lines.append("  DRAW: ran out of bounded search/budget without a verified win. "
                     "(DEVIL_WINS is not implemented in this release.)")
    return "\n".join(lines)


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    p = argparse.ArgumentParser(prog="neural-fraisse-fight")
    p.add_argument("problem", help="path to a fight problem JSON")
    p.add_argument("--builder", default=None,
                   choices=["active_symbolic", "neural_active"],
                   help="override builder mode (default: from the problem file)")
    p.add_argument("--epochs", type=int, default=120)
    a = p.parse_args(argv)
    problem = load_problem(a.problem)
    builder_mode = a.builder or problem.get("builder", {}).get("mode", "active_symbolic")
    result = run_fight(problem, builder_mode, epochs=a.epochs)
    print(render(result, problem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
