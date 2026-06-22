"""Minimal Mirco GAN prototype demo (Gate R5).

Shows the minimal Prototype-1 architecture wired together on two tiny problems:

    neural Devil / Adversary   -> NeuralDevilPolicy via NeuralActiveDevil (Gate R4)
    neural God / Builder       -> NeuralBuilder + FraisseNeuralPrior (existing learned God)
    symbolic Judge / Verifier  -> the existing bounded judge / run_devil

    Demo A: cycle3            -> GOD_WINS
    Demo B: cycle2 impossible -> DEVIL_WINS  (+ obstruction certificate)

This is NOT full adversarial GAN training. The neural God here is the repo's
existing supervised-pretrained Builder (`NeuralBuilder`/`FraisseNeuralPrior`),
aliased "NeuralGod" in the trace; the neural Devil only *scores already-legal*
symbolic challenges (it never invents moves). No reward learning, no adversarial
loop -- those are later gates.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .certificate import build_obstruction_certificate
from .fight import _fmt_cell, _fmt_edit, build_task, load_problem
from .game import GameState, judge, legal_replies, play_game
from .neural_devil import NeuralActiveDevil, NeuralDevilPolicy
from .players import NeuralBuilder, FraisseNeuralPrior, enumerate_legal_devil_moves

ROOT = Path(__file__).resolve().parents[4]
EXAMPLES = ROOT / "examples" / "problems"

# Clear aliases between the demo's friendly names and the actual repo classes.
NEURAL_GOD_IMPL = "NeuralBuilder + FraisseNeuralPrior"      # the existing learned God
NEURAL_DEVIL_IMPL = "NeuralActiveDevil + NeuralDevilPolicy"  # the Gate R4 neural Devil


def build_neural_god(epochs: int = 40, seed: int = 0) -> NeuralBuilder:
    """Instantiate the existing learned Builder/God (supervised-pretrained)."""
    from .tasks import generate_tasks
    from .training import train_builder

    tmp = Path(tempfile.gettempdir()) / "logan_mirco_gan"
    tmp.mkdir(parents=True, exist_ok=True)
    pt = tmp / "mirco_god.pt"
    train_tasks = generate_tasks([3, 4], 30, seed=seed)
    train_builder(train_tasks, tmp / "mirco_god.jsonl", pt, epochs=epochs, seed=seed)
    return NeuralBuilder(FraisseNeuralPrior(str(pt)))


def _fmt_move(move) -> str:
    asg = "" if not move.assignment else " " + ",".join(f"{k}={v}" for k, v in move.assignment.items())
    return f"{move.kind} clause={move.clause_name}{asg} target={_fmt_cell(move.target_cell)}"


def run_demo(label: str, problem_path: Path, devil: NeuralActiveDevil,
             god: NeuralBuilder) -> dict:
    problem = load_problem(problem_path)
    task = build_task(problem)
    budget = int(problem["resources"]["budget"])
    out = play_game(task, devil, god, budget)

    certificate = None
    if out.outcome == "won":
        outcome = "GOD_WINS"
    elif out.outcome == "lost":
        cert = build_obstruction_certificate(task, hint=problem.get("obstruction_hint"))
        cert["budget_exhausted"] = False
        outcome, certificate = ("DEVIL_WINS", cert) if cert.get("judge_verified") else ("DRAW", None)
    else:
        outcome = "DRAW"
    return {"label": label, "problem": problem, "task": task, "out": out,
            "outcome": outcome, "certificate": certificate}


def render_demo(result: dict, devil: NeuralActiveDevil, god: NeuralBuilder) -> str:
    label = result["label"]
    problem = result["problem"]
    task = result["task"]
    out = result["out"]
    lines = [f"{label}: {problem.get('name')}", "-" * 52]
    lines.append(f"Assumptions: {', '.join(problem['theory'])}")
    g = problem["goal"]
    lines.append("Goal: refute " + g["claim"] if g["mode"] == "refute" else "Goal: build a model")
    lines.append(f"Players: NeuralDevil [{NEURAL_DEVIL_IMPL}]  vs  "
                 f"NeuralGod [{NEURAL_GOD_IMPL}]  |  Judge [symbolic run_devil]")
    lines.append("")

    # Opening exchange from the empty structure: makes the three roles visible for
    # both demos (the neural Devil picks a legal challenge; the neural God ranks the
    # legal repair replies; the Judge classifies the resulting position).
    state = GameState(task_empty_structure(task), task)
    moves = enumerate_legal_devil_moves(state)
    if moves:
        chosen = devil.policy.choose(state, moves)
        lines.append(f"NeuralDevil: selected challenge  {_fmt_move(chosen)}")
        lines.append(f"             (chosen among {len(moves)} legal symbolic challenges)")
        replies = legal_replies(chosen)
        ordered = god.order(state.structure, chosen, replies)
        if ordered:
            lines.append(f"NeuralGod: selected reply  {_fmt_edit(ordered[0])}")
            lines.append(f"           (ranked top of {len(ordered)} legal repair replies)")
        lines.append(f"Judge: opening position is '{judge(state.structure, task)}'")
    lines.append("")

    # Full verified line for a God win.
    if result["outcome"] == "GOD_WINS" and out.decisions:
        lines.append("Verified winning line (NeuralDevil challenges, NeuralGod repairs):")
        for i, (struct, move, edit) in enumerate(out.decisions, 1):
            lines.append(f"  Round {i}")
            lines.append(f"    NeuralDevil: {_fmt_move(move)}")
            lines.append(f"    NeuralGod:   {_fmt_edit(edit)}")
            lines.append(f"    Judge: progress")

    lines.append("")
    lines.append(f"Outcome: {result['outcome']}")
    if result["outcome"] == "GOD_WINS":
        lines.append(f"  Judge: NeuralGod survived the NeuralDevil and met the goal "
                     f"({len(out.decisions)} rounds, {out.nodes} nodes).")
    elif result["outcome"] == "DEVIL_WINS":
        c = result["certificate"] or {}
        lines.append("  Judge: no legal NeuralGod completion meets the goal within the bounds.")
        lines.append("Obstruction certificate:")
        lines.append(f"  domain_size: {c.get('domain_size')}")
        if c.get("reason") is not None:
            lines.append(f"  reason: {c['reason']}")
        if c.get("consequence") is not None:
            lines.append(f"  consequence: {c['consequence']}")
        lines.append(f"  budget_exhausted: {str(bool(c.get('budget_exhausted', False))).lower()}")
        lines.append(f"  judge_verified: {str(bool(c.get('judge_verified', False))).lower()}")
    return "\n".join(lines)


def task_empty_structure(task):
    from ..core.partial_structure import PartialStructure
    return PartialStructure.empty(task.signature, task.domain_size)


def run(epochs: int = 40, seed: int = 0) -> str:
    policy = NeuralDevilPolicy(seed=seed)
    devil = NeuralActiveDevil(policy)
    god = build_neural_god(epochs=epochs, seed=seed)

    header = ["LOGAN -- minimal Mirco GAN prototype demo",
              "(neural Devil + neural God + symbolic Judge; not full adversarial training)",
              "=" * 60, ""]
    a = render_demo(run_demo("Demo A", EXAMPLES / "cycle3_fight.json", devil, god), devil, god)
    b = render_demo(run_demo("Demo B", EXAMPLES / "cycle2_impossible_fight.json", devil, god), devil, god)
    return "\n".join(header) + a + "\n\n" + b + "\n"


def main(argv=None) -> int:
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
