"""Gate R8: minimal judged God/Devil co-training demo.

The smallest honest step from "trained Devil vs fixed God" (R6) to "trained Devil
+ trained God under the symbolic Judge". Both neural players are updated by
REINFORCE; the symbolic Judge decides every outcome.

This is NOT full ADAMANTIUM, NOT a production GAN. It is a tiny judged co-training
demo on the two existing cyclic tasks only.

Per the Gate R7 audit (Path B), the trainable God is a NEW, clearly-named policy
(`NeuralGodPolicy`) over the existing `legal_replies` enumerator -- it is NOT the
frozen `FraisseNeuralPrior` (which remains the default-fight God, untouched).

Loop (per episode):
    GameState -> enumerate_legal_devil_moves -> NeuralDevilPolicy samples challenge
              -> legal_replies -> NeuralGodPolicy samples reply
              -> SymbolicJudge (play_game) decides outcome (+certificate)
              -> God/Devil rewards -> update BOTH policies.

Same honest caveat as R6: on these tiny fixed tasks the Judge's outcome is
task-determined (cycle3 winnable, cycle2 impossible), so neither player can flip
the outcome here; the loop still produces real reward signals and real parameter
updates for BOTH policies.
"""
from __future__ import annotations

import torch
from torch.distributions import Categorical

from .adversarial_train_demo import REWARDS, rewards_for
from .certificate import build_obstruction_certificate
from .fight import _fmt_edit, build_task, load_problem
from .game import GameState, legal_replies, play_game
from .mirco_gan_demo import EXAMPLES, _fmt_move, task_empty_structure
from .neural_devil import NeuralActiveDevil, NeuralDevilPolicy
from .neural_god import NeuralGodBuilder, NeuralGodPolicy
from .players import enumerate_legal_devil_moves

TASKS = [
    ("Demo A", EXAMPLES / "cycle3_fight.json", "GOD_WINS"),
    ("Demo B", EXAMPLES / "cycle2_impossible_fight.json", "DEVIL_WINS"),
]


def judged_outcome(task, problem, devil_policy, god_policy, budget):
    """Play the game under the symbolic Judge with both neural policies."""
    out = play_game(task, NeuralActiveDevil(devil_policy), NeuralGodBuilder(god_policy), budget)
    if out.outcome == "won":
        return "GOD_WINS", None
    if out.outcome == "lost":
        cert = build_obstruction_certificate(task, hint=problem.get("obstruction_hint"))
        cert["budget_exhausted"] = False
        return ("DEVIL_WINS", cert) if cert.get("judge_verified") else ("DRAW", None)
    return "DRAW", None


def cotrain_episode(devil_policy, god_policy, opt_d, opt_g, label, path) -> dict:
    """One co-training episode: sample challenge + reply, judge, update BOTH policies."""
    problem = load_problem(path)
    task = build_task(problem)
    budget = int(problem["resources"]["budget"])
    structure = task_empty_structure(task)
    state = GameState(structure, task)

    # Devil samples a legal challenge (differentiable).
    moves = enumerate_legal_devil_moves(state)
    sd = devil_policy.score_moves(state, moves)
    dist_d = Categorical(logits=sd)
    idx_d = dist_d.sample()
    log_prob_d = dist_d.log_prob(idx_d)
    challenge = moves[int(idx_d.item())]

    # God samples a legal reply to that challenge (differentiable).
    replies = legal_replies(challenge)
    sg = god_policy.score_replies(structure, challenge, replies)
    dist_g = Categorical(logits=sg)
    idx_g = dist_g.sample()
    log_prob_g = dist_g.log_prob(idx_g)
    reply = replies[int(idx_g.item())]

    # Symbolic Judge decides the outcome (+ certificate on DEVIL_WINS).
    outcome, certificate = judged_outcome(task, problem, devil_policy, god_policy, budget)
    god_r, devil_r = rewards_for(outcome)

    # Update BOTH policies (REINFORCE on the sampled actions). No no_grad here.
    god_loss = -god_r * log_prob_g
    devil_loss = -devil_r * log_prob_d
    opt_g.zero_grad(); god_loss.backward(); opt_g.step()
    opt_d.zero_grad(); devil_loss.backward(); opt_d.step()

    return {
        "label": label, "task": problem.get("name"), "challenge": challenge, "reply": reply,
        "outcome": outcome, "certificate": certificate, "god_reward": god_r, "devil_reward": devil_r,
        "god_loss": float(god_loss.item()), "devil_loss": float(devil_loss.item()),
    }


def run(epochs: int = 3, seed: int = 0, lr: float = 0.1) -> str:
    torch.manual_seed(seed)
    devil_policy = NeuralDevilPolicy(seed=seed)
    god_policy = NeuralGodPolicy(seed=seed)
    opt_d = torch.optim.SGD(devil_policy.parameters(), lr=lr)
    opt_g = torch.optim.SGD(god_policy.parameters(), lr=lr)

    counts = {"GOD_WINS": 0, "DEVIL_WINS": 0, "DRAW": 0}
    lines = [
        "LOGAN -- minimal judged God/Devil co-training demo (Gate R8)",
        "(minimal judged co-training; NOT a full GAN / production trainer)",
        "Both NeuralDevil and NeuralGod (NeuralGodPolicy, a new trainable policy) are trained.",
        "=" * 62, "",
    ]
    for epoch in range(epochs):
        for episode, (label, path, _expected) in enumerate(TASKS):
            m = cotrain_episode(devil_policy, god_policy, opt_d, opt_g, label, path)
            counts[m["outcome"]] += 1
            lines.append(f"epoch {epoch}  episode {episode}  task {m['task']}  ({label})")
            lines.append(f"  NeuralDevil selected challenge: {_fmt_move(m['challenge'])}")
            lines.append(f"  NeuralGod selected reply: {_fmt_edit(m['reply'])}")
            lines.append(f"  Judge outcome: {m['outcome']}")
            lines.append(f"  God reward: {m['god_reward']:+.2f}   Devil reward: {m['devil_reward']:+.2f}")
            lines.append(f"  God loss: {m['god_loss']:.4f}   Devil loss: {m['devil_loss']:.4f}")
            if m["outcome"] == "DEVIL_WINS" and m["certificate"]:
                jv = str(bool(m["certificate"].get("judge_verified"))).lower()
                be = str(bool(m["certificate"].get("budget_exhausted"))).lower()
                lines.append(f"  Obstruction certificate: judge_verified={jv}, budget_exhausted={be}")
            lines.append("")

    lines.append("Totals:")
    lines.append(f"GOD_WINS count: {counts['GOD_WINS']}")
    lines.append(f"DEVIL_WINS count: {counts['DEVIL_WINS']}")
    lines.append(f"DRAW count: {counts['DRAW']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    print(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
