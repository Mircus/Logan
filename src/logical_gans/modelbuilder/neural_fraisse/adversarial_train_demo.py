"""Gate R6: minimal God/Devil adversarial training loop under symbolic judgment.

This is a *minimal adversarial training loop*, not a full GAN, not a production
trainer, not a theorem prover. One bounded policy-gradient (REINFORCE) update of
the neural Devil per episode, with rewards derived from the symbolic Judge's
outcome.

Loop shape (per episode):

    GameState
      -> enumerate_legal_devil_moves(state)          (legal symbolic challenges)
      -> NeuralDevilPolicy samples a challenge        (differentiable log-prob)
      -> NeuralBuilder/God ranks legal repair replies (existing learned God)
      -> SymbolicJudge plays out + verifies outcome   (GOD_WINS/DEVIL_WINS/DRAW)
      -> bounded rewards from the outcome
      -> update neural Devil parameters

HONESTY NOTE (Gate R6): only the neural Devil is trained. The neural God
(NeuralBuilder + FraisseNeuralPrior) is the existing supervised-pretrained,
checkpoint-loaded, eval/no_grad model; updating it cleanly is invasive and is
deferred. The trace prints "NeuralGod fixed in Gate R6; Devil trained."

On these tiny fixed tasks the Judge's outcome is task-determined (cycle3 is
winnable, cycle2 is impossible), so the Devil cannot flip the outcome here; the
loop still produces a real bounded reward signal and a real parameter update.
"""
from __future__ import annotations

import torch
from torch.distributions import Categorical

from .fight import _fmt_edit, build_task, load_problem
from .game import GameState, legal_replies
from .mirco_gan_demo import (EXAMPLES, _fmt_move, build_neural_god, run_demo,
                             task_empty_structure)
from .neural_devil import NeuralActiveDevil, NeuralDevilPolicy
from .players import enumerate_legal_devil_moves

# (God reward, Devil reward) per outcome -- simple bounded scheme.
REWARDS = {
    "GOD_WINS": (1.0, -1.0),
    "DEVIL_WINS": (-1.0, 1.0),
    "DRAW": (-0.2, -0.2),
}

TASKS = [
    ("Demo A", EXAMPLES / "cycle3_fight.json", "GOD_WINS"),
    ("Demo B", EXAMPLES / "cycle2_impossible_fight.json", "DEVIL_WINS"),
]


def rewards_for(outcome: str):
    return REWARDS[outcome]


def train_episode(policy: NeuralDevilPolicy, optimizer, god, label, path) -> dict:
    """One adversarial episode + one Devil policy-gradient update."""
    problem = load_problem(path)
    task = build_task(problem)

    # --- Devil: sample a legal challenge (differentiable) -----------------
    state = GameState(task_empty_structure(task), task)
    moves = enumerate_legal_devil_moves(state)
    scores = policy.score_moves(state, moves)
    dist = Categorical(logits=scores)
    idx = dist.sample()
    log_prob = dist.log_prob(idx)
    chosen = moves[int(idx.item())]

    # --- God: rank legal repair replies for that challenge (fixed) --------
    replies = legal_replies(chosen)
    ranked = god.order(state.structure, chosen, replies)
    god_reply = ranked[0] if ranked else None

    # --- Judge: play out the game, get the verified outcome ---------------
    result = run_demo(label, path, NeuralActiveDevil(policy), god)
    outcome = result["outcome"]
    certificate = result["certificate"]

    god_r, devil_r = rewards_for(outcome)

    # --- update neural Devil (REINFORCE on the sampled challenge) ----------
    devil_loss = -devil_r * log_prob
    optimizer.zero_grad()
    devil_loss.backward()
    optimizer.step()

    return {
        "label": label, "task": problem.get("name"), "chosen": chosen,
        "god_reply": god_reply, "outcome": outcome, "certificate": certificate,
        "god_reward": god_r, "devil_reward": devil_r,
        "devil_loss": float(devil_loss.item()),
    }


def run(epochs: int = 3, seed: int = 0, god_epochs: int = 12, lr: float = 0.1) -> str:
    torch.manual_seed(seed)
    god = build_neural_god(epochs=god_epochs, seed=seed)
    policy = NeuralDevilPolicy(seed=seed)
    optimizer = torch.optim.SGD(policy.parameters(), lr=lr)

    counts = {"GOD_WINS": 0, "DEVIL_WINS": 0, "DRAW": 0}
    lines = [
        "LOGAN -- minimal God/Devil adversarial training loop (Gate R6)",
        "(minimal adversarial training loop; NOT a full GAN / production trainer)",
        "NeuralGod fixed in Gate R6; Devil trained.",
        "=" * 62, "",
    ]

    for epoch in range(epochs):
        for episode, (label, path, _expected) in enumerate(TASKS):
            m = train_episode(policy, optimizer, god, label, path)
            counts[m["outcome"]] += 1
            lines.append(f"epoch {epoch}  episode {episode}  task {m['task']}  ({label})")
            lines.append(f"  NeuralDevil selected challenge: {_fmt_move(m['chosen'])}")
            reply = _fmt_edit(m["god_reply"]) if m["god_reply"] is not None else "(none)"
            lines.append(f"  NeuralGod selected reply: {reply}")
            lines.append(f"  Judge outcome: {m['outcome']}")
            lines.append(f"  God reward: {m['god_reward']:+.2f}   Devil reward: {m['devil_reward']:+.2f}")
            lines.append(f"  God loss: n/a (NeuralGod fixed)   Devil loss: {m['devil_loss']:.4f}")
            if m["outcome"] == "DEVIL_WINS" and m["certificate"]:
                jv = str(bool(m["certificate"].get("judge_verified"))).lower()
                lines.append(f"  Obstruction certificate: judge_verified={jv}, "
                             f"budget_exhausted={str(bool(m['certificate'].get('budget_exhausted'))).lower()}")
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
