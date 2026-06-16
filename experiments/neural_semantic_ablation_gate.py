"""Gate 2 -- neural semantic ablation on a mixed-signature countermodel.

This is a DEMO/EVIDENCE script, not the tool. It uses the package engine
(``logical_gans.modelbuilder.learned.semantic_search``) -- it no longer defines
its own search, and nothing in the tool imports this file.

    Σ = { E/2, s/1, a }
    T = { ∀x E(x,s(x)) ; ∀x s(s(s(x)))=x }
    C = { s(a)=a }, n = 3

Tree-policy-only MCTS (random rollouts disabled); the trained neural prior is
compared against uniform / obligation-first under identical budget/seed/Devil.

Run:    PYTHONPATH=src python experiments/neural_semantic_ablation_gate.py
Output: results/neural_semantic_ablation_gate.json
"""
from __future__ import annotations

import json
from pathlib import Path

from logical_gans.modelbuilder.core.loader import load_claim, load_theory
from logical_gans.modelbuilder.learned.priors import (
    NeuralSemanticPrior,
    ObligationFirstPrior,
    UniformPrior,
)
from logical_gans.modelbuilder.learned.semantic_search import (
    build_training_examples,
    tree_policy_search,
)
from logical_gans.modelbuilder.learned.semantic_training import (
    train_semantic_policy,
    write_semantic_training_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "examples" / "theories" / "cycle3_succ.json"
CLAIM = ROOT / "examples" / "claims" / "succ_fixes_a.json"
DATA = ROOT / "results" / "training" / "cycle3_ablation_semantic.jsonl"
MODEL = ROOT / "models" / "cycle3_ablation_semantic_policy.pt"
OUT = ROOT / "results" / "neural_semantic_ablation_gate.json"

N = 3
BUDGET = 800
SEED = 0


def _structure_has_all_kinds(struct) -> bool:
    if not struct:
        return False
    return (any(v in ("true", "false") for v in struct["relations"].values())
            and any(v is not None for v in struct["functions"].values())
            and any(v is not None for v in struct["constants"].values()))


def _arm(theory, claim, prior):
    # exhaustive Devil (k=None, budget=None) reproduces the original ablation.
    return tree_policy_search(theory, claim, N, prior, mode="refute",
                              k=None, devil_budget=None, search_budget=BUDGET, c_puct=2.0)


def main(epochs: int = 150) -> int:
    theory = load_theory(THEORY)
    claim = load_claim(CLAIM)

    examples = build_training_examples(theory, claim, N)
    write_semantic_training_jsonl(examples, DATA)
    train_meta = train_semantic_policy(str(DATA), str(MODEL), epochs=epochs, seed=SEED)

    results = {
        "uniform": _arm(theory, claim, UniformPrior()),
        "obligation_first": _arm(theory, claim, ObligationFirstPrior()),
        "neural": _arm(theory, claim, NeuralSemanticPrior(str(MODEL))),
    }
    neural, uni, obl = results["neural"], results["uniform"], results["obligation_first"]

    def worse(base) -> bool:
        return (not base["success"]) or (
            neural["success"] and base["nodes_evaluated"] >= 3 * neural["nodes_evaluated"])

    neural_ok = (
        neural["success"]
        and neural["success_via"] == "guided_tree_policy"
        and neural["theory_status"] == "satisfied"
        and neural["goal_status"] == "failed"
        and neural["witness"] is not None
        and _structure_has_all_kinds(neural["structure"])
    )
    accepted = bool(neural_ok and worse(uni) and worse(obl))

    out = {
        "gate": "neural_semantic_ablation",
        "problem": {"signature": "E/2, s/1, a",
                    "theory": "forall x E(x,s(x)); forall x s(s(s(x)))=x",
                    "claim": "s(a)=a", "n": N},
        "protocol": {"budget_nodes": BUDGET, "seed": SEED, "random_rollouts": "disabled",
                     "training_examples": train_meta["examples"],
                     "training_final_loss": train_meta["final_loss"]},
        "arms": results,
        "accepted": accepted,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"neural: success={neural['success']} via={neural['success_via']} "
          f"nodes={neural['nodes_evaluated']} theory={neural['theory_status']} "
          f"claim={neural['goal_status']}")
    print(f"uniform: success={uni['success']} nodes={uni['nodes_evaluated']}")
    print(f"obligation_first: success={obl['success']} nodes={obl['nodes_evaluated']}")
    print(f"accepted={accepted} -> {OUT}")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
