"""MCTS seeded-completion demo: neural priors vs uniform priors.

Seed R(0,1)=TRUE, R(1,2)=TRUE -> transitivity forces R(0,2)=TRUE.
MCTS (Devil-verified) must complete a valid preorder; with neural priors it
should explore fewer nodes than with uniform priors.

Run:  PYTHONPATH=src python experiments/neural_mcts_preorder_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_seed_open_world, load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.data import (
    make_relation_training_examples,
    write_training_examples_jsonl,
)
from logical_gans.modelbuilder.learned.mcts import mcts_relation_build
from logical_gans.modelbuilder.learned.train import train_relation_policy

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "examples" / "theories" / "preorder.json"
SEED = ROOT / "examples" / "seeds" / "preorder_chain_3.json"
DATA = ROOT / "results" / "training" / "preorder_R_n3.jsonl"
MODEL = ROOT / "models" / "preorder_R_n3.pt"
OUT = ROOT / "results" / "neural_mcts_preorder_demo.json"

RELATION, N = "R", 3


def _summary(theory, result):
    rels = result["structure"]["relations"]
    final = PartialStructure.empty(theory.signature, N)
    for key, val in rels.items():
        inside = key[key.index("(") + 1: key.index(")")]
        i, j = (int(x) for x in inside.split(","))
        final.set_relation(RELATION, (i, j), Truth(val))
    return {
        "status": result["status"],
        "uses_neural_policy": result["uses_neural_policy"],
        "nodes": result["nodes"],
        "forced_fact_present": rels.get("R(0,2)") == "true",
        "seed_facts_preserved": rels.get("R(0,1)") == "true" and rels.get("R(1,2)") == "true",
        "exhaustive_devil": run_devil(final, theory.clauses).status,
    }


def main(samples: int = 600, epochs: int = 200, rollouts: int = 100) -> int:
    theory = load_theory(THEORY)

    examples = make_relation_training_examples(theory, RELATION, N, num_samples=samples, seed=0)
    write_training_examples_jsonl(examples, DATA)
    train_relation_policy(str(DATA), str(MODEL), N, epochs=epochs, lr=1e-3, seed=0)

    seed = load_seed_open_world(SEED, theory.signature)
    neural = mcts_relation_build(theory, RELATION, N, model_path=str(MODEL),
                                 seed_structure=seed, rollouts=rollouts)
    uniform = mcts_relation_build(theory, RELATION, N, model_path=None,
                                  seed_structure=seed, rollouts=rollouts)

    out = {
        "theory": theory.name, "relation": RELATION, "n": N, "rollouts": rollouts,
        "neural": _summary(theory, neural),
        "uniform": _summary(theory, uniform),
        "neural_trace": neural["trace"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    nn, un = out["neural"], out["uniform"]
    print(f"neural:  status={nn['status']} nodes={nn['nodes']} R(0,2)={'true' if nn['forced_fact_present'] else 'NO'} "
          f"exhaustive={nn['exhaustive_devil']}")
    print(f"uniform: status={un['status']} nodes={un['nodes']} R(0,2)={'true' if un['forced_fact_present'] else 'NO'} "
          f"exhaustive={un['exhaustive_devil']}  -> {OUT}")
    ok = (nn["status"] == "satisfied" and nn["forced_fact_present"]
          and nn["seed_facts_preserved"] and nn["exhaustive_devil"] == "ok")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
