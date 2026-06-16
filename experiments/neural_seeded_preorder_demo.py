"""Seeded completion demo: the neural Builder must close a forced fact.

Seed (open-world):  R(0,1)=TRUE, R(1,2)=TRUE  (everything else UNKNOWN)
Transitivity forces: R(0,2)=TRUE

1. mine training data (closure-oracle labels)
2. train the relation policy
3. load the open-world seed
4. run the neural Builder (Devil-verified)
5. verify the final structure with the exhaustive Devil
6. save results/neural_seeded_preorder_demo.json

Run:  PYTHONPATH=src python experiments/neural_seeded_preorder_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_seed_open_world, load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.builder import neural_relation_build
from logical_gans.modelbuilder.learned.data import (
    make_relation_training_examples,
    write_training_examples_jsonl,
)
from logical_gans.modelbuilder.learned.train import train_relation_policy

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "examples" / "theories" / "preorder.json"
SEED = ROOT / "examples" / "seeds" / "preorder_chain_3.json"
DATA = ROOT / "results" / "training" / "preorder_R_n3.jsonl"
MODEL = ROOT / "models" / "preorder_R_n3.pt"
OUT = ROOT / "results" / "neural_seeded_preorder_demo.json"

RELATION = "R"
N = 3
FORCED = (0, 2)


def main() -> int:
    theory = load_theory(THEORY)

    # 1-2. data + train
    examples = make_relation_training_examples(theory, RELATION, N, num_samples=800, seed=0)
    write_training_examples_jsonl(examples, DATA)
    train_relation_policy(str(DATA), str(MODEL), N, epochs=300, lr=1e-3, seed=0)

    # 3-4. load seed and build
    seed = load_seed_open_world(SEED, theory.signature)
    result = neural_relation_build(theory, RELATION, N, str(MODEL),
                                   seed_structure=seed, max_steps=50)

    rels = result["structure"]["relations"]

    # 5. independent exhaustive verification
    final = PartialStructure.empty(theory.signature, N)
    for key, val in rels.items():
        inside = key[key.index("(") + 1: key.index(")")]
        i, j = (int(x) for x in inside.split(","))
        final.set_relation(RELATION, (i, j), Truth(val))
    exhaustive = run_devil(final, theory.clauses).status

    forced_key = f"R({FORCED[0]},{FORCED[1]})"
    out = {
        "theory": theory.name,
        "relation": RELATION,
        "n": N,
        "builder": result["builder"],
        "status": result["status"],
        "exhaustive_devil": exhaustive,
        "forced_fact": forced_key,
        "forced_fact_present": rels.get(forced_key) == "true",
        "seed_facts_preserved": rels.get("R(0,1)") == "true" and rels.get("R(1,2)") == "true",
        "trace": result["trace"],
        "structure": result["structure"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"status={out['status']} exhaustive_devil={out['exhaustive_devil']} "
          f"{forced_key}={'true' if out['forced_fact_present'] else 'NOT-true'} "
          f"seed_preserved={out['seed_facts_preserved']} -> {OUT}")
    ok = (out["status"] == "satisfied" and out["exhaustive_devil"] == "ok"
          and out["forced_fact_present"] and out["seed_facts_preserved"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
