"""End-to-end demo of the neural relation Builder on preorders (n=3).

1. mine training data from Devil witnesses
2. train the relation policy
3. run the neural Builder (Devil-verified)
4. independently verify the final structure with the exhaustive Devil
5. save results/neural_preorder_demo.json

Run:  PYTHONPATH=src python experiments/neural_preorder_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_theory
from logical_gans.modelbuilder.learned.builder import neural_relation_build
from logical_gans.modelbuilder.learned.data import (
    make_relation_training_examples,
    write_training_examples_jsonl,
)
from logical_gans.modelbuilder.learned.train import train_relation_policy

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "examples" / "theories" / "preorder.json"
DATA = ROOT / "results" / "training" / "preorder_R_n3.jsonl"
MODEL = ROOT / "models" / "preorder_R_n3.pt"
OUT = ROOT / "results" / "neural_preorder_demo.json"

RELATION = "R"
N = 3


def main() -> int:
    theory = load_theory(THEORY)

    # 1. training data
    examples = make_relation_training_examples(theory, RELATION, N, num_samples=800, seed=0)
    write_training_examples_jsonl(examples, DATA)

    # 2. train
    train_info = train_relation_policy(str(DATA), str(MODEL), N, epochs=300, lr=1e-3, seed=0)

    # 3. neural build (unbounded Devil -> the policy must construct a COMPLETE model)
    result = neural_relation_build(theory, RELATION, N, str(MODEL), max_steps=50)

    # 4. independent exhaustive verification of the final structure
    from logical_gans.modelbuilder.core.partial_structure import PartialStructure
    from logical_gans.modelbuilder.core.types import Truth

    final = PartialStructure.empty(theory.signature, N)
    for key, val in result["structure"]["relations"].items():
        # key like "R(0,1)"
        inside = key[key.index("(") + 1: key.index(")")]
        i, j = (int(x) for x in inside.split(","))
        final.set_relation(RELATION, (i, j), Truth(val))
    devil_check = run_devil(final, theory.clauses).status  # "ok" iff a real model

    out = {
        "theory": theory.name,
        "relation": RELATION,
        "n": N,
        "builder": result["builder"],
        "status": result["status"],
        "exhaustive_devil": devil_check,
        "training_examples": train_info["examples"],
        "final_loss": train_info["final_loss"],
        "trace": result["trace"],
        "structure": result["structure"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"status={out['status']} exhaustive_devil={out['exhaustive_devil']} "
          f"builder={out['builder']} -> {OUT}")
    return 0 if out["status"] == "satisfied" else 1


if __name__ == "__main__":
    raise SystemExit(main())
