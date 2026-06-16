"""The LOGAN proof: neural-guided MCTS generates a countermodel.

Signature: E/2, s/1, a.  Theory: E(x,s(x)); s(s(x))=x; E(a,a).
Claim to refute: forall x E(x,x).

Pipeline:
  1. mine semantic-policy training data from generic refute traces (oracle)
  2. train SemanticPolicyNet
  3. run neural-guided MCTS countermodel search
  4. verify theory exhaustively (symbolic Devil)
  5. verify claim failure + witness (symbolic Devil)
  6. save results/neural_semantic_countermodel_demo.json

Run:  PYTHONPATH=src python experiments/neural_semantic_countermodel_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_claim, load_theory
from logical_gans.modelbuilder.learned.generic_mcts import mcts_semantic_refute
from logical_gans.modelbuilder.learned.priors import NeuralSemanticPrior
from logical_gans.modelbuilder.learned.semantic_training import (
    make_semantic_training_examples,
    train_semantic_policy,
    write_semantic_training_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
THEORY = ROOT / "examples" / "theories" / "toy_mixed_signature.json"
CLAIM = ROOT / "examples" / "claims" / "toy_reflexive_E.json"
DATA = ROOT / "results" / "training" / "toy_mixed_semantic.jsonl"
MODEL = ROOT / "models" / "toy_mixed_semantic_policy.pt"
OUT = ROOT / "results" / "neural_semantic_countermodel_demo.json"
N = 2


def main(samples: int = 150, epochs: int = 60, rollouts: int = 500) -> int:
    theory = load_theory(THEORY)
    claim = load_claim(CLAIM)

    # 1-2. training data (oracle traces) + train neural policy
    examples = make_semantic_training_examples(theory, claim, N, samples, seed=0)
    write_semantic_training_jsonl(examples, DATA)
    train_semantic_policy(str(DATA), str(MODEL), epochs=epochs, seed=0)

    # 3. neural-guided MCTS countermodel search
    prior = NeuralSemanticPrior(str(MODEL))
    res = mcts_semantic_refute(theory, claim, N, prior_provider=prior, rollouts=rollouts, seed=0)

    # 4-5. independent symbolic verification of the generated structure
    from logical_gans.modelbuilder.core.partial_structure import PartialStructure
    from logical_gans.modelbuilder.core.types import Truth
    A = PartialStructure.empty(theory.signature, N)
    sj = res["structure"]
    for key, val in sj["relations"].items():
        inside = key[key.index("(") + 1: key.index(")")]
        A.set_relation(key[: key.index("(")], tuple(int(x) for x in inside.split(",")), Truth(val))
    for key, val in sj["functions"].items():
        if val is not None:
            inside = key[key.index("(") + 1: key.index(")")]
            A.set_function(key[: key.index("(")], tuple(int(x) for x in inside.split(",")), val)
    for name, val in sj["constants"].items():
        if val is not None:
            A.set_constant(name, val)
    theory_ok = run_devil(A, theory.clauses).status == "ok"
    claim_devil = run_devil(A, claim.clauses)

    out = {
        "status": res["status"],
        "builder": res["builder"],
        "uses_neural_policy": res["uses_neural_policy"],
        "theory_status": "satisfied" if theory_ok else "unsatisfied",
        "claim_status": "failed" if claim_devil.status == "failed" else claim_devil.status,
        "structure": sj,
        "claim_witness": claim_devil.witness.to_json() if claim_devil.witness else None,
        "nodes": res["nodes"],
        "trace": res["trace"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"status={out['status']} uses_neural_policy={out['uses_neural_policy']} "
          f"theory={out['theory_status']} claim={out['claim_status']} "
          f"witness={out['claim_witness']['assignment'] if out['claim_witness'] else None} -> {OUT}")
    ok = (out["status"] == "refuted" and out["uses_neural_policy"]
          and out["theory_status"] == "satisfied" and out["claim_status"] == "failed"
          and out["claim_witness"] is not None)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
