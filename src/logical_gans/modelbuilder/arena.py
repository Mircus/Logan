"""The LOGAN arena: the central engine.

    Σ, T, goal, resources  ->  GOD_WINS / DEVIL_WINS / DRAW

The Builder (neural-guided MCTS over generic semantic edits, by default) plays
against the bounded Devil verifier. God = the constructive Builder; God wins by
producing a finite structure A the Devil verifies as achieving the goal under the
declared resources (n, depth k, Devil budget b, search budget).

Outcome contract (strong):
    GOD_WINS    Builder produced A with, at (k,b):
                  refute:  A ⊩ T and A ⊭ C
                  satisfy: A ⊩ T and A ⊩ P
    DEVIL_WINS  genuine bounded exhaustion / certified impossibility ONLY.
                Not produced in the P0 gate (left unsupported rather than weakened).
    DRAW        budget/space exhausted without a verified Builder win.

The neural builder is the LOGAN default; symbolic priors are explicit *baselines*
(``--builder uniform_baseline`` / ``obligation_baseline``). There is NO silent
fallback to symbolic search, and NO silent arity truncation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .capsule import _formula_to_clause
from .core.loader import parse_clause, parse_signature
from .core.theory import Claim, Theory
from .learned.semantic_features import MAX_ARITY
from .learned.semantic_search import build_training_examples, tree_policy_search

_BASELINES = {"uniform_baseline", "uniform", "obligation_baseline", "obligation_first"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass
class ArenaResult:
    outcome: str                       # GOD_WINS | DEVIL_WINS | DRAW
    mode: str
    resources: dict
    builder: dict
    structure: Optional[dict] = None
    witness: Optional[dict] = None
    model_relation: Optional[str] = None
    counterclaim_relation: Optional[str] = None
    goal_relation: Optional[str] = None
    reason: Optional[str] = None

    def to_json(self) -> dict:
        if self.outcome == "GOD_WINS":
            d = {"outcome": "GOD_WINS", "mode": self.mode}
            if self.mode == "refute":
                d["model_relation"] = self.model_relation
                d["counterclaim_relation"] = self.counterclaim_relation
            else:
                d["model_relation"] = self.model_relation
                d["goal_relation"] = self.goal_relation
            d.update({"resources": self.resources, "builder": self.builder,
                      "structure": self.structure, "witness": self.witness,
                      "status": "accepted"})
            return d
        # DRAW / DEVIL_WINS share the no-structure shape.
        return {"outcome": self.outcome, "mode": self.mode, "reason": self.reason,
                "resources": self.resources, "builder": self.builder,
                "structure": None, "witness": None}


def _build_goal(problem, signature):
    goal = problem["goal"]
    mode = goal["mode"]
    if mode == "refute":
        clause = parse_clause(_formula_to_clause(goal["claim"], "claim", signature))
        return mode, Claim(name=problem.get("name", "claim"), clauses=[clause])
    if mode == "satisfy":
        props = goal.get("properties", [])
        clauses = [parse_clause(_formula_to_clause(f, f"goal_{i}", signature))
                   for i, f in enumerate(props)]
        return mode, Claim(name=problem.get("name", "goal"), clauses=clauses)
    raise ValueError(f"unknown goal.mode {mode!r} (expected 'refute' or 'satisfy')")


def _resolve_prior(kind, mode, theory, goal_obj, n, k, b, epochs, name):
    """Return (prior, resolved_kind, uses_neural). No silent symbolic fallback."""
    if kind == "neural_mcts":
        max_arity = max([0]
                        + [s.arity for s in theory.signature.relations.values()]
                        + [s.arity for s in theory.signature.functions.values()])
        if max_arity > MAX_ARITY:
            raise ValueError(
                f"neural builder supports arity <= {MAX_ARITY}; this signature has arity "
                f"{max_arity}. The neural feature model is not yet that-arity-parametric, and "
                f"silent truncation is not allowed. Use --builder uniform_baseline (a named "
                f"baseline) or reduce arity.")
        if mode != "refute":
            raise ValueError("neural_mcts auto-train currently supports mode=refute only; "
                             "use --builder uniform_baseline for satisfy mode.")
        try:
            import torch  # noqa: F401
        except Exception:
            raise RuntimeError("Neural builder unavailable (PyTorch not installed). "
                               "Use --builder uniform_baseline explicitly for a baseline.")
        from .learned.priors import NeuralSemanticPrior
        from .learned.semantic_training import (
            train_semantic_policy,
            write_semantic_training_jsonl,
        )

        examples = build_training_examples(theory, goal_obj, n, k=k, b=b)
        if not examples:
            raise RuntimeError(
                "auto-train found no oracle refuting traces for this problem, so the small "
                "problem-specific neural prior cannot be trained. No silent fallback to symbolic "
                "search. Use --builder uniform_baseline, or adjust the problem/resources.")
        root = _repo_root()
        data = root / "results" / "training" / f"{name}_arena.jsonl"
        model = root / "models" / f"{name}_arena.pt"
        write_semantic_training_jsonl(examples, data)
        train_semantic_policy(str(data), str(model), epochs=epochs, seed=0)
        return NeuralSemanticPrior(str(model)), "neural_mcts", True

    if kind in ("uniform_baseline", "uniform"):
        from .learned.priors import UniformPrior
        return UniformPrior(), "uniform_baseline", False
    if kind in ("obligation_baseline", "obligation_first"):
        from .learned.priors import ObligationFirstPrior
        return ObligationFirstPrior(), "obligation_baseline", False
    raise ValueError(f"unknown builder kind {kind!r}")


def solve_arena(problem: dict, *, builder_override: Optional[str] = None,
                epochs: int = 150) -> ArenaResult:
    signature = parse_signature(problem["signature"])
    theory = Theory(name=problem.get("name", "theory"), signature=signature,
                    clauses=[parse_clause(_formula_to_clause(f, f"axiom_{i}", signature))
                             for i, f in enumerate(problem["theory"])])
    mode, goal_obj = _build_goal(problem, signature)

    r = problem["resources"]
    n, k, b, sb = (int(r["domain_size"]), int(r["depth"]),
                   int(r["devil_budget"]), int(r["search_budget"]))
    resources = {"domain_size": n, "depth": k, "devil_budget": b, "search_budget": sb}

    builder_cfg = problem.get("builder", {})
    # default LOGAN builder is neural_mcts (NOT symbolic); explicit override wins.
    kind = builder_override or builder_cfg.get("kind", "neural_mcts")
    name = problem.get("name", "problem")

    prior, kind, uses_neural = _resolve_prior(
        kind, mode, theory, goal_obj, n, k, b, epochs, name)

    res = tree_policy_search(theory, goal_obj, n, prior, mode=mode, k=k,
                             devil_budget=b, search_budget=sb, c_puct=2.0)
    builder = {"kind": kind, "uses_neural_policy": uses_neural,
               "success_via": res["success_via"]}

    if res["success"]:
        model_relation = f"A ⊩_{{{k},{b}}} T"
        if mode == "refute":
            return ArenaResult("GOD_WINS", mode, resources, builder,
                               structure=res["structure"], witness=res["witness"],
                               model_relation=model_relation,
                               counterclaim_relation=f"A ⊭_{{{k},{b}}} C")
        return ArenaResult("GOD_WINS", mode, resources, builder,
                           structure=res["structure"], witness=res["witness"],
                           model_relation=model_relation,
                           goal_relation=f"A ⊩_{{{k},{b}}} P")

    # No verified Builder win. DEVIL_WINS is intentionally NOT claimed in P0
    # (it would require a certified bounded-impossibility proof); report DRAW.
    reason = "search_space_exhausted" if res.get("exhausted") else "search_budget_exhausted"
    return ArenaResult("DRAW", mode, resources, builder, reason=reason)
