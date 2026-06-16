# Arena alignment memo

Audit of what the LOGAN arena currently implements and how it maps to the
UCMT paper. Scope claim: this is a **P0 arena**, not full LOGAN-UCMT.

Branch: `signature-parametric-builder`. Verified at commit `bb1c503` (memo adds
itself on top).

## 1. What is the implemented LOGAN arena?

A single engine with the interface:

    Σ, T, goal, resources  →  GOD_WINS / DEVIL_WINS / DRAW

- **Σ**: a finite first-order signature (relations/functions/constants with arities).
- **T**: universal Horn/equational clauses over Σ.
- **goal**: `mode = refute` (claim C) or `mode = satisfy` (properties P).
- **resources**: domain size `n`, depth bound `k`, Devil budget `b`, search budget.
- **Builder** (default `neural_mcts`): neural-guided, tree-policy-only MCTS over
  generic semantic edits (`SetRelation/SetFunction/SetConstant`), no random rollouts.
- **Devil**: `run_devil_bounded(k, b)` verifies every candidate; returns
  ok / unknown / failed with a replayable witness.
- Entry points: `src/.../modelbuilder/arena.py` (`solve_arena`), CLI `arena-solve`.

Outcome contract (strong): GOD_WINS = Builder produced A the Devil verifies as
achieving the goal at (k,b); DRAW = no verified win within resources; DEVIL_WINS
is reserved for certified bounded impossibility and is **not** emitted in P0.

## 2. What command proves the current arena?

    PYTHONPATH=src python -m logical_gans.modelbuilder.cli arena-solve \
      examples/problems/cycle3_arena.json

Verified output (Σ = {E/2, s/1, a}; T = {∀x E(x,s(x)), ∀x s(s(s(x)))=x};
C = s(a)=a; n=3, k=3, b=800):

    outcome = GOD_WINS
    model_relation       = A ⊩_{3,800} T
    counterclaim_relation = A ⊭_{3,800} C
    builder.kind = neural_mcts, uses_neural_policy = true, success_via = guided_tree_policy
    structure: a=1, s(0)=2, s(1)=0, s(2)=1, E(0,2)=E(1,0)=E(2,1)=true
    witness: s(a)=s(1)=0 ≠ 1=a  (claim FAILED)

The tiny-budget twin returns the honest negative:

    arena-solve examples/problems/cycle3_arena_tiny_budget.json
    → outcome = DRAW, reason = search_budget_exhausted

## 3. What does this instantiate from the paper?

- **finite first-order signature** — `core/signature.py`.
- **partial finite Σ-structures** (cells true/false/unknown, function/constant or
  unknown) — `core/partial_structure.py`.
- **semantic edits** that fill unknown cells, derived from the signature —
  `learned/semantic_actions.py`.
- **bounded depth k** and **Devil budget b** — `core/depth.py`, `run_devil_bounded`.
- **Builder vs Devil** adversarial loop — `learned/semantic_search.py` + `arena.py`.
- **witness / certificate** — replayable `Witness` and the UCMT relations
  `A ⊩_{k,b} T`, `A ⊭_{k,b} C` emitted by the arena.

So the arena instantiates the P0 fragment of the UCMT picture: a bounded
Builder–Devil game over finite partial structures, producing a checkable
countermodel certificate.

## 4. What is still missing?

- **DEVIL_WINS** is not implemented except as future *certified bounded
  impossibility*; P0 reports DRAW rather than weakening the meaning.
- The **neural policy is not fully arity-parametric**: `semantic_features.py` has
  `MAX_ARITY=2` and hand-built features; arity > 2 is rejected with a clear error
  (no silent truncation), not handled.
- The Builder is a **problem-specific neural prior + MCTS** (auto-trained per
  problem from oracle traces), **not yet a real general GAN-like generator**.
- **Formula support is the P0 Horn/equational fragment**, not full FOL.
- Therefore this is a **P0 arena, not full LOGAN-UCMT**.

## 5. What must NOT be claimed?

- No **scalability** claim (evidence is small n; no scaling study).
- No **completeness** claim (DRAW ≠ impossibility; no certified DEVIL_WINS).
- No **arbitrary FOL** claim (P0 universal Horn/equational only).
- No **universal pretrained builder** claim (auto-train is per-problem).

## 6. Next real research gate (one sentence)

Make the neural Builder genuinely signature/arity-parametric — replace
`MAX_ARITY=2` and the hand features with a typed cell / factor-graph policy.

## Verification (this memo's commit)

    pytest -q                                    → 123 passed
    arena-solve cycle3_arena.json                → GOD_WINS (A ⊩_{3,800} T, A ⊭_{3,800} C)
    arena-solve cycle3_arena_tiny_budget.json    → DRAW (search_budget_exhausted)
