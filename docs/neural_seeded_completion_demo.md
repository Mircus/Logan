# Neural seeded completion demo

This demo proves the learned Builder does more than reproduce the trivial
sparse identity preorder: given a **partial logical world**, it discovers and
adds a **forced fact**.

## The task

```text
Theory: preorder   (reflexive + transitive)
Seed (open-world):
    R(0,1) = TRUE
    R(1,2) = TRUE
    everything else UNKNOWN
Forced by transitivity:
    R(0,1) and R(1,2) -> R(0,2)
    => R(0,2) = TRUE
```

The Builder produces the reflexive-transitive closure of the chain — diagonal
TRUE, `R(0,1)`, `R(1,2)`, `R(0,2)` TRUE, everything else FALSE — and `R(0,2)` is
the only non-seed, non-diagonal TRUE cell. It is there because it had to be.

## Why this is different from "generate a graph"

```text
The user does not ask for an example object.
The user gives a partial logical world (the open-world seed).
The Devil discovers a forced axiom instance (transitivity on the chain).
The neural Builder chooses a semantic edit (set R(0,2)=TRUE).
The Devil verifies the completed structure (exhaustive check passes).
The trace shows exactly why R(0,2) had to be added.
```

A language model emitting an adjacency matrix gives none of this: no seed
semantics, no per-edit witness, no forced-fact derivation, no independent
symbolic verification.

## Open-world vs closed-world

The seed uses the **open-world** loader (`load_seed_open_world`): listed cells
are fixed, everything else stays UNKNOWN. This is deliberately different from the
**closed-world** loader used by `check` (which forces unlisted relation cells to
FALSE). Completion only makes sense open-world: the Builder must *decide* the
unknowns, not have them pre-decided as FALSE.

## How the policy learns to close, not just sparsify

Training targets come from a theory-general **closure oracle**: for each sampled
partial structure the backtracking search is seeded from it, and the blocked
cell's value in the model it finds becomes the label. So forced cells are labeled
TRUE and free cells FALSE. A pure SparseHornPolicy heuristic (premise-block →
FALSE) would instead set `R(0,2)=FALSE` at the first transitivity premise-block
and the Devil would then fail — the corner-painting that motivates MCTS next.

## Run

```bash
PYTHONPATH=src python experiments/neural_seeded_preorder_demo.py
```

Expected `results/neural_seeded_preorder_demo.json`:

```json
{
  "status": "satisfied",
  "exhaustive_devil": "ok",
  "forced_fact": "R(0,2)",
  "forced_fact_present": true,
  "seed_facts_preserved": true
}
```

Or via the CLI with a trained model:

```bash
logan-modelbuilder neural-build-relation \
  --theory examples/theories/preorder.json --relation R --n 3 \
  --model models/preorder_R_n3.pt --seed examples/seeds/preorder_chain_3.json
```
