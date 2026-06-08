# Neural relation Builder (learned-builder-v0)

The first **learned** LOGAN Builder: a PyTorch policy proposes edits to a
partial binary relation, and the existing symbolic Devil verifies every edit.

```text
Old LOGAN:   neural generator -> graph adjacency matrix -> logical pressure
This:        neural policy     -> partial relation edits -> Devil/witness feedback
```

Scope is intentionally one binary relation (e.g. `R/2`), especially preorders.
No functions/constants, MCTS, LLM, rings, groups, EF, or SAT/SMT.

## Who generates?

The neural `RelationPolicyNet` generates **semantic edits**:

```text
set R(i,j) = TRUE   or   set R(i,j) = FALSE
```

Input is a `(4, n, n)` tensor — channels FALSE / TRUE / UNKNOWN / witness-touched
— and the output ranks the `n*n*2` possible relation edits. It does **not**
generate code or prose. The Builder applies the highest-ranked *legal* edit (an
UNKNOWN cell only; known cells are never revised).

## Who attacks?

The symbolic **Devil** (`run_devil` / `run_devil_bounded`) attacks grounded
axiom instances of the loaded theory, under the depth bound `k` and challenge
budget `b`. It returns a witness: a `failed` violation or an `unknown`
obligation (a cell to discharge).

## Who verifies?

The Devil. After **every** neural edit the Builder re-runs the bounded Devil. The
neural model is not trusted: a proposed edit that breaks an axiom surfaces as a
`failed` witness on the next step. The structure is only `satisfied` when the
Devil says so.

## Why is this not just "Codex generating a graph"?

Because the run is a verifiable semantic object, not an opaque artifact. Each run
records:

- the **theory pack** that defines the attack surface,
- the **generated structure** (partial, with explicit cell values),
- the **sequence of neural semantic edits** (`set R(i,j)=...`),
- the **Devil witnesses** that prompted each edit,
- the **final bounded verification** result, and
- the bounds `k` and `budget`.

A graph emitted by a language model has none of this: no theory, no per-edit
witness trail, no independent symbolic verification, no bounded-attack semantics.

## Training signal

Supervised examples are mined from Devil witnesses on random partial structures
(`learned/data.py`): when the Devil is blocked on an UNKNOWN relation cell, the
label is the `SparseHornPolicy` target — set a blocked **conclusion** cell TRUE,
or a blocked **premise** cell FALSE. The theory is loaded data, so the same
pipeline works for any binary-relation Horn theory (preorder, equivalence, ...).

## Commands

```bash
logan-modelbuilder make-relation-training-data \
  --theory examples/theories/preorder.json --relation R --n 3 \
  --samples 200 --out results/training/preorder_R_n3.jsonl

logan-modelbuilder train-relation-policy \
  --data results/training/preorder_R_n3.jsonl --n 3 --epochs 50 \
  --out models/preorder_R_n3.pt

logan-modelbuilder neural-build-relation \
  --theory examples/theories/preorder.json --relation R --n 3 \
  --model models/preorder_R_n3.pt --k 1 --budget 20
```

See `experiments/neural_preorder_demo.py` for the end-to-end demo.
