# Your first Neural Fraïssé fight

This walks you through running one God-vs-Devil fight and reading the result.

## What can I do?

You describe a small finite mathematical problem — a signature, some assumptions,
and a goal — and watch a **Builder** (God) try to construct a finite structure
while an **active Devil** challenges it. A **Judge** checks each step. At the end
you get an outcome and, if God wins, a concrete structure and a witness.

## The example problem

`examples/problems/cycle3_fight.json`:

```json
{
  "signature": { "relations": [{"name":"E","arity":2}],
                 "functions": [{"name":"s","arity":1}],
                 "constants": ["a"] },
  "theory": ["forall x: E(x, s(x))", "forall x: s(s(s(x))) = x"],
  "goal":   { "mode": "refute", "claim": "s(a) = a" },
  "resources": { "domain_size": 3, "depth": 3, "budget": 800, "max_rounds": 50 },
  "builder": { "mode": "active_symbolic", "train": "auto" }
}
```

### What the signature means
- `E/2` — a binary relation (think: an edge `E(x,y)`).
- `s/1` — a unary function (a "successor" map `s(x)`).
- `a` — a constant (a named element).
The domain is `{0, 1, ..., domain_size-1}`.

### What the assumptions (theory) mean
- `forall x: E(x, s(x))` — every element has an edge to its successor.
- `forall x: s(s(s(x))) = x` — applying `s` three times returns you to the start
  (so `s` is a permutation whose order divides 3).

### What the goal means
`mode: refute, claim: s(a)=a` means: **build a structure that satisfies the
assumptions but makes the claim false** — i.e. find `A` with `A ⊨ T` and `A ⊭ C`.
Here that forces `s` to be a fixed-point-free 3-cycle and `a` a non-fixed point.
(Use `"mode": "model"` instead to just build a model of the assumptions.)

### What depth and budget are
- `depth` (k) — how deep the Devil's logical challenges may go (term depth). The
  axiom `s(s(s(x)))=x` has depth 3, so `depth` must be at least 3.
- `budget` — how many construction steps (nodes) the search may spend before
  giving up.
- `max_rounds` — a soft cap on the displayed fight length.

## Who are the players?
- **God / Builder** — fills in one interpretation cell at a time
  (`set_function`, `set_relation`, `set_constant`) trying to reach the goal.
- **Devil** — *actively chooses* the next challenge: it points at a specific cell
  that currently blocks an assumption (or a claim cell that must be committed).
  It is a symbolic active Devil (not learned in this release).
- **Judge** — checks the structure after each reply with the bounded verifier and
  records progress, a win, or a dead end.

## How do I run it?

```bash
python -m logical_gans.modelbuilder.neural_fraisse.fight \
  examples/problems/cycle3_fight.json
```

Two builder modes (the file picks the default; override with `--builder`):
- `active_symbolic` — active Devil + symbolic Builder (fast; the default).
- `neural_active` — active Devil + **learned** Builder, auto-trained on the cyclic
  task family (slower):

```bash
python -m logical_gans.modelbuilder.neural_fraisse.fight \
  examples/problems/cycle3_fight.json --builder neural_active
```

## How do I read the trace?

Each round prints the Devil's challenge, God's reply, and the Judge's note:

```text
Round 1
  Devil: ChallengeClauseInstance clause=axiom_0 x=0 target=s(0)
  God:   set_function s(0)=1
  Judge: progress
...
Round 7
  Devil: ChallengeGoalCell clause=claim target=a
  God:   set_constant a=0
  Judge: progress
```

### The first three rounds, read out loud
1. The Devil sees that `E(0, s(0))` cannot be evaluated because `s(0)` is unknown,
   so it challenges the cell `s(0)`. God answers `s(0)=1`. Judge: progress.
2. Now `s(0)=1`, so the Devil demands the edge `E(0,1)`; God sets `E(0,1)=true`.
3. The Devil moves to `x=1`: `s(1)` is unknown, so it challenges `s(1)`; God sets
   `s(1)=2`. The pattern continues until `s` is a full 3-cycle and `a` is chosen.

## How do I read the outcome?

```text
GOD_WINS:
  God/Builder found a structure satisfying the assumptions and achieving the goal.

DRAW:
  The game ran out of bounded resources without a verified win.

DEVIL_WINS:
  Not implemented in this release.
```

On `GOD_WINS` you also get:

```text
Final structure:
  constants: a=0
  functions: s(0)=1, s(1)=2, s(2)=0
  relations: E(0,1)=true, E(1,2)=true, E(2,0)=true
Witness:
  clause 'claim' FAILED ... : premises all TRUE but conclusion is false
```

The **structure** is the model God built; the **witness** is the concrete reason
the claim fails — here `s(a)=s(0)=1 ≠ 0=a`.

## Change one thing and rerun (exercise)

Copy `examples/problems/cycle3_fight.json`, change **only** the claim from
`s(a) = a` to `E(a, a)`, and rerun the fight on your copy.

```json
"goal": { "mode": "refute", "claim": "E(a, a)" }
```

Expected result: still `GOD_WINS`. God builds the same 3-cycle
(`s(0)=1, s(1)=2, s(2)=0`) and picks `a=0`; because `E(0,0)` is *not* one of the
forced successor-edges (`E(0,1), E(1,2), E(2,0)`), it stays false, so the claim
`E(a,a) = E(0,0)` is refuted. Compare the trace: the early rounds (building `s`
and the edges) are the same; the difference is the final claim cell the Devil
challenges (`E(0,0)` instead of the constant/`s(a)` equation).

Other small edits to try:
- Change `domain_size` (4, 5).
- Change the order axiom (e.g. `s(s(x)) = x` for an involution).
- Switch `goal.mode` to `"model"` to just build a model of the assumptions.

Keep everything small and bounded.

## Common failure modes
- **`DRAW`** — the search hit `budget`/`max_rounds` without a verified win. Raise
  `budget`, raise `domain_size` (a solution may not exist at the current size), or
  check the claim is actually refutable under the theory.
- **Parse error** — a formula uses a symbol not in the signature, or wrong arity.
- **Depth too small** — if `depth` (k) is below a deep axiom's term depth (e.g. 3
  for `s(s(s(x)))=x`), the Devil cannot pose that challenge; raise `depth`.
- **`neural_active` is slow** — it auto-trains a small Builder first; use the
  default `active_symbolic` for a quick run.

## Current limitations

```text
small bounded finite problems only
controlled formula fragment (universal Horn / equational)
active symbolic Devil, not a learned Devil
learned Builder only
not full first-order logic
not a theorem prover
not a GAN claim
not general model theory
no DEVIL_WINS yet
```

For the held-out evidence that the learned Builder beats non-neural baselines,
see `reports/neural_fraisse_poc.md`.
