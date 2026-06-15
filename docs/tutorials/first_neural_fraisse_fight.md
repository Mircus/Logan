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

Read it as: the Devil demands a value for `s(0)` (because `E(0, s(0))` is blocked),
God answers `s(0)=1`, the Judge confirms it did not break anything, and so on.

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

## How do I modify the example?
- Change `domain_size` (try 4, 5).
- Change the order axiom (`s(s(x))=x` for an involution, etc.).
- Change the claim (`E(a,a)`, `forall x: E(x,x)`).
- Switch `goal.mode` to `"model"` to just build a model.
Keep things small and bounded.

## Current limits

```text
small bounded finite problems only
controlled formula fragment (universal Horn / equational)
active symbolic Devil
learned Builder only
not full FOL
not a theorem prover
not a GAN claim
not general model theory
```

For the held-out evidence that the learned Builder beats non-neural baselines,
see `reports/neural_fraisse_poc.md`.
