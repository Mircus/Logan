# LOGAN-ModelBuilder v0.1-alpha walkthrough

Run from the repo root with the package on the path:

```bash
export PYTHONPATH=src   # or: pip install -e .
```

All commands print one JSON object (`status`, `structure`, `trace`, ...). See
[output schema](modelbuilder_output_schema.md) for field definitions.

## 1. Synthesize a preorder — sparse policy

```bash
logan-modelbuilder synthesize \
  --theory examples/theories/preorder.json \
  --n 3 \
  --policy sparse_horn \
  --k 1 \
  --budget 20
```

The **sparse** policy sends UNKNOWN premises to FALSE, so it builds the
**identity preorder** on `{0,1,2}`: only `R(i,i)` is TRUE.

## 2. Synthesize a preorder — maximal policy

```bash
logan-modelbuilder synthesize \
  --theory examples/theories/preorder.json \
  --n 3 \
  --policy maximal_horn \
  --k 1 \
  --budget 20
```

The **maximal** policy sends UNKNOWN premises to TRUE, so it builds the
**total preorder**: every `R(i,j)` is TRUE. Same theory, different model — that
is the point of policies being a *construction* choice, not semantics.

## 3. Search for a semigroup operation

```bash
logan-modelbuilder search \
  --theory examples/theories/semigroup.json \
  --n 2 \
  --k 2 \
  --budget 100 \
  --max-nodes 10000
```

Backtracking DFS **fills the `mul/2` table** until associativity holds on all
checked instances — it finds an associative operation on 2 elements (the search
order yields the constant operation, which is associative). `nodes` reports how
many search nodes were explored.

## 4. Refute "all preorders are antisymmetric"

```bash
logan-modelbuilder refute \
  --theory examples/theories/preorder.json \
  --claim examples/claims/antisymmetry.json \
  --n 2 \
  --policy maximal_horn
```

Builds a model of the *theory* (the 2-element total preorder), then attacks the
*claim*. Antisymmetry fails, so it returns `status: refuted` with a **2-element
countermodel** and a witness at `x=0, y=1` (both `R(0,1)` and `R(1,0)` TRUE but
`0 != 1`).

## What `budget_exhausted: true` does NOT prove

With `--budget` (or `--k`) set, the Devil checks only a bounded slice of the
attack surface. If a run ends `satisfied` because a `challenge` reported
`status: ok` with `budget_exhausted: true`, the structure **survived the bounded
attack** — it was not falsified within the budget. That is **not** a proof of
global satisfaction, and the structure may still contain UNKNOWN cells.

For an exhaustive guarantee, run **without** `--k` and `--budget` and confirm the
resulting structure has no UNKNOWN cells. Bounded survival is an operational,
budget-relative statement — by design.
