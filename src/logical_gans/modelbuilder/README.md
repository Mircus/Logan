# LOGAN ModelBuilder

A bounded **partial finite-model generation** kernel that lives inside LOGAN
at `src/logical_gans/modelbuilder/`. It is stdlib-only and does not import
torch.

Given:

- a finite domain size `n`,
- a signature (relation / function / constant symbols),
- axioms in a restricted first-order fragment,

the **Generator** fills an initially *unknown* interpretation table while the
**Devil** checks bounded axiom instances and returns concrete
witnesses/obligations.

## P0 fragments

- **Fragment A — universal relational Horn clauses:** `forall x...: A1 & ... & Am -> B`
  (premises may be empty, e.g. reflexivity). Covers preorders, partial orders,
  graph constraints.
- **Fragment B — universal equations:** `forall x...: t1 = t2`. Covers algebraic
  theories (semigroups, monoids, groups, rings).

Not yet implemented: `Exists`, `Or`, arbitrary `Not`, full first-order logic.

## Layout

```text
core/
  types.py            # Truth: TRUE / FALSE / UNKNOWN
  signature.py        # RelationSymbol, FunctionSymbol, ConstantSymbol, Signature
  terms.py            # Var, Const, Func
  atoms.py            # RelAtom, EqAtom
  clauses.py          # HornClause (premises -> conclusion)
  partial_structure.py# PartialStructure with UNKNOWN cells
  eval.py             # three-valued bounded evaluator
  devil.py            # deterministic exhaustive checker -> Witness
  generator.py        # monotone-fill generator
  witness.py          # replayable Witness
examples/
  preorder.py         # R/2: reflexive + transitive; antisymmetry refutation
  semigroup.py        # mul/2: associativity
fixtures/
  known_groups.py     # hand-written Cayley tables — sanity fixtures only,
                      # NOT the builder
cli.py
```

## Quick start

```bash
PYTHONPATH=src python -m logical_gans.modelbuilder.cli synthesize-preorder --n 3
PYTHONPATH=src python -m logical_gans.modelbuilder.cli refute-preorder-antisymmetry
PYTHONPATH=src python -m logical_gans.modelbuilder.cli synthesize-semigroup --n 1
pytest -q tests/test_core_preorder.py tests/test_core_semigroup.py
```

Each command prints JSON with `status`, the resulting `structure`, and a
`trace`/`witness`.

## Roles

- **Generator** fills unknown interpretation cells (monotone, no revision yet).
- **Devil (Opponent)** probes bounded axiom instances and returns witnesses
  (a `failed` violation) or obligations (an `unknown` cell to discharge).

Known finite groups (`C_n`, `V_4`, `D_m`) are kept only under `fixtures/` as
regression sanity checks — they are not the model builder.
