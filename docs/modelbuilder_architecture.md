# LOGAN-ModelBuilder architecture (v0.1-alpha)

## Two systems, one invariant

**Original LOGAN**
: neural graph-GAN + EF/game-inspired logical pressure (torch / torch_geometric;
  EF games over NetworkX graphs).

**LOGAN-ModelBuilder** (this subpackage)
: symbolic bounded partial finite-model synthesis. stdlib-only, no torch.

**Shared invariant**
: *generation under logical attack* — a Builder proposes/fills a structure while
  an Opponent (the Devil) probes it with bounded logical challenges.

These are different implementations of the same idea, not the same code. This
note exists so the project is not mistaken for *either* a neural-GAN extension
only *or* a finite-group constructor.

## Current P0 kernel

- universal **relational Horn clauses** (`premises -> conclusion`)
- universal **equations** over function symbols
- finite **partial structures** (relation cells TRUE/FALSE/UNKNOWN; function and
  constant cells defined or UNKNOWN)
- a **three-valued Devil** (deterministic; exhaustive `run_devil` and bounded
  `run_devil_bounded`)
- **monotone builder policies** (`SparseHornPolicy`, `MaximalHornPolicy`)
- **DFS backtracking** search over admissible cell fills
- `k` = logical/term **depth** bound on the attack surface
- `b` = Devil **challenge budget** (max grounded instances per run)

### Component map

```
core/
  types        Truth = TRUE | FALSE | UNKNOWN
  signature    relation / function / constant symbols
  terms,atoms,clauses   Var/Const/Func ; RelAtom/EqAtom ; HornClause
  partial_structure     finite structure with UNKNOWN cells
  eval         three-valued evaluator
  depth        term/atom/clause depth (for k)
  devil        run_devil (exhaustive), run_devil_bounded (k, b)
  policy       BuilderPolicy: Sparse/Maximal
  generator    monotone fill (never revises)
  backtracking backtracking_generate (DFS, can revise via search)
  loader       JSON theory/claim/structure packs (closed-world structures)
  runner       synthesize / check / refute
  theory       Theory, Claim containers
  witness      replayable Witness
examples/      Python convenience builders (preorder, semigroup)
fixtures/      known_groups Cayley tables (regression sanity only)
cli.py
```

(JSON theory/claim/structure data packs live at repo-root `examples/`.)

## Deliberately NOT in v0.1-alpha

- full first-order logic
- existentials (`Exists`)
- certificates (do-not-break store)
- obligation lifecycle
- EF graph probes
- SAT/SMT backends
- Lean export
- serious rings/groups theory packs
- open-world structure loading

These are roadmap items; the kernel is intentionally small and honest.
