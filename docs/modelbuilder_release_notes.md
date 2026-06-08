# LOGAN-ModelBuilder release notes

## v0.1.0-alpha

**LOGAN-ModelBuilder v0.1-alpha: bounded partial finite-model synthesis for
universal Horn/equational theory packs.**

A symbolic, torch-free subpackage at `src/logical_gans/modelbuilder/` that fills
unknown interpretation tables of a finite partial structure while a three-valued
Devil probes bounded axiom instances.

### Highlights

- Theory packs as **data** (JSON): signatures, Horn/equational clauses, claims,
  structures.
- **Generators**: monotone fill (`SparseHornPolicy`, `MaximalHornPolicy`) and
  **DFS backtracking** search.
- **Bounded Devil**: depth bound `k` and challenge budget `b`, with
  `budget_exhausted` honesty (survived-bounded ≠ globally verified).
- CLI: `synthesize`, `search`, `check`, `refute` (+ convenience wrappers).
- Docs: [architecture](modelbuilder_architecture.md),
  [output schema](modelbuilder_output_schema.md),
  [walkthrough](modelbuilder_v0_1_walkthrough.md).

### Scope (honest boundaries)

P0 fragment only: universal relational Horn clauses + universal equations over
finite structures. **Not** included: full FO, existentials, certificates,
obligation lifecycle, EF graph probes, SAT/SMT, Lean export, serious rings/groups
packs, open-world loading.

### Tag

To tag this release:

```bash
git tag -a v0.1.0-alpha -m "LOGAN-ModelBuilder v0.1-alpha"
git push origin v0.1.0-alpha
```
