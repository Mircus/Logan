# LOGAN ModelBuilder

Companion sub-repo for **LOGAN as a mathematical model and countermodel builder**.

This is designed to sit inside the existing LOGAN repo, for example:

```text
LOGAN/
  src/logical_gans/...
  applications/logan_modelbuilder/   # this package, or
  src/logical_gans/modelbuilder/     # integrated package namespace
```

The first implemented arena is finite group theory via Cayley-table synthesis and witness-guided checking.

## What it does now

- Builds concrete finite group models: cyclic groups `C_n`, Klein four `V_4`, and dihedral groups `D_m` of order `2m`.
- Checks group axioms by explicit LOGAN witnesses: associativity triples, identity failures, inverse failures.
- Finds counterexamples to candidate mathematical claims, currently:
  - `all finite groups are abelian` using `D_3` / `S_3`.
- Emits JSON-style certificates/witnesses for model and countermodel runs.

## Quick start

```bash
python -m pip install -e .[dev]
pytest -q
logan-modelbuilder build-group --kind cyclic --n 5
logan-modelbuilder counterexample --claim all_groups_abelian --max-order 8
python experiments/e1_groups.py --out results_groups.jsonl
```

## Design

LOGAN separates three roles:

- **Builder** proposes or repairs a finite structure.
- **Opponent / Devil** probes it with bounded challenges and returns small witnesses.
- **Judge** turns the result into a model certificate, counterexample certificate, or failure trace.

A model is therefore not just an object satisfying axioms silently. It is an object that survives a bounded attack surface and carries a witness/certificate trace explaining what was checked.

## Next domains

- finite monoids and semigroups;
- relational structures / graphs with EF probes;
- finite arithmetic fragments;
- Lean/JSON export of witnesses.
