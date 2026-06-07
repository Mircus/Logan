# Claude Action List: LOGAN ModelBuilder

Goal: turn the draft paper and companion sub-repo into a credible LOGAN extension that showcases model and countermodel construction, beginning with finite groups.

## 0. Preserve context and do not restart

- Treat LOGAN as an existing repo, not a new project.
- Preserve the original LOGAN roles: Builder / Devil-Opponent / Judge.
- Preserve the UCMT framing: bounded modelhood is survival under depth `k` and budget `b`.
- Preserve the group-theory strand: Cayley tables, finite groups, EF/Horn probes, witness-guided repair.
- The deliverable is a companion module inside LOGAN plus a paper that demonstrates it.

## 1. Integrate repository layout

Preferred location inside existing LOGAN:

```text
LOGAN/
  src/logical_gans/
    modelbuilder/
      __init__.py
      structures.py
      witnesses.py
      engine.py
      cli.py
      theories/
        groups.py
      opponents/
        horn_group.py
        ef_group.py          # later
      builders/
        group_builder.py
        partial_cayley.py    # next
      repair/
        local.py             # next
      export/
        json_cert.py         # next
        lean_stub.py         # later
  experiments/modelbuilder/
    e1_groups.py
    e2_random_tables.py
    e3_partial_repair.py
  tests/modelbuilder/
    test_groups.py
    test_counterexamples.py
    test_witness_replay.py
  papers/logan_modelbuilder/
    logan_modelbuilder_paper.tex
```

Action:
- Move current scaffold into this tree.
- Avoid creating a second unrelated package unless absolutely necessary.
- Update the root `pyproject.toml` so tests discover the new module.

## 2. Make witnesses replayable

Current witness data is JSON-like. Harden it.

Tasks:
- Add `Witness.replay(A) -> bool`.
- Add witness classes:
  - `AssociativityWitness(x,y,z,left,right)`
  - `IdentityWitness(e,x,left,right)`
  - `InverseWitness(e,x)`
  - `CommutativityWitness(x,y,xy,yx)`
- Add tests proving each witness replays correctly.
- Add negative tests: if the Cayley table is edited, the old witness should either replay as false or explicitly report stale.

Definition of done:
- `pytest tests/modelbuilder/test_witness_replay.py -q` passes.

## 3. Make budget `b` operational

Current P0 checks whole finite tables. This is acceptable for the draft but not enough for LOGAN.

Tasks:
- Implement sampled Horn opponent:
  - at most `b` challenges;
  - deterministic with seed;
  - challenge classes weighted by policy.
- Implement exhaustive mode as `budget='full'` or `--full`.
- Log which tuple instances were actually challenged.
- Add tests that `budget=1` checks exactly one challenge and `budget=full` checks all relevant tuples.

Definition of done:
- CLI output includes `k`, `budget`, `sampled_challenges`, and `witnesses`.

## 4. Add partial Cayley-table synthesis

The paper is stronger if LOGAN does not only instantiate known groups but also fills unknowns.

Tasks:
- Create `PartialCayleyTable` with entries in `{0,...,n-1,None}`.
- Implement monotone fill builder:
  - never revises known cells;
  - tries to satisfy identity first;
  - then inverse obligations;
  - then associativity constraints.
- Implement local revision builder:
  - may revise cells touched by witness;
  - logs edit cost;
  - refuses edits that break certificates.
- Implement `CertificateStore` with `do_not_break` constraints.

Definition of done:
- A test starts from an empty/partial table for `n=3` and converges to `C_3` or reports unknown with trace.

## 5. Add random-table baselines

Tasks:
- Implement `random_magma(n, seed)`.
- Run opponent against random tables.
- Record first witness type distribution.
- Add experiment `e2_random_tables.py`.

Definition of done:
- Produces JSONL rows with `n`, `seed`, `first_witness_kind`, `status`, and `time_ms`.

## 6. Improve counterexample search

Current counterexample to `all_groups_abelian` uses deterministic dihedral search.

Tasks:
- Add claim registry:
  - `all_groups_abelian`
  - `all_groups_cyclic`
  - `all_groups_have_exponent_2`
  - `all_groups_of_order_4_are_cyclic`
- For each claim define:
  - assumptions checker;
  - target checker;
  - counterexample witness.
- Add known models/countermodels:
  - `C_n`
  - `V_4`
  - `D_3`
  - optionally `Q_8` later.

Definition of done:
- `logan-modelbuilder counterexample --claim all_groups_cyclic` returns `V_4` with a witness/certificate explaining why not cyclic.

## 7. Add certificate JSON schema

Tasks:
- Create `schemas/model_certificate.schema.json`.
- Create `schemas/counterexample_certificate.schema.json`.
- Validate CLI output against schema in tests.
- Include fields:
  - `logan_version`
  - `theory`
  - `structure`
  - `domain_size`
  - `k`
  - `budget`
  - `opponent_policy`
  - `builder_policy`
  - `assumptions_verified`
  - `target_refuted`
  - `witnesses`
  - `trace_hash`

Definition of done:
- Every CLI command can emit `--json` and validate against schema.

## 8. Result tables for the paper

Tasks:
- Add script `scripts/make_tables.py` that reads JSONL and emits LaTeX table rows.
- Run sweeps:
  - `C_2, C_3, C_4, C_5`
  - `V_4`
  - `D_3, D_4`
  - random tables for `n=3,4,5`
- Produce tables:
  - model survival table;
  - counterexample table;
  - first-witness distribution for random tables.

Definition of done:
- Paper can include real output, not placeholder output.

## 9. Paper improvements

Tasks:
- In the LaTeX paper, distinguish sharply:
  - classical satisfaction `A \models T`;
  - LOGAN bounded survival `A \Vdash_{k,b} T`;
  - countermodel certificate for `T => phi`.
- Add a section “Why this is model-side AI for mathematics.”
- Add one full worked example with the actual JSON certificate from `D_3`.
- Add a subsection relating this to UCMT but do not let UCMT swallow the implementation paper.
- Add a short limitations section:
  - finite only;
  - bounded only;
  - known builders in P0;
  - no proof of general theorem from finite checks.

Definition of done:
- The paper compiles with `pdflatex` or `latexmk`.
- It includes at least one real result table generated from the repo.

## 10. CI and quality

Tasks:
- Add GitHub Action:
  - install package;
  - run tests;
  - run group experiment;
  - optionally compile paper.
- Add `ruff` or basic linting if the parent repo already uses it.
- Add `mypy` only if parent repo already tolerates typing overhead.

Definition of done:
- New LOGAN PR is green.

## 11. P1/P2 domain expansion

After finite groups P0 is stable:

P1:
- finite monoids and semigroups;
- graph properties with explicit small witnesses;
- EF probes for graph indistinguishability.

P2:
- arithmetic fragments `[0,N]` with truncated/modular operations;
- feasible frontier / FIS-inspired activation of terms;
- Lean export stubs for witnesses.

P3:
- finite categories;
- small posets/lattices;
- positive topology tasks if needed.

## 12. Suggested Claude work order

1. Run tests on current scaffold.
2. Move scaffold into real LOGAN tree.
3. Implement replayable witness classes.
4. Make budgeted Horn opponent real.
5. Add partial Cayley tables and monotone fill.
6. Add counterexample claim registry.
7. Generate real JSONL experiment output.
8. Generate LaTeX result tables.
9. Patch paper with real tables and JSON certificates.
10. Open PR or commit with clean message:

```text
Add LOGAN ModelBuilder for finite model/countermodel synthesis
```
