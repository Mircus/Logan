# UCMT cycle3 countermodel capsule

**Gate name:** ucmt_cycle3_capsule (UCMT proof capsule built on Gate 2)

**SHA:** branch `signature-parametric-builder`, built on `41ecb3c` (Gate 2:
"Add neural semantic ablation gate"). Capsule added in the commit that adds
this file.

Capsule implementation SHA:
a221852a0b387a584b47a339f62727a230bcff4b

**Command:**
```
PYTHONPATH=src python -m logical_gans.modelbuilder.cli prove-countermodel \
  examples/problems/cycle3_countermodel.json
# or:
bash scripts/showcase_logan_ucmt.sh
```

**Bound k,b:** k = 3 (depth), b = 800 (Devil instance budget). k=3 is the
minimum depth that admits the order-3 axiom `s(s(s(x)))=x`; b=800 covers all 6
grounded theory instances at n=3, so `run_devil_bounded(k=3,b=800)` coincides
with exhaustive verification here.

**Expected output shape:**
- readable certificate with `A ⊩_{3,800} T` and `A ⊭_{3,800} C`
- `results/cycle3_countermodel_certificate.json` with `accepted=true`,
  `certificate.{model_relation,counterclaim_relation,theory_status,claim_status}`,
  `generator.{kind,uses_neural_policy,random_rollouts,success_via}`, `structure`,
  `witness`, and an `ablation` block for uniform / obligation_first / neural.

**Observed result summary:**
- `A ⊩_{3,800} T`, `A ⊭_{3,800} C`; theory_status=satisfied, claim_status=failed.
- Structure: `a=1`, `s={0→2,1→0,2→1}` (a fixed-point-free 3-cycle, `s³=id`),
  `E(0,2)=E(1,0)=E(2,1)=true`.
- Witness: `s(a)=s(1)=0 ≠ 1=a`.
- Generator: `neural_semantic_mcts`, `random_rollouts=disabled`,
  `success_via=guided_tree_policy`.
- Ablation: uniform failed at 800 nodes, obligation_first failed at 800 nodes,
  neural succeeded at 39 nodes.

**What this proves:**
- A neural-prior-guided tree policy (no random rollouts) constructs a finite
  Σ-structure that, under the bounded Devil at (k=3,b=800), satisfies the theory
  and refutes the claim `s(a)=a`, with a replayable equality witness.
- The learned prior is load-bearing: under identical budget/seed/verification,
  uniform and obligation-first priors fail to refute within budget while neural
  succeeds in 39 Devil-evaluated nodes — success comes only from the guided tree
  descent, not from lucky simulation.

**What this does not prove:**
- Nothing about infinite models, completeness, or unbounded depth: the relations
  are bounded (k=3, b=800), not global validity.
- No claim beyond this single Σ/T/C at n=3; no groups/rings/semigroups, no solver
  backend, no live LLM.
- Not a statement that uniform/obligation-first can never succeed — only that
  they do not within this fixed budget, whereas neural does (and much sooner).
- The bound here is wide enough to equal exhaustive checking at n=3; it is not a
  demonstration of partial/under-budget verification.
