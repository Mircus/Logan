# Bring Your Own Problem (BYOP)

Write one JSON file describing a finite first-order problem, then run one command.
The capsule translates it to the kernel format, optionally trains a small
problem-specific neural prior, searches for a finite countermodel, and verifies
the result with the bounded Devil.

## 1. What a problem file is

A single self-contained JSON file (see `examples/problems/TEMPLATE_problem.json`)
with: a `signature`, a `theory` (list of formulas), a single `claim`, a
`domain_size`, a `bound` (`depth`, `budget`), and a `generator` block.

The capsule looks for a finite Σ-structure `A` that **satisfies the theory but
refutes the claim**: `A ⊩_{k,b} T` and `A ⊭_{k,b} C`.

## 2. Supported signature syntax

```json
"signature": {
  "relations": [{"name": "R", "arity": 2}],
  "functions": [{"name": "f", "arity": 1}],
  "constants": ["c"]
}
```

Relations and functions have a name and a non-negative integer arity. Constants
are bare names. Symbol names are case-sensitive.

## 3. Supported theory/claim fragment

Each formula is a string in the kernel's **universal Horn / equational P0
fragment**:

- `forall x: <atom>` or `forall x, y: <atom>` (universally quantified)
- a ground atom with no quantifier (e.g. a claim like `R(c,c)` or `f(a) = a`)
- atoms are either a relation `R(t1, t2)` or an equation `t1 = t2`
- terms are variables (declared in the `forall`), constants, or function
  applications `f(t)` (nesting allowed: `f(f(x))`)

Examples that work:
```
forall x: R(x, f(x))
forall x: f(f(x)) = x
R(c,c)
f(a) = a
```

Identifiers used in a term must be a declared constant, a declared function, or a
variable bound by the enclosing `forall`. Relations/functions must be used at
their declared arity.

## 4. Meaning of n, k, b

- `domain_size` (**n**): the finite domain is `{0, 1, ..., n-1}`.
- `bound.depth` (**k**): the Devil only checks clauses whose term depth ≤ k.
  Set k at least as large as your deepest term (e.g. `f(f(f(x)))` has depth 3).
- `bound.budget` (**b**): the Devil checks at most b grounded instances. Make b
  large enough to cover your clauses at size n (a few hundred is plenty for small
  problems).

If k or b are too small, a "satisfied" result only means *survived the bounded
attack*, not globally verified.

## 5. What `--auto-train` does

`--auto-train` fits a **small, problem-specific** `SemanticPolicyNet` from oracle
traces mined for *your* problem, then uses it as the search prior. It is **not** a
universal pretrained model — it learns nothing transferable beyond this problem.
Auto-train is on by default (or set `generator.auto_train`); the flag forces it.
If the oracle finds no refuting trajectory, the run falls back to a non-neural
prior and reports `not_found` cleanly.

## 6. One working example

```bash
PYTHONPATH=src python -m logical_gans.modelbuilder.cli validate-problem \
  examples/problems/byop_sample_countermodel.json

PYTHONPATH=src python -m logical_gans.modelbuilder.cli prove-countermodel \
  examples/problems/byop_sample_countermodel.json --auto-train
```

Σ = {Q/2, g/1, d}; T = {∀x Q(x,g(x)); ∀x g(g(x))=x}; C = g(d)=d; n=2, k=2, b=500.
The capsule returns `g = swap(0,1)`, `d = 1`, so `g(d) = 0 ≠ 1 = d` — i.e.
`A ⊩_{2,500} T` and `A ⊭_{2,500} C`. The certificate is written to
`results/byop_sample_countermodel_certificate.json`.

## 7. Common failure modes

- **`unknown relation/function symbol`** — a symbol in a formula is not declared
  in the signature, or is misspelled.
- **`used with arity N, expected M`** — wrong number of arguments for a symbol.
- **`neither a declared constant nor a quantified variable`** — a bare identifier
  in a term that you forgot to declare or to bind with `forall`.
- **`status = not_found`** — no countermodel was found within (n, k, b). The
  claim may be unrefutable (the theory forces it), or the bound/budget is too
  small, or the current oracle could not seed training for this problem shape.
- **`status = refuted_unverified`** — a candidate was found but did not pass the
  bounded re-check; treat as not accepted.

## 8. What LOGAN does not yet support

- arbitrary full first-order logic;
- quantifier alternation beyond the universal Horn/equational fragment above;
- existential goals or unbounded (infinite-domain) semantics;
- a pretrained, universal model-building model — `--auto-train` is per-problem;
- any completeness or scalability guarantee.

The supported surface is exactly: finite signatures (relations/functions/
constants), the bounded universal clause/equational fragment, a finite domain n,
and explicit k and b.
