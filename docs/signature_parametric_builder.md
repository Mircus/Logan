# Signature-parametric Builder

## The pivot

New LOGAN is **not** a builder for groups, rings, graphs, preorders, or
semigroups. Those are only theory packs (example inputs).

New LOGAN is a builder for **input first-order signatures**. Given a signature
Σ with arbitrary relation / function / constant symbols and axioms in the P0
fragment, it generates partial finite Σ-structures by semantic edits while the
Devil attacks and verifies bounded formula instances.

## Core action space

The Builder operates over generic semantic edits, generated from the input
signature — never relation-only:

```text
SetRelation(symbol, args, TRUE/FALSE)
SetFunction(symbol, args, value in domain)
SetConstant(symbol, value in domain)
```

(`learned/semantic_actions.py`, `legal_semantic_edits` / `apply_semantic_edit`.)

## The loop

```text
MCTS              generates semantic-edit trajectories
                  (learned/generic_mcts.py: mcts_semantic_build)
Obligations       core/obligations.py turns an UNKNOWN Devil result into a
                  generic relation/function/constant obligation; the candidate
                  edits at a node discharge that obligation
Priors            learned/priors.py: PriorProvider over semantic edits
                  (Uniform / ObligationFirst / MockLLM); a learned net can be
                  wrapped as a provider later
LLM               learned/generic_llm_protocol.py validates set_relation /
                  set_function / set_constant proposals; untrusted, explanation
                  ignored
Devil             core/devil.py verifies bounded formula instances after every
                  edit and at every node
```

So the answer to "who generates?" is signature-parametric:

```text
MCTS generates semantic edit trajectories.
NN and LLM supply priors over semantic edits.
The Devil verifies bounded formula instances.
```

## Status of the two builders

- The relation-specific learned Builder (`learned/encoding.py`,
  `policy_net.py`, `builder.py`, `mcts.py`, `llm_protocol.py`) is a **prototype**
  — useful, but pinned to one binary relation `R`.
- The **signature-parametric Builder** (`semantic_actions.py`,
  `core/obligations.py`, `generic_mcts.py`, `generic_llm_protocol.py`,
  `priors.py`) is the **main line**.

## Example (not a named math domain)

`examples/theories/toy_mixed_signature.json` — relation `E/2`, function `s/1`,
constant `a`; axioms `E(x,s(x))`, `s(s(x))=x`, `E(a,a)`.

```bash
logan-modelbuilder mcts-semantic \
  --theory examples/theories/toy_mixed_signature.json --n 2 --rollouts 200
```

returns `satisfied`, assigning the constant, the function table for `s`, and
relation cells for `E` — with a trace containing `SetConstant`, `SetFunction`,
and `SetRelation`. Because no `k`/`budget` is set, `satisfied` means the
exhaustive Devil accepts the structure.

After this generic mechanism is solid, groups/rings/etc. can be added as
ordinary theory packs — but as inputs, never as hard-coded specializations.
