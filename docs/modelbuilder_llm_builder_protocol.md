# LLM Builder protocol (strict, no live API)

A protocol for letting an LLM **propose** semantic edits to a partial relation,
while keeping the LLM completely untrusted. No network calls are made in this
milestone; a deterministic mock adapter stands in for a real model.

## Trust model

- The LLM **does not generate trusted models**. It proposes `set_relation`
  actions (and, later, macro-actions) as JSON.
- The **MCTS / Builder** may use those proposals as **priors** or candidate
  actions — never as facts.
- The **symbolic Devil verifies** every resulting structure, exactly as before.
- **Bad LLM output is rejected**: malformed JSON, unknown action kinds, invalid
  truth values, and actions outside the legal action set (including any cell
  that is known/seed or out of range) are all dropped. The LLM's free-text
  `explanation` is **ignored by the verifier**.

## Messages

```text
LLMBuilderInput   { theory_name, signature_summary, current_structure,
                    last_witness, allowed_actions, goal, k, budget }
LLMProposedAction { kind="set_relation", relation, args, value }
LLMBuilderOutput  { proposed_actions[], explanation }
```

## Validation pipeline

```text
json_text --parse_llm_output--> LLMBuilderOutput
          --validate_llm_actions(allowed)--> ValidatedLLMPlan{ validated[], rejected[] }
          --apply_validated_llm_action--> PartialStructure (a copy)
```

Rejection reasons: `unknown_action_kind`, `invalid_truth_value`,
`malformed_action`, `not_in_allowed_actions`. A malformed *envelope* (bad JSON,
missing `proposed_actions` list) raises `LLMProtocolError` at parse time.

## Mock adapter

`learned/mock_llm.py` provides deterministic strategies (no network):
`first_true`, `first_false`, `witness_match` (target the witness's conclusion
cell with TRUE). It emits the same JSON shape a real LLM would, so the whole
pipeline and the demos run offline.

## MCTS hook

`mcts_relation_build(..., llm_prior=hook)` accepts an optional hook
`(structure, relation, devil_result) -> RelationEdit | None`. The hook runs the
full untrusted pipeline (emit -> parse -> validate against the legal actions).
If it returns a valid edit **on the current obligation cell**, that action's
prior is boosted; otherwise it is ignored. MCTS, the Devil, and legality rules
are unchanged — the LLM only nudges the search.

## Who generates?

```text
MCTS generates the edit trajectory.
The neural net (RelationPolicyNet) supplies learned priors.
The LLM supplies semantic / macro-action priors (untrusted, validated).
The symbolic Devil verifies every node and the final structure.
```

The LLM is one more *prior source*, never an authority. Honest framing: a
proposal is only a hint; nothing enters the structure until it passes legality
validation, and nothing is "satisfied" until the Devil says so.

## CLI

```bash
logan-modelbuilder llm-propose-relation \
  --theory examples/theories/preorder.json --relation R --n 3 \
  --seed examples/seeds/preorder_chain_3.json --mock witness_match

logan-modelbuilder mcts-relation \
  --theory examples/theories/preorder.json --relation R --n 3 \
  --seed examples/seeds/preorder_chain_3.json --llm-prior mock:witness_match
```
