# MCTS relation Builder (optional neural priors)

A learned-search controller over partial binary-relation structures.

```text
state  = PartialStructure (one binary relation)
action = set the Devil's current obligation cell R(i,j) to TRUE/FALSE
prior  = RelationPolicyNet logits (softmax over the obligation's two truths),
         or uniform if no model is given
reward = symbolic Devil result
search = MCTS (PUCT selection, expansion + immediate Devil evaluation)
```

## Who generates?

- **MCTS** generates edit trajectories over the partial structure.
- The **neural policy** (`RelationPolicyNet`) supplies priors — learned by
  imitation from Devil/search traces (the closure oracle), not "discovered".
- The **Devil** supplies the reward and the witness, and verifies every node.

Honest framing: the neural Builder *learns a closure policy from Devil/search
traces and proposes semantic edits that the Devil independently verifies*. MCTS
adds search on top of that policy.

## Why this matters

The greedy neural Builder commits to its single top edit each step. MCTS no
longer does that: it can **explore both truth values** of an obligation cell,
back up the Devil's reward (FAILED = -1, complete model = +1), and **recover
when the policy is uncertain or wrong**. Priors make the search cheaper — on the
seeded chain, neural priors reach the closure in far fewer nodes than uniform
priors — but uniform priors still succeed, because any complete preorder
extending the seed forces `R(0,2)=TRUE`.

## Reward

```text
Devil FAILED                  -> -1.0
Devil OK (not budget-limited) -> +1.0
Devil OK (budget exhausted)   -> +0.3   (survived bounded, not complete)
Devil UNKNOWN                 ->  0.0
```

## Branching choice

At each node the legal actions are the **two truths of the Devil's current
obligation cell** (always an UNKNOWN cell). Known/seed cells are never touched.
This keeps branching at 2 and follows the Devil's own notion of what matters,
which is what makes uniform-prior MCTS tractable.

## Still honest limitations

- Relational P0 only (one binary relation; no functions/constants/groups/rings).
- Training comes from symbolic traces / the search oracle.
- The **LLM is not yet integrated**.

## Run

```bash
# neural priors
logan-modelbuilder mcts-relation --theory examples/theories/preorder.json \
  --relation R --n 3 --seed examples/seeds/preorder_chain_3.json \
  --model models/preorder_R_n3.pt --rollouts 100

# uniform priors (no model)
logan-modelbuilder mcts-relation --theory examples/theories/preorder.json \
  --relation R --n 3 --seed examples/seeds/preorder_chain_3.json --rollouts 100
```

See `experiments/neural_mcts_preorder_demo.py` for the neural-vs-uniform
comparison written to `results/neural_mcts_preorder_demo.json`.
