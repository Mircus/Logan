# Trainable God feasibility audit (Gate R7)

Branch: `adamantium-r7-r8-trainable-god`. Scope: find the smallest honest way to make
the God/Builder trainable inside the judged adversarial loop. **Audit only — no code change.**

Files inspected: `neural_fraisse/{players.py,game.py,neural_devil.py,adversarial_train_demo.py,
mirco_gan_demo.py}`, `learned/{semantic_search.py,semantic_actions.py}`.

## Verdict

**Path B — God training needs one small new trainable wrapper, but no core rewrite.**

The existing learned God (`NeuralBuilder` + `FraisseNeuralPrior`) is frozen by construction
(`eval()` + `no_grad()` + `sorted()`), and is also entangled with a supervised checkpoint. Rather
than un-freezing it (invasive, and would conflate supervised pretraining with RL), the smallest
honest step is a **new `NeuralGodPolicy`** that mirrors the working `NeuralDevilPolicy`: it scores
the *already-legal* Builder replies (from the existing `legal_replies` enumerator) with a trainable
MLP and is updated by REINFORCE under the symbolic Judge. The frozen `FraisseNeuralPrior` stays as
the default fight God; only the co-training demo uses `NeuralGodPolicy`.

## Answers

**1. Where is NeuralGod / NeuralBuilder defined?**
`src/.../neural_fraisse/players.py` — `class NeuralBuilder` (≈L106): `order(structure, move, replies)`
ranks replies via its prior. In the demos the "neural God" is `NeuralBuilder(FraisseNeuralPrior(...))`
(`mirco_gan_demo.build_neural_god`).

**2. Where is FraisseNeuralPrior defined?**
`players.py` (≈L139). `__init__(model_path)` loads a checkpoint into `TokenSemanticPolicyNet`
(`learned/semantic_policy_net.py`) and calls `model.eval()`. `score_edits` runs the net to score edits.

**3. Where is the Builder reply chosen?**
`game.play_game.dfs` (`game.py:86`): `for edit in builder.order(structure, move, replies):` then
DFS-backtracking takes the first reply that leads to a win. Legal replies come from
`legal_replies(move)` (`game.py:63-64`) = `obligation_edits(move.obligation)`
(`semantic_search.py`). So selection = `order()` ranking + DFS first-success (discrete, not a
differentiable pick).

**4. Where are the non-differentiable barriers?**
- `FraisseNeuralPrior.__init__`: `model.eval()` (`players.py` ≈L154).
- `FraisseNeuralPrior.score_edits`: `with torch.no_grad():` (`players.py` ≈L164) — scores are detached.
- `NeuralBuilder.order`: `sorted(replies, key=-score)` (`players.py:112`) — discrete ranking.
- `play_game`: DFS first-success reply selection (`game.py:86-93`) — discrete, no log-prob.
- The God net is loaded from a frozen checkpoint (supervised), not an RL-updatable parameter set.

**5. Does the existing Builder policy expose scores/logits over legal replies?**
Yes — `FraisseNeuralPrior.score_edits(structure, obligation, edits)` returns `{edit: float}`
(`players.py:157-167`). The underlying `TokenSemanticPolicyNet(tensors)` *is* differentiable, but the
call is wrapped in `no_grad()` and the model is in `eval()`, so the exposed scores carry no gradient.

**6. Is there already a legal Builder reply enumeration function?**
Yes — `legal_replies(move)` (`game.py:63`) → `obligation_edits(obl)` (`semantic_search.py`) returns the
suggested `SetRelation/SetFunction/SetConstant` edits for the challenged cell. (Broader:
`semantic_actions.legal_semantic_edits(structure)` enumerates all UNKNOWN-cell edits.)

**7. If not, what is the smallest safe enumerator needed?**
None new. The judged loop is move-driven, so `legal_replies(move)` is exactly the right, already-tested
candidate set. No `players.py` enumerator change is required.

**8. Can we compute log_prob(selected_reply) for the Builder without a broad refactor?**
Yes. Add a small `NeuralGodPolicy` (new `neural_god.py`) that featurizes each legal reply (edit kind
one-hot, symbol/clause/args hashed one-hots, value, scalars) into a fixed vector, scores them with a
trainable MLP, forms `Categorical(logits=scores)`, samples, and returns `log_prob`. This is the exact
pattern already proven for `NeuralDevilPolicy` (`neural_devil.py`). No change to `FraisseNeuralPrior`.

**9. Can NeuralGod be trained with REINFORCE the same way as NeuralDevil?**
Yes — identical one-step bandit REINFORCE: sample a reply, get the outcome from the symbolic Judge,
reward = (+1 / -1 / -0.2) per the R6 scheme, `loss = -reward * log_prob`, SGD step. Same honest caveat
as R6: on these tiny fixed tasks the outcome is task-determined, so the gradient is a real bounded
signal and parameters genuinely change, but it cannot flip these outcomes.

**10. Which exact files must change in R8?**
- **new** `src/.../neural_fraisse/neural_god.py` — `NeuralGodPolicy` (+ optional `NeuralGodBuilder`
  wrapper exposing `.order()` so it can also play inside `play_game` for outcome computation).
- `src/.../neural_fraisse/adversarial_train_demo.py` — add the God update (or a new
  `cotraining_demo.py`); print real God loss.
- `src/.../neural_fraisse/neural_devil.py` — only if a tiny shared featurizer helper is factored out
  (optional; can be avoided).
- **new** `tests/test_trainable_god_demo.py`.
- `players.py` — **not needed** (reuse `legal_replies`). `core/` — **not touched**.

**11. What tests would prove God parameters really change?**
Snapshot `[p.detach().clone() for p in god_policy.parameters()]`, run one co-training step (sample a
reply, reward from the Judge, `loss.backward()`, `optimizer.step()`), assert
`any(not torch.allclose(b, a))`. (Mirrors the R6 Devil param-change test, which already caught a real
degenerate-gradient bug.) Also assert God loss is a finite real number, not a placeholder.

**12. What should stay fixed/default so existing fight behavior does not break?**
- `fight.py` default path unchanged (`ObligationFirstBuilder` / symbolic Devil).
- `mirco_gan_demo` unchanged (still uses the frozen `FraisseNeuralPrior` God).
- `FraisseNeuralPrior` stays frozen; `NeuralBuilder` unchanged.
- `NeuralGodPolicy` is used **only** in the co-training demo. This keeps GOD_WINS/DEVIL_WINS, the
  certificate path, and the full suite intact.

## R8 implementation path (Path B)

1. `neural_god.py`: `reply_features(state, move, edit, num_replies)` + `NeuralGodPolicy(nn.Module)` with
   `score_replies(state, move, replies) -> Tensor[N]` and `choose(...)`; optional `NeuralGodBuilder`
   (`.order()` ranks replies by the policy) so the same policy can drive `play_game`.
2. Co-training step: Devil samples a challenge (`log_prob_d`); God samples a reply for it
   (`log_prob_g`); `play_game` (symbolic Judge) returns the outcome (+ certificate on DEVIL_WINS);
   compute `(god_r, devil_r)`; `god_loss = -god_r * log_prob_g`, `devil_loss = -devil_r * log_prob_d`;
   step both optimizers.
3. Print epoch/episode/task, NeuralDevil challenge, NeuralGod reply, Judge outcome, God/Devil rewards,
   **real** God loss + Devil loss, and GOD_WINS/DEVIL_WINS/DRAW counts.
4. `tests/test_trainable_god_demo.py` proves both parameter sets change, Judge decides outcomes, the
   DEVIL_WINS certificate still verifies, GOD_WINS still works, and the full suite passes.

No `core/` change. No un-freezing of the supervised checkpoint. The trainable God is a clearly named
new policy, not a misrepresented re-label of the frozen `FraisseNeuralPrior`.
