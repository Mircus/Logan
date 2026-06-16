# Colab quickstart

Run the LOGAN Neural Fraïssé notebook in Google Colab — no local install needed.

1. Click the **Open in Colab** badge in the [README](../../README.md), or open
   `notebooks/neural_fraisse_quickstart.ipynb` directly in Colab.
2. **Runtime → Run all.**
3. The first cell clones LOGAN and installs it (`pip install -e .`). This takes a
   minute on first run.
4. You should then see:
   - the example problem `cycle3_fight.json`;
   - an alternating fight trace with `Devil` (`ChallengeClauseInstance` /
     `ChallengeGoalCell`), `God`/Builder (`set_function` / `set_relation` /
     `set_constant`), and `Judge` events;
   - the final outcome: **GOD_WINS** or **DRAW**;
   - the final structure and a **witness** (when GOD_WINS);
   - a small held-out benchmark table.
5. If the install cell fails (transient network), **Runtime → Restart runtime**
   and **Run all** again.

Notes:
- The notebook uses the `active_symbolic` builder for speed. `neural_active`
  auto-trains a small Builder and is slower.
- This is a proof of concept on one controlled task family — see
  [`reports/neural_fraisse_poc.md`](../../reports/neural_fraisse_poc.md) and the
  limitations in the [first-fight tutorial](first_neural_fraisse_fight.md).
