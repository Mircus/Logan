# DEVIL_WINS feasibility audit (Gate R1)

Branch: `adamantium-prototype1-mirco-gan`. Scope: find the smallest safe path to a real
`DEVIL_WINS` for the n=2 impossible cyclic demo (Demo B). **No implementation in this gate.**

Files inspected: `src/logical_gans/modelbuilder/neural_fraisse/{game.py,players.py,fight.py,tasks.py}`,
`src/logical_gans/modelbuilder/learned/semantic_search.py`,
`src/logical_gans/modelbuilder/core/{devil.py,obligations.py,partial_structure.py}`,
`examples/problems/`, `tests/`.

## Headline finding

The obstruction signal **already exists** and is currently discarded. `play_game`
(`neural_fraisse/game.py:71`) returns three internal statuses:
- `"won"` — a winning completion was found;
- `"draw"` — `_BudgetExhausted` raised (`game.py:88,99-100`): the node budget was hit;
- `"lost"` — the DFS returned `None` (`game.py:101-102`): the obligation-ordered completion
  tree was **fully closed within budget** with no win.

`"lost"` is a *budget-not-hit, frontier-closed* result — i.e. a bounded obstruction over the
explored win-relevant space. But `fight.py:97` does
`outcome = "GOD_WINS" if out.outcome == "won" else "DRAW"`, collapsing **both** `"lost"` and
`"draw"` into `DRAW`. The smallest path is: split that mapping (`"lost"` → `DEVIL_WINS` with a
certificate; `"draw"` → `DRAW`) and add an independent Judge re-verification.

## Answers

**1. What outcomes currently exist?**
User-facing (`fight.py`): `GOD_WINS`, `DRAW`. Internal (`game.Outcome.outcome`): `won`, `lost`,
`draw`. `DEVIL_WINS` appears only in a comment (`fight.py:173`); not represented anywhere.
(Note: `arena.py` has a separate `ArenaResult` with its own `GOD_WINS`/`DRAW`; the Prototype-1
path is the `neural_fraisse.fight`/`game` path, not arena.)

**2. Where are outcomes represented?**
`neural_fraisse/game.py` — `Outcome` dataclass (`outcome`, `nodes`, `decisions`, `trace`), set in
`play_game`. Mapped to the public string in `fight.run_fight` (`fight.py:96-100`) and rendered in
`fight.render` (`fight.py:156-173`).

**3. Where is GOD_WINS decided?**
`judge(structure, task)` (`game.py:55-60`) returns `"won"` when `classify(...)` (refute mode,
`semantic_search.py:53-55`) gives theory `ok` AND claim `failed`; `play_game` returns
`Outcome("won", …)`; `fight.py:97` maps `"won"` → `GOD_WINS`.

**4. Where is DRAW decided?**
Two paths today: (a) `_BudgetExhausted` → `Outcome("draw")` (`game.py:99-100`) = budget hit;
(b) `dfs` returns `None` → `Outcome("lost")` (`game.py:101-102`) = frontier closed. `fight.py:97`
maps **both** to `DRAW`. The conflation in (b) is precisely what hides `DEVIL_WINS`.

**5. Where should DEVIL_WINS be represented?**
In `fight.run_fight`: map `out.outcome == "lost"` → `DEVIL_WINS` (keep `"draw"` → `DRAW`); attach an
obstruction certificate; render it in `fight.render`. `game.Outcome` already distinguishes the two,
so no change to `play_game`'s control flow is required for the signal itself (a certificate builder is
added separately).

**6. Can the current evaluator express s³=id and s(a)≠a over n=2?**
Yes. `capsule._formula_to_clause` parses `forall x: s(s(s(x))) = x` (nested `Func` + `EqAtom`) and
`s(a) = a`; `run_devil_bounded` (`core/devil.py`) + `classify` evaluate them over a partial
Σ={E/2,s/1,a} structure at n=2. (`fight.build_task` already builds exactly this.)

**7. Can the partial-structure machinery enumerate all total completions on n=2?**
Yes, two ways. (a) `legal_semantic_edits` enumerates every unknown cell × value; `play_game`'s
obligation-ordered DFS with backtracking explores all *win-relevant* completions (cells not touched by
theory/goal are never branched, which is sound for the win test). (b) Independent brute force is
trivial at n=2: all `s:[2]→[2]` is 4 maps, `a∈{0,1}` is 2, so the relevant space is tiny. Exhaustion
is cheap.

**8. Smallest verified obstruction rule for Demo B.**
Enumerate every total `s:[n]→[n]` with `s³ = id`; for each, check the claim cell. On n=2 the only such
`s` is the identity (a transposition has order 2, so `s³ = s ≠ id`); hence `s(a)=a` for every model of
T, so "refute `s(a)=a`" has no model. The Judge verifies this by the finite enumeration above
(≤ nⁿ maps), independent of the search. This is the recommended certificate backend: it is a *verified*
obstruction, not "the DFS didn't find one."

**9. What should the obstruction certificate contain?**
A Judge-verifiable record:
- `outcome: DEVIL_WINS`, `domain_size n`, `depth k`, `budget b`;
- `theory`, `goal`;
- `method: "exhaustive bounded completion check"`, `budget_exhausted: false` (proves not a timeout);
- `completions_checked: N` and `winning_completions: 0`, with each completion classified
  (`theory_failed` / `dead_end` / no-goal-cell);
- a one-line mathematical reason (e.g. "s³=id forces s=id on n=2, so s(a)=a is unavoidable");
- everything re-checkable by re-running the finite enumeration.

**10. Files that would change in Gate R2.**
- `src/logical_gans/modelbuilder/neural_fraisse/fight.py` — map `"lost"`→`DEVIL_WINS`; build + render the certificate; return it from `run_fight`.
- `src/logical_gans/modelbuilder/neural_fraisse/game.py` (or a small new `obstruction.py` in the same package) — add `build_obstruction_certificate(task)` doing the independent exhaustive enumeration + Judge re-verification, and surface `budget_exhausted` cleanly.
- `examples/problems/cycle2_impossible_fight.json` — new (n=2, T={∀x s(s(s(x)))=x}, goal refute s(a)=a).
- `tests/test_devil_wins.py` — new.
- `core/` — expected **unchanged** (`run_devil_bounded` already suffices).

**11. Tests that prove DEVIL_WINS is real (not timeout).**
- `cycle2_impossible_fight.json` → `outcome == "DEVIL_WINS"` with a certificate present.
- Certificate has `budget_exhausted == false` (frontier closed, **not** a budget timeout).
- Independent exhaustive enumeration in the test confirms `winning_completions == 0` and that every
  total `s` with `s³=id` on n=2 satisfies `s(a)=a` (Judge-verified, not trusting the search).
- Contrast test: the *same* task run with a deliberately tiny budget yields `DRAW`
  (`budget_exhausted == true`), demonstrating that `DEVIL_WINS` and timeout are distinguished.
- Regression: `cycle3_fight.json` still yields `GOD_WINS` (the `"won"` path is untouched).

## Recommended R2 path (smallest safe)

1. Keep `play_game` as-is; its `"lost"` already means frontier-closed/budget-not-hit.
2. In `run_fight`, on `"lost"` call `build_obstruction_certificate(task)` (independent exhaustive
   enumeration that re-verifies no winning completion exists) and return `DEVIL_WINS` + certificate;
   on `"draw"` keep `DRAW`.
3. Render `Outcome: DEVIL_WINS` + `Obstruction certificate: …` in `fight.render`.
4. Add the n=2 problem file and `test_devil_wins.py` per Q11.

This touches only `neural_fraisse/` (+ one example + one test), needs no neural Devil, no training,
and yields a Judge-verified bounded obstruction certificate rather than a search timeout.

## Pass condition
This file exists and gives a concrete implementation path for Demo B (above). Gate R1 complete.
