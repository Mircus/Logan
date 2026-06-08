# LOGAN-ModelBuilder output schema (v0.1-alpha)

All CLI commands print a single JSON object. This documents what is currently
emitted and stable for `v0.1-alpha`. Fields not listed for a command are not
emitted by that command.

## Unified field reference

| field       | meaning                                              | synthesize | search | check | refute |
|-------------|------------------------------------------------------|:----------:|:------:|:-----:|:------:|
| `status`    | outcome (see per-command values)                     | ✓ | ✓ | ✓ | ✓ |
| `n`         | domain size requested                                | ✓ | ✓ |   | ✓\*  |
| `k`         | logical/term depth bound (null = unbounded)          | ✓ | ✓ |   |       |
| `budget`    | Devil challenge-instance budget (null = unbounded)   | ✓ | ✓ |   |       |
| `max_nodes` | DFS search-effort bound                              |   | ✓ |   |       |
| `nodes`     | search nodes actually explored                       |   | ✓ |   |       |
| `policy`    | builder policy used                                  | ✓ |   |   | ✓ |
| `structure` | the (partial) structure (see below)                  | ✓ | ✓ | ✓ | ✓ |
| `witness`   | a failing/blocking witness, or null                  |   |   | ✓ | ✓ |
| `trace`     | ordered list of trace events                         | ✓ | ✓ |   |   |

\* `n` is the requested domain size, or `null` when `refute` was given a
loaded `--structure` instead of `--n`.

## Per-command `status` values

- `synthesize` / `search`: `satisfied` | `unsat` | `unknown`
- `check`: `satisfied` | `unknown` | `failed`
- `refute`: `refuted` | `not_refuted` | `unknown` | `model_<status>`
  (`model_unknown` / `model_unsat` when no model of the theory could be obtained)

## structure

```json
{
  "domain": [0, 1, 2],
  "constants": {"e": 0},
  "relations": {"R(0,0)": "true", "R(0,1)": "false"},
  "functions": {"mul(0,0)": 0}
}
```

- relation cells: `"true" | "false" | "unknown"`
- function/constant cells: integer element, or `null` for UNKNOWN

## witness

```json
{
  "clause_name": "antisymmetric",
  "assignment": {"x": 0, "y": 1},
  "premise_values": ["true", "true"],
  "conclusion_value": "false",
  "status": "failed",
  "touched_atoms": [ ... ],
  "message": "clause 'antisymmetric' FAILED under {'x': 0, 'y': 1}: ..."
}
```

- `status`: `failed` (a violation) or `unknown` (a blocking obligation)
- `conclusion_value`: `"true" | "false" | "unknown"`, or `null` when an
  UNKNOWN premise blocked the instance before the conclusion was reached
- `touched_atoms[]` entries are grounded atoms:
  - relation: `{"kind":"rel","relation":"R","args":["x","y"],"arg_values":[0,1],"truth":"true"}`
  - equality: `{"kind":"eq","left":"x","right":"y","left_value":0,"right_value":1,"truth":"false"}`

## trace events

Common: `{"event":"result","status":...}` ends every run (search adds `"nodes"`).

`synthesize` (monotone generator):
- `{"event":"start","policy":"sparse_horn"}`
- `{"step":i,"event":"challenge","status":...,"clause":...}`
- `{"event":"obligation","witness":{...}}`
- `{"event":"edit","action":"set_relation|set_function|set_constant", ...}`
- `{"event":"witness","witness":{...}}` (on `unsat`)

`search` (backtracking):
- `{"node":i,"depth":d,"event":"challenge","status":...,"clause":...}`
- `{"node":i,"depth":d,"event":"branch","cell":{...},"value":...}`
- `{"node":i,"event":"backtrack","cell":{...},"value":...}`
- `{"node":i,"event":"deadend","reason":"failed|no_branch"}`

Bounded metadata on `challenge` events (present only when `k` or `budget` is set):
`"k"`, `"budget"`, `"checked_instances"`, `"budget_exhausted"`, `"skipped_by_depth"`.

## Honesty note: `budget_exhausted`

When a `challenge` reports `"status":"ok"` together with
`"budget_exhausted": true`, the structure **survived the bounded attack** — it
was not falsified within `budget` challenges. This is **not** a claim of global
satisfaction. A `satisfied` result obtained this way means *survived bounded
scrutiny*, not *exhaustively verified*. Only an unbounded run (no `k`, no
`budget`) that ends with `unknown_*_cells` empty is exhaustively decided.
