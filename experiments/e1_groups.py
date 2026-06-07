from __future__ import annotations

import argparse
import json
from pathlib import Path

from logical_gans.modelbuilder.engine import run_group_episode
from logical_gans.modelbuilder.theories.groups import find_counterexample_all_groups_abelian


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results_groups.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    rows = []
    for kind, n in [("cyclic", 3), ("cyclic", 4), ("klein", 4), ("dihedral", 3)]:
        rows.append(run_group_episode(kind, n, k=2, budget=100, include_commutativity=False).to_dict())
        rows.append(run_group_episode(kind, n, k=2, budget=100, include_commutativity=True).to_dict())
    rows.append(find_counterexample_all_groups_abelian(max_order=8).to_dict())
    out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
