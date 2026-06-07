from __future__ import annotations

import argparse
import json
from typing import Any

from .engine import run_group_episode
from .theories.groups import find_counterexample_all_groups_abelian


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="logan-modelbuilder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-group", help="build and challenge a finite group candidate")
    p_build.add_argument("--kind", choices=["cyclic", "klein", "dihedral"], default="cyclic")
    p_build.add_argument("--n", type=int, default=4, help="cyclic order or dihedral m parameter")
    p_build.add_argument("--k", type=int, default=2)
    p_build.add_argument("--budget", type=int, default=100)
    p_build.add_argument("--abelian", action="store_true", help="also challenge commutativity")

    p_cex = sub.add_parser("counterexample", help="find a LOGAN counterexample certificate")
    p_cex.add_argument("--claim", choices=["all_groups_abelian"], required=True)
    p_cex.add_argument("--max-order", type=int, default=8)

    args = parser.parse_args(argv)
    if args.cmd == "build-group":
        result = run_group_episode(args.kind, args.n, k=args.k, budget=args.budget, include_commutativity=args.abelian)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if args.cmd == "counterexample":
        if args.claim == "all_groups_abelian":
            cert = find_counterexample_all_groups_abelian(args.max_order)
            print(json.dumps(cert.to_dict(), indent=2))
            return 0
    parser.error("unreachable command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
