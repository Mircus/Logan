"""CLI for the bounded partial-model kernel."""
from __future__ import annotations

import argparse
import json


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="logan-modelbuilder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("synthesize-preorder", help="fill an unknown preorder of size n")
    p_pre.add_argument("--n", type=int, default=3)

    sub.add_parser(
        "refute-preorder-antisymmetry",
        help="produce a 2-element preorder refuting antisymmetry",
    )

    p_sg = sub.add_parser("synthesize-semigroup", help="fill an unknown semigroup of size n")
    p_sg.add_argument("--n", type=int, default=1)

    args = parser.parse_args(argv)

    if args.cmd == "synthesize-preorder":
        from .core.generator import generate
        from .examples.preorder import empty_preorder_structure, preorder_clauses

        res = generate(empty_preorder_structure(args.n), preorder_clauses())
        print(json.dumps(
            {"status": res.status, "structure": res.structure.to_json(), "trace": res.trace},
            indent=2,
        ))
        return 0

    if args.cmd == "refute-preorder-antisymmetry":
        from .examples.preorder import antisymmetry_refutation

        print(json.dumps(antisymmetry_refutation(), indent=2))
        return 0

    if args.cmd == "synthesize-semigroup":
        from .core.generator import generate
        from .examples.semigroup import empty_semigroup_structure, semigroup_clauses

        res = generate(empty_semigroup_structure(args.n), semigroup_clauses())
        print(json.dumps(
            {"status": res.status, "structure": res.structure.to_json(), "trace": res.trace},
            indent=2,
        ))
        return 0

    parser.error("unreachable command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
