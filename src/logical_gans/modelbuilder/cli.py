"""CLI for the bounded partial-model kernel.

Generic, data-driven commands operate on JSON theory packs:
    synthesize --theory T.json --n N [--policy P]
    check      --theory T.json --structure S.json
    refute     --theory T.json --claim C.json (--n N | --structure S.json) [--policy P]

Convenience wrappers (synthesize-preorder, synthesize-semigroup,
refute-preorder-antisymmetry) call the same generic machinery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_POLICY_CHOICES = ["sparse_horn", "maximal_horn"]


def _emit(d: dict) -> int:
    print(json.dumps(d, indent=2))
    return 0


def _repo_root() -> Path:
    # cli.py -> modelbuilder -> logical_gans -> src -> <repo root>
    return Path(__file__).resolve().parents[3]


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(prog="logan-modelbuilder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_syn = sub.add_parser("synthesize", help="synthesize a model of a JSON theory")
    p_syn.add_argument("--theory", required=True)
    p_syn.add_argument("--n", type=int, required=True)
    p_syn.add_argument("--policy", default="sparse_horn", choices=_POLICY_CHOICES)
    p_syn.add_argument("--k", type=int, default=None, help="logical/term depth bound")
    p_syn.add_argument("--budget", type=int, default=None, help="Devil challenge-instance budget")

    p_chk = sub.add_parser("check", help="check a JSON structure against a JSON theory")
    p_chk.add_argument("--theory", required=True)
    p_chk.add_argument("--structure", required=True)

    p_search = sub.add_parser("search", help="backtracking search for a model of a JSON theory")
    p_search.add_argument("--theory", required=True)
    p_search.add_argument("--n", type=int, required=True)
    p_search.add_argument("--max-nodes", type=int, default=10000)
    p_search.add_argument("--k", type=int, default=None, help="logical/term depth bound")
    p_search.add_argument("--budget", type=int, default=None, help="Devil challenge-instance budget")

    p_ref = sub.add_parser("refute", help="refute a JSON claim with a model of a JSON theory")
    p_ref.add_argument("--theory", required=True)
    p_ref.add_argument("--claim", required=True)
    p_ref.add_argument("--n", type=int, default=None)
    p_ref.add_argument("--structure", default=None)
    p_ref.add_argument("--policy", default="sparse_horn", choices=_POLICY_CHOICES)

    # convenience wrappers
    p_pre = sub.add_parser("synthesize-preorder", help="wrapper: synthesize a preorder")
    p_pre.add_argument("--n", type=int, default=3)
    p_pre.add_argument("--policy", default="sparse_horn", choices=_POLICY_CHOICES)
    p_sg = sub.add_parser("synthesize-semigroup", help="wrapper: synthesize a semigroup")
    p_sg.add_argument("--n", type=int, default=1)
    p_sg.add_argument("--policy", default="sparse_horn", choices=_POLICY_CHOICES)
    sub.add_parser("refute-preorder-antisymmetry",
                   help="wrapper: refute antisymmetry with a 2-element total preorder")

    args = parser.parse_args(argv)

    from .core.loader import TheoryLoadError, load_claim, load_structure, load_theory
    from .core.policy import get_policy
    from .core.runner import check, refute, synthesize

    try:
        if args.cmd == "synthesize":
            theory = load_theory(args.theory)
            res = synthesize(theory, args.n, get_policy(args.policy), k=args.k, budget=args.budget)
            return _emit({"status": res.status, "n": args.n, "k": args.k, "budget": args.budget,
                          "policy": res.policy, "structure": res.structure.to_json(),
                          "trace": res.trace})

        if args.cmd == "check":
            theory = load_theory(args.theory)
            structure = load_structure(args.structure, theory.signature)
            return _emit(check(theory, structure))

        if args.cmd == "search":
            from .core.backtracking import backtracking_generate

            theory = load_theory(args.theory)
            res = backtracking_generate(theory, args.n, k=args.k, budget=args.budget,
                                        max_nodes=args.max_nodes)
            return _emit({"status": res.status, "n": args.n, "k": args.k, "budget": args.budget,
                          "max_nodes": args.max_nodes, "nodes": res.nodes,
                          "structure": res.structure.to_json(), "trace": res.trace})

        if args.cmd == "refute":
            if args.n is None and args.structure is None:
                parser.error("refute requires --n or --structure")
            theory = load_theory(args.theory)
            claim = load_claim(args.claim)
            structure = (
                load_structure(args.structure, theory.signature)
                if args.structure else None
            )
            out = refute(theory, claim, n=args.n, structure=structure,
                         policy=get_policy(args.policy))
            out["n"] = args.n
            out["policy"] = args.policy
            return _emit(out)

        if args.cmd == "synthesize-preorder":
            theory = load_theory(_repo_root() / "examples" / "theories" / "preorder.json")
            res = synthesize(theory, args.n, get_policy(args.policy))
            return _emit({"status": res.status, "policy": res.policy,
                          "structure": res.structure.to_json(), "trace": res.trace})

        if args.cmd == "synthesize-semigroup":
            theory = load_theory(_repo_root() / "examples" / "theories" / "semigroup.json")
            res = synthesize(theory, args.n, get_policy(args.policy))
            return _emit({"status": res.status, "policy": res.policy,
                          "structure": res.structure.to_json(), "trace": res.trace})

        if args.cmd == "refute-preorder-antisymmetry":
            root = _repo_root() / "examples"
            theory = load_theory(root / "theories" / "preorder.json")
            claim = load_claim(root / "claims" / "antisymmetry.json")
            return _emit(refute(theory, claim, n=2, policy=get_policy("maximal_horn")))
    except TheoryLoadError as e:
        parser.error(str(e))

    parser.error("unreachable command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
