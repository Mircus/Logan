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

    # learned (neural) relation-builder commands
    p_mkd = sub.add_parser("make-relation-training-data",
                           help="mine Devil-witness training examples for a binary relation")
    p_mkd.add_argument("--theory", required=True)
    p_mkd.add_argument("--relation", required=True)
    p_mkd.add_argument("--n", type=int, required=True)
    p_mkd.add_argument("--samples", type=int, default=200)
    p_mkd.add_argument("--seed", type=int, default=0)
    p_mkd.add_argument("--k", type=int, default=None)
    p_mkd.add_argument("--budget", type=int, default=None)
    p_mkd.add_argument("--out", required=True)

    p_trn = sub.add_parser("train-relation-policy", help="train the relation policy net")
    p_trn.add_argument("--data", required=True)
    p_trn.add_argument("--n", type=int, required=True)
    p_trn.add_argument("--epochs", type=int, default=50)
    p_trn.add_argument("--lr", type=float, default=1e-3)
    p_trn.add_argument("--seed", type=int, default=0)
    p_trn.add_argument("--out", required=True)

    p_nb = sub.add_parser("neural-build-relation",
                          help="build a binary relation with the neural policy (Devil-verified)")
    p_nb.add_argument("--theory", required=True)
    p_nb.add_argument("--relation", required=True)
    p_nb.add_argument("--n", type=int, required=True)
    p_nb.add_argument("--model", required=True)
    p_nb.add_argument("--seed", default=None, help="open-world seed structure JSON")
    p_nb.add_argument("--k", type=int, default=None)
    p_nb.add_argument("--budget", type=int, default=None)
    p_nb.add_argument("--max-steps", type=int, default=50)

    p_mcts = sub.add_parser("mcts-relation",
                            help="MCTS over relation edits (optional neural priors), Devil-verified")
    p_mcts.add_argument("--theory", required=True)
    p_mcts.add_argument("--relation", required=True)
    p_mcts.add_argument("--n", type=int, required=True)
    p_mcts.add_argument("--seed", default=None, help="open-world seed structure JSON")
    p_mcts.add_argument("--model", default=None, help="optional policy net (uniform priors if omitted)")
    p_mcts.add_argument("--k", type=int, default=None)
    p_mcts.add_argument("--budget", type=int, default=None)
    p_mcts.add_argument("--rollouts", type=int, default=100)
    p_mcts.add_argument("--c-puct", type=float, default=1.5)
    p_mcts.add_argument("--llm-prior", default=None,
                        help="optional LLM prior hook, e.g. mock:first_true / mock:witness_match")

    p_ms = sub.add_parser("mcts-semantic",
                          help="signature-parametric MCTS over generic semantic edits")
    p_ms.add_argument("--theory", required=True)
    p_ms.add_argument("--n", type=int, required=True)
    p_ms.add_argument("--seed", default=None, help="open-world seed structure JSON")
    p_ms.add_argument("--k", type=int, default=None)
    p_ms.add_argument("--budget", type=int, default=None)
    p_ms.add_argument("--rollouts", type=int, default=200)
    p_ms.add_argument("--c-puct", type=float, default=1.5)
    p_ms.add_argument("--prior", default="obligation",
                      choices=["uniform", "obligation", "mock"])

    p_llm = sub.add_parser("llm-propose-relation",
                           help="ask a (mock) LLM for relation edits and validate them")
    p_llm.add_argument("--theory", required=True)
    p_llm.add_argument("--relation", required=True)
    p_llm.add_argument("--n", type=int, required=True)
    p_llm.add_argument("--seed", default=None, help="open-world seed structure JSON")
    p_llm.add_argument("--k", type=int, default=None)
    p_llm.add_argument("--budget", type=int, default=None)
    p_llm.add_argument("--mock", default="witness_match",
                       choices=["first_true", "first_false", "witness_match"])

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

    from .core.loader import (
        TheoryLoadError,
        load_claim,
        load_seed_open_world,
        load_structure,
        load_theory,
    )
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

        if args.cmd == "make-relation-training-data":
            from .learned.data import make_relation_training_examples, write_training_examples_jsonl

            theory = load_theory(args.theory)
            examples = make_relation_training_examples(
                theory, args.relation, args.n, args.samples,
                seed=args.seed, k=args.k, budget=args.budget,
            )
            write_training_examples_jsonl(examples, args.out)
            return _emit({"out": args.out, "examples": len(examples),
                          "relation": args.relation, "n": args.n})

        if args.cmd == "train-relation-policy":
            from .learned.train import train_relation_policy

            return _emit(train_relation_policy(
                args.data, args.out, args.n, epochs=args.epochs, lr=args.lr, seed=args.seed,
            ))

        if args.cmd == "neural-build-relation":
            from .learned.builder import neural_relation_build

            theory = load_theory(args.theory)
            seed_structure = (
                load_seed_open_world(args.seed, theory.signature) if args.seed else None
            )
            return _emit(neural_relation_build(
                theory, args.relation, args.n, args.model,
                seed_structure=seed_structure,
                k=args.k, budget=args.budget, max_steps=args.max_steps,
            ))

        if args.cmd == "mcts-relation":
            from .learned.mcts import mcts_relation_build

            theory = load_theory(args.theory)
            seed_structure = (
                load_seed_open_world(args.seed, theory.signature) if args.seed else None
            )
            llm_prior = None
            if args.llm_prior:
                if not args.llm_prior.startswith("mock:"):
                    parser.error("--llm-prior must look like mock:<strategy>")
                from .learned.mock_llm import mock_llm_prior
                llm_prior = mock_llm_prior(args.llm_prior.split(":", 1)[1])
            return _emit(mcts_relation_build(
                theory, args.relation, args.n, model_path=args.model,
                seed_structure=seed_structure, k=args.k, budget=args.budget,
                rollouts=args.rollouts, c_puct=args.c_puct, llm_prior=llm_prior,
            ))

        if args.cmd == "mcts-semantic":
            from .learned.generic_mcts import mcts_semantic_build
            from .learned.priors import MockLLMPrior, ObligationFirstPrior, UniformPrior

            theory = load_theory(args.theory)
            seed_structure = (
                load_seed_open_world(args.seed, theory.signature) if args.seed else None
            )
            provider = {"uniform": UniformPrior, "obligation": ObligationFirstPrior,
                        "mock": MockLLMPrior}[args.prior]()
            return _emit(mcts_semantic_build(
                theory, args.n, seed_structure=seed_structure, k=args.k, budget=args.budget,
                rollouts=args.rollouts, c_puct=args.c_puct, prior_hook=provider,
            ))

        if args.cmd == "llm-propose-relation":
            from .learned.llm_protocol import (
                build_llm_input,
                edit_to_dict,
                parse_llm_output,
                validate_llm_actions,
            )
            from .learned.actions import legal_relation_edits
            from .learned.mock_llm import mock_llm_json
            from .core.devil import run_devil_bounded
            from .core.partial_structure import PartialStructure

            theory = load_theory(args.theory)
            structure = (
                load_seed_open_world(args.seed, theory.signature) if args.seed
                else PartialStructure.empty(theory.signature, args.n)
            )
            result = run_devil_bounded(structure, theory.clauses, k=args.k, budget=args.budget)
            allowed = legal_relation_edits(structure, args.relation)
            llm_input = build_llm_input(theory, args.relation, structure, result.witness,
                                        k=args.k, budget=args.budget)
            text = mock_llm_json(args.mock, allowed, result.witness)
            output = parse_llm_output(text)
            plan = validate_llm_actions(output, allowed)
            return _emit({
                "mock_strategy": args.mock,
                "goal": llm_input.goal,
                "allowed_actions": [edit_to_dict(e) for e in allowed],
                "proposed_actions": [
                    {"kind": a.kind, "relation": a.relation,
                     "args": list(a.args) if a.args else None, "value": a.value}
                    for a in output.proposed_actions
                ],
                "explanation_ignored": output.explanation,
                "validation": {
                    "validated": [edit_to_dict(e) for e in plan.validated],
                    "rejected": plan.rejected,
                },
            })

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
