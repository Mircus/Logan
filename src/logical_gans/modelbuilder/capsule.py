"""UCMT countermodel proof capsule.

Reads one self-contained problem file (``examples/problems/*.json``), translates
its human-readable theory/claim formulas into the existing kernel atom/term JSON,
runs the existing Gate-2 neural-semantic ablation machinery (no rollouts), and
emits a UCMT-style certificate:

    A  ⊩_{k,b}  T        (A models T under the bounded Devil at depth k, budget b)
    A  ⊭_{k,b}  C        (A refutes the claim C under the same bound)

The kernel parser is NOT redesigned: the translator only converts the problem
file's formula strings into the {var/const/func/rel/eq} dicts that
``core.loader`` already understands, then hands them to ``parse_clause`` /
``parse_signature``. The search/verification reuse existing code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple


# --------------------------------------------------------------------------
# Tiny formula translator -> existing kernel JSON (no parser redesign).
# Grammar handled (exactly the UCMT P0 fragment used here):
#   formula  ::= [ "forall" vars ":" ] atom
#   atom     ::= term "=" term | Name "(" termlist ")"
#   term     ::= Name "(" termlist ")" | constant | variable
# --------------------------------------------------------------------------
def _split_top(text: str, sep: str) -> List[str]:
    """Split on `sep` only at paren depth 0."""
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def _application(text: str) -> Tuple[str, List[str]]:
    text = text.strip()
    i = text.index("(")
    name = text[:i].strip()
    inner = text[i + 1: text.rindex(")")]
    args = [a.strip() for a in _split_top(inner, ",") if a.strip()]
    return name, args


def _term(text: str, signature, variables) -> dict:
    text = text.strip()
    if "(" in text:
        name, args = _application(text)
        if name not in signature.functions:
            raise ValueError(f"unknown function symbol {name!r} in term {text!r}")
        return {"func": name, "args": [_term(a, signature, variables) for a in args]}
    if text in signature.constants:
        return {"const": text}
    if text in variables:
        return {"var": text}
    raise ValueError(f"symbol {text!r} is neither a declared constant nor a quantified variable")


def _atom(text: str, signature, variables) -> dict:
    eq = _split_top(text, "=")
    if len(eq) == 2:
        return {"eq": [_term(eq[0], signature, variables), _term(eq[1], signature, variables)]}
    name, args = _application(text)
    if name not in signature.relations:
        raise ValueError(f"unknown relation symbol {name!r} in atom {text!r}")
    return {"rel": name, "args": [_term(a, signature, variables) for a in args]}


def _formula_to_clause(text: str, name: str, signature) -> dict:
    text = text.strip()
    variables: List[str] = []
    body = text
    if text.lower().startswith("forall"):
        head, body = text[len("forall"):].split(":", 1)
        variables = [v.strip() for v in _split_top(head, ",") if v.strip()]
        body = body.strip()
    return {"name": name, "variables": variables, "premises": [],
            "conclusion": _atom(body, signature, variables)}


def translate_problem(problem: dict):
    """problem dict -> (Theory, Claim) using the existing kernel loader."""
    from .core.loader import parse_clause, parse_signature
    from .core.theory import Claim, Theory

    signature = parse_signature(problem["signature"])
    theory_clauses = [
        parse_clause(_formula_to_clause(f, f"axiom_{i}", signature))
        for i, f in enumerate(problem["theory"])
    ]
    theory = Theory(name=problem.get("name", "theory"), signature=signature, clauses=theory_clauses)
    claim_clause = parse_clause(_formula_to_clause(problem["claim"], "claim", signature))
    claim = Claim(name=problem.get("name", "claim"), clauses=[claim_clause])
    return theory, claim


def signature_summary(problem: dict) -> str:
    """e.g. 'E/2, s/1, a' from a problem signature block."""
    sig = problem.get("signature", {})
    parts = [f"{r['name']}/{r['arity']}" for r in sig.get("relations", [])]
    parts += [f"{f['name']}/{f['arity']}" for f in sig.get("functions", [])]
    parts += [c if isinstance(c, str) else c.get("name", "?") for c in sig.get("constants", [])]
    return ", ".join(parts)


# --------------------------------------------------------------------------
# Problem validation (used by `validate-problem` and before every capsule run).
# --------------------------------------------------------------------------
_KNOWN_GENERATORS = {"neural_semantic_mcts"}


def _arity_errors(signature, clauses) -> List[str]:
    from .core.atoms import EqAtom, RelAtom
    from .core.terms import Func

    errs: List[str] = []

    def check_term(t):
        if isinstance(t, Func):
            fn = signature.functions.get(t.name)
            if fn is None:
                errs.append(f"unknown function symbol {t.name!r}")
            elif len(t.args) != fn.arity:
                errs.append(f"function {t.name!r} used with arity {len(t.args)}, expected {fn.arity}")
            for a in t.args:
                check_term(a)

    def check_atom(a):
        if isinstance(a, RelAtom):
            rel = signature.relations.get(a.relation)
            if rel is None:
                errs.append(f"unknown relation symbol {a.relation!r}")
            elif len(a.args) != rel.arity:
                errs.append(f"relation {a.relation!r} used with arity {len(a.args)}, expected {rel.arity}")
            for t in a.args:
                check_term(t)
        elif isinstance(a, EqAtom):
            check_term(a.left)
            check_term(a.right)

    for c in clauses:
        for p in c.premises:
            check_atom(p)
        check_atom(c.conclusion)
    return errs


def validate_problem(problem: dict) -> List[str]:
    """Return a list of human-readable errors; empty list means the file is OK."""
    if not isinstance(problem.get("signature"), dict):
        return ["missing or malformed 'signature' object"]

    errors: List[str] = []
    n = problem.get("domain_size")
    if not isinstance(n, int) or n <= 0:
        errors.append("'domain_size' must be a positive integer")
    bound = problem.get("bound")
    if not isinstance(bound, dict):
        errors.append("missing 'bound' object with 'depth' and 'budget'")
    else:
        if not isinstance(bound.get("depth"), int):
            errors.append("missing/invalid 'bound.depth' (integer)")
        if not isinstance(bound.get("budget"), int):
            errors.append("missing/invalid 'bound.budget' (integer)")
    if not isinstance(problem.get("theory"), list) or not problem.get("theory"):
        errors.append("'theory' must be a non-empty list of formula strings")
    if not isinstance(problem.get("claim"), str):
        errors.append("'claim' must be a single formula string")
    kind = problem.get("generator", {}).get("kind", "neural_semantic_mcts")
    if kind not in _KNOWN_GENERATORS:
        errors.append(f"unrecognized generator.kind {kind!r} (known: {sorted(_KNOWN_GENERATORS)})")

    # Parse the formulas (symbol existence) and check arities against the signature.
    if isinstance(problem.get("theory"), list) and isinstance(problem.get("claim"), str):
        try:
            theory, claim = translate_problem(problem)
            errors.extend(_arity_errors(theory.signature, list(theory.clauses) + list(claim.clauses)))
        except Exception as e:  # translator/loader rejects unknown symbols, bad syntax
            errors.append(f"theory/claim parse error: {e}")
    return errors


# --------------------------------------------------------------------------
# Capsule runner -- reuses the committed Gate-2 ablation functions.
# --------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _structure_has_all_kinds(struct) -> bool:
    if not struct:
        return False
    has_rel = any(v in ("true", "false") for v in struct["relations"].values())
    has_fn = any(v is not None for v in struct["functions"].values())
    has_const = any(v is not None for v in struct["constants"].values())
    return has_rel and has_fn and has_const


def run_countermodel_capsule(problem_path, out_path=None, epochs: int = 150, auto_train=None) -> dict:
    from .core.devil import run_devil_bounded
    from .core.partial_structure import PartialStructure
    from .learned.priors import NeuralSemanticPrior, ObligationFirstPrior, UniformPrior
    from .learned.semantic_search import build_training_examples, tree_policy_search
    from .learned.semantic_training import train_semantic_policy, write_semantic_training_jsonl

    problem = json.loads(Path(problem_path).read_text(encoding="utf-8"))
    errors = validate_problem(problem)
    if errors:
        raise ValueError("invalid problem file:\n  - " + "\n  - ".join(errors))
    theory, claim = translate_problem(problem)

    n = int(problem["domain_size"])
    k = int(problem["bound"]["depth"])
    b = int(problem["bound"]["budget"])
    rollouts = int(problem.get("generator", {}).get("rollouts", b))
    name = problem.get("name", "problem")
    root = _repo_root()
    data_path = root / "results" / "training" / f"{name}_semantic.jsonl"
    model_path = root / "models" / f"{name}_semantic_policy.pt"

    # auto_train: CLI flag wins, else the problem file's generator.auto_train, default True.
    if auto_train is None:
        auto_train = bool(problem.get("generator", {}).get("auto_train", True))

    # 1. Optionally mine oracle traces and train a SMALL problem-specific prior.
    #    (Not a universal pretrained model.) Falls back gracefully if the oracle
    #    finds no refuting trajectory for this problem.
    uses_neural = False
    neural_prior = ObligationFirstPrior()
    train_meta = {"examples": 0, "final_loss": None}
    if auto_train:
        examples = build_training_examples(theory, claim, n)
        if examples:
            write_semantic_training_jsonl(examples, data_path)
            train_meta = train_semantic_policy(str(data_path), str(model_path),
                                               epochs=epochs, seed=0)
            neural_prior = NeuralSemanticPrior(str(model_path))
            uses_neural = True

    # 2. three arms, identical budget/seed, rollouts disabled (tree policy only).
    arms = {
        "uniform": UniformPrior(),
        "obligation_first": ObligationFirstPrior(),
        "neural": neural_prior,
    }
    results = {nm: tree_policy_search(theory, claim, n, prior, mode="refute",
                                      k=None, devil_budget=None, search_budget=rollouts, c_puct=2.0)
               for nm, prior in arms.items()}
    neural, uni, obl = results["neural"], results["uniform"], results["obligation_first"]

    # 3. UCMT relations verified *literally* at (k, b) with the bounded Devil.
    structure_json = neural["structure"]
    theory_relation = claim_relation = None
    theory_bounded = claim_bounded = None
    if neural["success"] and structure_json is not None:
        A = PartialStructure.empty(theory.signature, n)
        for key, val in structure_json["relations"].items():
            inside = key[key.index("(") + 1: key.rindex(")")]
            A.set_relation(key[: key.index("(")], tuple(int(x) for x in inside.split(",")),
                           gate_truth(val))
        for key, val in structure_json["functions"].items():
            if val is not None:
                inside = key[key.index("(") + 1: key.rindex(")")]
                A.set_function(key[: key.index("(")], tuple(int(x) for x in inside.split(",")), val)
        for cname, val in structure_json["constants"].items():
            if val is not None:
                A.set_constant(cname, val)
        theory_bounded = run_devil_bounded(A, theory.clauses, k=k, budget=b).status
        claim_bounded = run_devil_bounded(A, claim.clauses, k=k, budget=b).status
        theory_relation = f"A ⊩_{{{k},{b}}} T"      # ⊩
        claim_relation = f"A ⊭_{{{k},{b}}} C"        # ⊭

    def worse(base) -> bool:
        return (not base["success"]) or (
            neural["success"] and base["nodes_evaluated"] >= 3 * neural["nodes_evaluated"])

    accepted = bool(
        uses_neural
        and neural["success"]
        and neural["success_via"] == "guided_tree_policy"
        and theory_bounded == "ok"
        and claim_bounded == "failed"
        and neural["witness"] is not None
        and _structure_has_all_kinds(structure_json)
        and worse(uni) and worse(obl)
    )

    certificate = {
        "problem": {
            "name": name,
            "signature": signature_summary(problem),
            "theory": list(problem.get("theory", [])),
            "claim": problem.get("claim", ""),
            "n": n,
        },
        "status": "refuted" if accepted else ("refuted_unverified" if neural["success"] else "not_found"),
        "certificate": {
            "model_relation": theory_relation,
            "counterclaim_relation": claim_relation,
            "depth": k,
            "budget": b,
            "theory_status": "satisfied" if theory_bounded == "ok" else theory_bounded,
            "claim_status": "failed" if claim_bounded == "failed" else claim_bounded,
        },
        "generator": {
            "kind": problem.get("generator", {}).get("kind", "neural_semantic_mcts"),
            "auto_train": bool(auto_train),
            "uses_neural_policy": bool(uses_neural and neural["success"]),
            "random_rollouts": problem.get("generator", {}).get("random_rollouts", "disabled"),
            "success_via": neural["success_via"],
            "note": "auto-train fits a small problem-specific neural prior; it is NOT a universal pretrained model",
        },
        "structure": structure_json or {},
        "witness": neural["witness"] or {},
        "ablation": {
            "uniform": {"success": uni["success"], "nodes_evaluated": uni["nodes_evaluated"]},
            "obligation_first": {"success": obl["success"], "nodes_evaluated": obl["nodes_evaluated"]},
            "neural": {"success": neural["success"], "nodes_evaluated": neural["nodes_evaluated"]},
        },
        "accepted": accepted,
        "_training": {"examples": train_meta["examples"], "final_loss": train_meta["final_loss"]},
    }

    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(certificate, indent=2, ensure_ascii=False), encoding="utf-8")

    return certificate


def gate_truth(value: str):
    from .core.types import Truth
    return Truth(value)


# --------------------------------------------------------------------------
# Human-readable certificate.
# --------------------------------------------------------------------------
def render_certificate(cert: dict) -> str:
    p, s, abl, g, c = (cert["problem"], cert["structure"], cert["ablation"],
                       cert["generator"], cert["certificate"])

    lines = [
        "LOGAN / UCMT countermodel certificate",
        "",
        "Problem:",
        f"  Σ = {{{p['signature']}}}",
        f"  T = {{{'; '.join(p['theory'])}}}",
        f"  C = {p['claim']}",
        f"  n = {p['n']}",
        "",
        "Bound:",
        f"  k = {c['depth']}",
        f"  b = {c['budget']}",
        "",
        "Generator:",
        f"  {g['kind']}",
        f"  random_rollouts = {g['random_rollouts']}",
        f"  auto_train = {g['auto_train']}  (problem-specific prior, not a universal model)",
        "",
        "Result:",
    ]
    if cert["accepted"]:
        funcs = ", ".join(f"{k}={v}" for k, v in s.get("functions", {}).items() if v is not None)
        consts = ", ".join(f"{k}={v}" for k, v in s.get("constants", {}).items() if v is not None)
        true_rels = ", ".join(f"{k}=true" for k, v in s.get("relations", {}).items() if v == "true")
        lines += [
            f"  {c['model_relation']}",
            f"  {c['counterclaim_relation']}",
            "",
            "Generated structure:",
            f"  constants: {consts}",
            f"  functions: {funcs}",
            f"  relations: {true_rels}",
            "",
            "Witness:",
            f"  {cert['witness'].get('message', '(no witness)')}",
        ]
    else:
        lines += [
            f"  no verified countermodel at (k={c['depth']}, b={c['budget']}) -> status={cert['status']}",
            "  (theory/claim parsed and ran; the bounded search did not return a verified refutation)",
        ]
    lines += [
        "",
        "Ablation:",
        f"  uniform: {'succeeded' if abl['uniform']['success'] else 'failed'} at {abl['uniform']['nodes_evaluated']} nodes",
        f"  obligation_first: {'succeeded' if abl['obligation_first']['success'] else 'failed'} at {abl['obligation_first']['nodes_evaluated']} nodes",
        f"  neural: {'succeeded' if abl['neural']['success'] else 'failed'} at {abl['neural']['nodes_evaluated']} nodes",
    ]
    return "\n".join(lines)
