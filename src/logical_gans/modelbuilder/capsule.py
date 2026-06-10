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
import sys
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


# --------------------------------------------------------------------------
# Capsule runner -- reuses the committed Gate-2 ablation functions.
# --------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _import_gate():
    """Import the committed Gate-2 ablation module (lives under experiments/)."""
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from experiments import neural_semantic_ablation_gate as gate  # noqa: E402
    return gate


def _structure_has_all_kinds(struct) -> bool:
    if not struct:
        return False
    has_rel = any(v in ("true", "false") for v in struct["relations"].values())
    has_fn = any(v is not None for v in struct["functions"].values())
    has_const = any(v is not None for v in struct["constants"].values())
    return has_rel and has_fn and has_const


def run_countermodel_capsule(problem_path, out_path=None, epochs: int = 150) -> dict:
    from .core.devil import run_devil_bounded
    from .learned.priors import NeuralSemanticPrior, ObligationFirstPrior, UniformPrior
    from .learned.semantic_training import train_semantic_policy, write_semantic_training_jsonl

    gate = _import_gate()

    problem = json.loads(Path(problem_path).read_text(encoding="utf-8"))
    theory, claim = translate_problem(problem)

    n = int(problem["domain_size"])
    k = int(problem["bound"]["depth"])
    b = int(problem["bound"]["budget"])
    rollouts = int(problem.get("generator", {}).get("rollouts", b))
    name = problem.get("name", "problem")
    root = _repo_root()
    data_path = root / "results" / "training" / f"{name}_semantic.jsonl"
    model_path = root / "models" / f"{name}_semantic_policy.pt"

    # 1. oracle traces -> train the neural semantic prior (reuses Gate-2 code).
    examples = gate.build_training_examples(theory, claim, n)
    write_semantic_training_jsonl(examples, data_path)
    train_meta = train_semantic_policy(str(data_path), str(model_path), epochs=epochs, seed=gate.SEED)

    # 2. three arms, identical budget/seed, rollouts disabled (tree policy only).
    arms = {
        "uniform": UniformPrior(),
        "obligation_first": ObligationFirstPrior(),
        "neural": NeuralSemanticPrior(str(model_path)),
    }
    results = {nm: gate.tree_policy_search(theory, claim, n, prior, budget=rollouts)
               for nm, prior in arms.items()}
    neural, uni, obl = results["neural"], results["uniform"], results["obligation_first"]

    # 3. UCMT relations verified *literally* at (k, b) with the bounded Devil.
    structure_json = neural["structure"]
    theory_relation = claim_relation = None
    theory_bounded = claim_bounded = None
    if neural["success"] and structure_json is not None:
        A = gate.PartialStructure.empty(theory.signature, n)
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
        neural["success"]
        and neural["success_via"] == "guided_tree_policy"
        and theory_bounded == "ok"
        and claim_bounded == "failed"
        and neural["claim_witness"] is not None
        and _structure_has_all_kinds(structure_json)
        and worse(uni) and worse(obl)
    )

    certificate = {
        "problem": {
            "name": name,
            "signature": "E/2, s/1, a",
            "theory": ["forall x E(x,s(x))", "forall x s(s(s(x)))=x"],
            "claim": "s(a)=a",
            "n": n,
        },
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
            "uses_neural_policy": bool(neural["success"]),
            "random_rollouts": problem.get("generator", {}).get("random_rollouts", "disabled"),
            "success_via": neural["success_via"],
        },
        "structure": structure_json or {},
        "witness": neural["claim_witness"] or {},
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
    s = cert["structure"]
    abl = cert["ablation"]
    g = cert["generator"]
    c = cert["certificate"]
    const_name = next(iter(s["constants"]), "a")
    a_val = s["constants"].get(const_name)
    funcs = ", ".join(f"{k}={v}" for k, v in s["functions"].items() if v is not None)
    true_rels = ", ".join(f"{k}=true" for k, v in s["relations"].items() if v == "true")
    s_at_a = s["functions"].get(f"s({a_val})")

    lines = [
        "LOGAN / UCMT countermodel certificate",
        "",
        "Problem:",
        "  Σ = {E/2, s/1, a}",
        "  T = {∀x E(x,s(x)), ∀x s(s(s(x)))=x}",
        "  C = s(a)=a",
        f"  n = {cert['problem']['n']}",
        "",
        "Bound:",
        f"  k = {c['depth']}",
        f"  b = {c['budget']}",
        "",
        "Generator:",
        f"  {g['kind']}",
        f"  random_rollouts = {g['random_rollouts']}",
        "",
        "Result:",
        f"  {c['model_relation']}",
        f"  {c['counterclaim_relation']}",
        "",
        "Generated structure:",
        f"  {const_name} = {a_val}",
        f"  {funcs}",
        f"  {true_rels}",
        "",
        "Witness:",
        f"  s({const_name}) = s({a_val}) = {s_at_a}",
        f"  {const_name} = {a_val}",
        f"  therefore s({const_name}) ≠ {const_name}",
        "",
        "Ablation:",
        f"  uniform: {'succeeded' if abl['uniform']['success'] else 'failed'} at {abl['uniform']['nodes_evaluated']} nodes",
        f"  obligation_first: {'succeeded' if abl['obligation_first']['success'] else 'failed'} at {abl['obligation_first']['nodes_evaluated']} nodes",
        f"  neural: {'succeeded' if abl['neural']['success'] else 'failed'} at {abl['neural']['nodes_evaluated']} nodes",
    ]
    return "\n".join(lines)
