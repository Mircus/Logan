"""Players: one active symbolic Devil + three Builders, plus the neural prior.

Devil is the SAME active symbolic Devil for the uniform / obligation-first /
neural-active methods, so those three differ ONLY in the Builder's reply ordering.
(Neural-passive uses free search instead; see benchmark.py.)
"""
from __future__ import annotations

import random
from itertools import product
from typing import List

from ..core.depth import clause_depth
from ..core.devil import DevilResult, _instance_status, _make_witness, run_devil_bounded
from ..core.obligations import extract_obligation
from ..learned.semantic_actions import SemanticEdit, SetFunction
from ..learned.semantic_token_features import TOKEN_DIM, edit_tokens
from .game import DevilMove, GameState

_CONTEXT_TYPE_INDEX = 4  # "context" slot in semantic_token_features._TYPES


def _cell(obl) -> dict:
    return {"kind": obl.kind, "symbol": obl.symbol,
            "args": list(obl.args) if obl.args else None}


def _move_key(kind, clause_name, assignment, cell):
    return (kind, clause_name, tuple(sorted((assignment or {}).items())),
            cell["kind"], cell["symbol"],
            tuple(cell["args"]) if cell["args"] else None)


def _moves_from_clauses(structure, clauses, k, kind, reason, seen) -> List[DevilMove]:
    """Every UNKNOWN (blockable, Builder-answerable) clause instance in the
    kernel's deterministic clause/lexicographic order, as DevilMoves. Uses the
    kernel's own instance evaluator so the ordering matches run_devil_bounded."""
    out: List[DevilMove] = []
    for clause in clauses:
        if k is not None and clause_depth(clause) > k:
            continue
        for values in product(structure.domain, repeat=len(clause.variables)):
            assignment = dict(zip(clause.variables, values))
            status, premise_truths, conclusion = _instance_status(clause, assignment, structure)
            if status != "unknown":          # "ok" (satisfied/vacuous) or "failed" (no fillable cell)
                continue
            witness = _make_witness(status, clause, assignment, premise_truths, conclusion, structure)
            obl = extract_obligation(structure, DevilResult(status, witness, clause, assignment))
            if obl is None:
                continue
            cell = _cell(obl)
            key = _move_key(kind, clause.name, assignment, cell)
            if key in seen:
                continue
            seen.add(key)
            out.append(DevilMove(kind, clause.name, dict(assignment), cell, reason, obligation=obl))
    return out


def enumerate_legal_devil_moves(state: GameState) -> List[DevilMove]:
    """All legal bounded Devil challenges at `state`, in the kernel's deterministic
    order. A legal move is an UNKNOWN clause instance the Builder can answer: a
    theory-defense obligation, or -- once T is satisfied, in refute mode -- an
    undetermined claim cell. The symbolic Devil picks the first of these."""
    structure, task = state.structure, state.task
    seen: set = set()
    moves = _moves_from_clauses(structure, task.theory.clauses, task.depth,
                                "ChallengeClauseInstance", "theory clause instance blocked", seen)
    if task.mode == "refute":
        dT = run_devil_bounded(structure, task.theory.clauses, k=task.depth, budget=None)
        if dT.status == "ok":
            moves += _moves_from_clauses(structure, task.goal.clauses, task.depth,
                                         "ChallengeGoalCell", "claim cell undetermined", seen)
    return moves


class SymbolicActiveDevil:
    """Chooses the next bounded challenge: a theory-defense obligation if the
    structure does not yet model T, else (refute mode) a claim cell to commit.

    Chooses the first move from the legal Devil-move enumeration (Gate R3)."""

    def choose_move(self, structure, task):
        moves = enumerate_legal_devil_moves(GameState(structure, task))
        return moves[0] if moves else None


class UniformBuilder:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def order(self, structure, move, replies):
        out = list(replies)
        self.rng.shuffle(out)
        return out


class ObligationFirstBuilder:
    """Deterministic symbolic heuristic: try the obligation's suggested values in
    order (obligation_edits already returns them in that order)."""

    def order(self, structure, move, replies):
        return list(replies)


class NeuralBuilder:
    def __init__(self, prior):
        self.prior = prior

    def order(self, structure, move, replies):
        scores = self.prior.score_edits(structure, move.obligation, replies)
        return sorted(replies, key=lambda e: -scores.get(e, 0.0))


# --------------------------------------------------------------------------
# Fraïssé featurizer + neural prior. Reuses the arity-parametric edit tokens and
# appends a small game-context token (no MAX_ARITY anywhere).
# --------------------------------------------------------------------------
def _ctx_token(scalars: List[float]) -> List[float]:
    onehot = [0.0, 0.0, 0.0, 0.0, 0.0]
    onehot[_CONTEXT_TYPE_INDEX] = 1.0
    return onehot + scalars + [0.0] * (TOKEN_DIM - 5 - 4)


def fraisse_edit_tokens(edit: SemanticEdit, structure, obligation):
    """edit_tokens(...) + one game-context token (collision / fixed-point)."""
    tokens = edit_tokens(edit, structure, obligation)
    if isinstance(edit, SetFunction):
        image = {v for (name, _), v in structure.functions.items()
                 if name == edit.symbol and v is not None}
        collision = 1.0 if edit.value in image else 0.0
        fixed = 1.0 if (len(edit.args) == 1 and edit.value == edit.args[0]) else 0.0
        tokens = tokens + [_ctx_token([collision, fixed, 0.0, 0.0])]
    else:
        tokens = tokens + [_ctx_token([0.0, 0.0, 0.0, 0.0])]
    return tokens


class FraisseNeuralPrior:
    """Token policy over fraisse_edit_tokens. Implements score_edits (active game)
    and the PriorProvider .score (free search / neural-passive)."""

    is_neural = True

    def __init__(self, model_path: str):
        import torch

        from ..learned.semantic_policy_net import TokenSemanticPolicyNet

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        token_dim = ckpt.get("token_dim", TOKEN_DIM)
        model = TokenSemanticPolicyNet(token_dim)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model = model

    def score_edits(self, structure, obligation, edits):
        import torch

        if not edits:
            return {}
        tensors = [torch.tensor(fraisse_edit_tokens(e, structure, obligation),
                                dtype=torch.float32) for e in edits]
        with torch.no_grad():
            scores = self._model(tensors)
        vals = scores.tolist() if scores.dim() else [float(scores)]
        return {e: float(v) for e, v in zip(edits, vals)}

    def score(self, structure, devil_result, allowed_edits):
        obl = None
        if devil_result is not None and getattr(devil_result, "status", None) == "unknown":
            obl = extract_obligation(structure, devil_result)
        return self.score_edits(structure, obl, allowed_edits)
