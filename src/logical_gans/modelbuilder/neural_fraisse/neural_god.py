"""Gate R8: a trainable neural God policy over legal Builder replies.

`NeuralGodPolicy` is the God-side analogue of `NeuralDevilPolicy`: it scores the
*already-legal* Builder replies (from the existing `legal_replies` enumerator)
with a trainable MLP and is updated by REINFORCE under the symbolic Judge. It does
NOT invent replies.

This is a NEW, clearly-named trainable policy -- it is deliberately distinct from
the existing frozen `FraisseNeuralPrior`/`NeuralBuilder` (which stays the default
fight God and is not un-frozen). Nothing here is a re-label of the frozen prior.
"""
from __future__ import annotations

import zlib
from typing import List, Optional

import torch
import torch.nn as nn

from ..core.types import Truth
from ..learned.semantic_actions import SetConstant, SetFunction, SetRelation
from .game import DevilMove

_EDIT_KINDS = ("SetRelation", "SetFunction", "SetConstant")
_MOVE_KINDS = ("ChallengeClauseInstance", "ChallengeGoalCell")
_N_BUCKETS = 8

# edit-kind one-hot (3) + symbol/args bucket one-hots (2*8) + move-kind one-hot (2)
# + 6 scalars (value_norm, is_true, domain, assigned, num_replies, arity)
REPLY_FEATURE_DIM = len(_EDIT_KINDS) + 2 * _N_BUCKETS + len(_MOVE_KINDS) + 6


def _bucket(text: Optional[str]) -> int:
    if not text:
        return 0
    return zlib.crc32(text.encode("utf-8")) % _N_BUCKETS


def _assigned_cells(structure) -> int:
    n = sum(1 for v in structure.constants.values() if v is not None)
    n += sum(1 for v in structure.functions.values() if v is not None)
    n += sum(1 for v in structure.relations.values() if v is not Truth.UNKNOWN)
    return n


def _edit_kind(edit) -> str:
    if isinstance(edit, SetRelation):
        return "SetRelation"
    if isinstance(edit, SetFunction):
        return "SetFunction"
    return "SetConstant"


def reply_features(structure, move: DevilMove, edit, num_replies: int) -> List[float]:
    """Encode a (structure, challenge, reply) triple into a fixed-length vector.

    Distinct legal replies must map to distinct vectors (e.g. s(0)=0 vs s(0)=1,
    E(0,1)=true vs =false), so the edit value is encoded explicitly."""
    feats = [0.0] * REPLY_FEATURE_DIM
    i = 0
    kind = _edit_kind(edit)
    for k in _EDIT_KINDS:
        feats[i] = 1.0 if kind == k else 0.0
        i += 1
    args = tuple(getattr(edit, "args", ()) or ())
    sym_b = _bucket(getattr(edit, "symbol", None))
    for b in range(_N_BUCKETS):
        feats[i + b] = 1.0 if b == sym_b else 0.0
    i += _N_BUCKETS
    arg_b = _bucket(str(args))
    for b in range(_N_BUCKETS):
        feats[i + b] = 1.0 if b == arg_b else 0.0
    i += _N_BUCKETS
    for k in _MOVE_KINDS:
        feats[i] = 1.0 if move.kind == k else 0.0
        i += 1
    # scalars
    n = max(1, len(structure.domain))
    value = getattr(edit, "value", None)
    is_true = 1.0 if (isinstance(value, Truth) and value is Truth.TRUE) else 0.0
    if isinstance(value, Truth):
        value_norm = is_true
    else:
        value_norm = (float(value) / n) if value is not None else -1.0
    feats[i] = value_norm; i += 1
    feats[i] = is_true; i += 1
    feats[i] = len(structure.domain) / 10.0; i += 1
    feats[i] = _assigned_cells(structure) / 10.0; i += 1
    feats[i] = num_replies / 10.0; i += 1
    feats[i] = len(args) / 4.0; i += 1
    return feats


class NeuralGodPolicy(nn.Module):
    """Trainable MLP scorer over legal Builder replies."""

    def __init__(self, hidden: int = 16, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.mlp = nn.Sequential(
            nn.Linear(REPLY_FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def score_replies(self, structure, move: DevilMove, replies: List) -> torch.Tensor:
        if not replies:
            return torch.empty(0)
        x = torch.tensor([reply_features(structure, move, e, len(replies)) for e in replies],
                         dtype=torch.float32)
        return self.mlp(x).squeeze(-1)

    def choose(self, structure, move: DevilMove, replies: List, sample: bool = False):
        if not replies:
            return None
        scores = self.score_replies(structure, move, replies)
        if sample:
            idx = int(torch.distributions.Categorical(logits=scores).sample().item())
        else:
            idx = int(torch.argmax(scores).item())
        return replies[idx]


class NeuralGodBuilder:
    """Builder adapter so a NeuralGodPolicy can drive `play_game`: ranks the legal
    replies by the policy's scores. Scores are detached for the (discrete) ranking;
    the policy is trained separately via REINFORCE on a sampled reply (full grad)."""

    def __init__(self, policy: Optional[NeuralGodPolicy] = None):
        self.policy = policy if policy is not None else NeuralGodPolicy()

    def order(self, structure, move, replies):
        if not replies:
            return list(replies)
        scores = self.policy.score_replies(structure, move, replies).detach().tolist()
        order = sorted(range(len(replies)), key=lambda j: -scores[j])
        return [replies[j] for j in order]
