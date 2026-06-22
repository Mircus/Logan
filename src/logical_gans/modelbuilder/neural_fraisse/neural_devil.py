"""Gate R4: a neural Devil policy that *scores already-legal* Devil moves.

The neural Devil never invents logical challenges. It receives a GameState and the
list of legal DevilMoves produced by the symbolic enumerator
(`players.enumerate_legal_devil_moves`), encodes each (state, move) pair into a
small feature vector, scores them with a trainable MLP, and returns one move from
that exact list (argmax by default, optional sampling).

This gate adds only the policy object (+ a thin wrapper Devil) and tests. No
training loop, no adversarial training, no GAN language, no CLI demo.
"""
from __future__ import annotations

import zlib
from typing import List, Optional

import torch
import torch.nn as nn

from .game import DevilMove, GameState
from .players import enumerate_legal_devil_moves

_KINDS = ("ChallengeClauseInstance", "ChallengeGoalCell")
_TARGET_KINDS = ("relation", "function", "constant")
_N_SYMBOL_BUCKETS = 8

# kind one-hot (2) + target-kind one-hot (3) + symbol bucket one-hot (8) + 4 scalars
FEATURE_DIM = len(_KINDS) + len(_TARGET_KINDS) + _N_SYMBOL_BUCKETS + 4


def _symbol_bucket(symbol: Optional[str]) -> int:
    if not symbol:
        return 0
    return zlib.crc32(symbol.encode("utf-8")) % _N_SYMBOL_BUCKETS


def _assigned_cells(structure) -> int:
    from ..core.types import Truth
    n = sum(1 for v in structure.constants.values() if v is not None)
    n += sum(1 for v in structure.functions.values() if v is not None)
    n += sum(1 for v in structure.relations.values() if v is not Truth.UNKNOWN)
    return n


def move_features(state: GameState, move: DevilMove, num_moves: int) -> List[float]:
    """Encode a (GameState, DevilMove) pair into a fixed-length feature vector."""
    feats = [0.0] * FEATURE_DIM
    i = 0
    # move kind one-hot
    for k in _KINDS:
        feats[i] = 1.0 if move.kind == k else 0.0
        i += 1
    # target cell kind one-hot
    cell = move.target_cell or {}
    for tk in _TARGET_KINDS:
        feats[i] = 1.0 if cell.get("kind") == tk else 0.0
        i += 1
    # target symbol hashed bucket one-hot
    bucket = _symbol_bucket(cell.get("symbol"))
    for b in range(_N_SYMBOL_BUCKETS):
        feats[i] = 1.0 if b == bucket else 0.0
        i += 1
    # scalars: domain size, assigned cells, pending obligations, target arity
    args = cell.get("args") or []
    feats[i] = float(len(state.structure.domain)); i += 1
    feats[i] = float(_assigned_cells(state.structure)); i += 1
    feats[i] = float(num_moves); i += 1
    feats[i] = float(len(args)); i += 1
    return feats


class NeuralDevilPolicy(nn.Module):
    """Trainable MLP scorer over legal Devil moves."""

    def __init__(self, hidden: int = 16, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def score_moves(self, state: GameState, moves: List[DevilMove]) -> torch.Tensor:
        """Return a score tensor of shape [len(moves)]."""
        if not moves:
            return torch.empty(0)
        x = torch.tensor([move_features(state, m, len(moves)) for m in moves],
                         dtype=torch.float32)
        return self.mlp(x).squeeze(-1)

    def choose(self, state: GameState, moves: List[DevilMove],
               sample: bool = False) -> Optional[DevilMove]:
        """Return one move from `moves` (argmax by default; optional sampling)."""
        if not moves:
            return None
        scores = self.score_moves(state, moves)
        if sample:
            idx = int(torch.distributions.Categorical(logits=scores).sample().item())
        else:
            idx = int(torch.argmax(scores).item())
        return moves[idx]


class NeuralActiveDevil:
    """Wrapper Devil: enumerate legal moves symbolically, then let the neural
    policy pick one. Same `choose_move(structure, task)` interface as
    SymbolicActiveDevil, but NOT wired into the default fight path."""

    def __init__(self, policy: Optional[NeuralDevilPolicy] = None, sample: bool = False):
        self.policy = policy if policy is not None else NeuralDevilPolicy()
        self.sample = sample

    def choose_move(self, structure, task) -> Optional[DevilMove]:
        state = GameState(structure, task)
        moves = enumerate_legal_devil_moves(state)
        if not moves:
            return None
        return self.policy.choose(state, moves, sample=self.sample)
