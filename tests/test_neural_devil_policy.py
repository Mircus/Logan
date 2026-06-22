"""Gate R4: neural Devil policy over legal moves.

A trainable neural Devil scores already-legal logical moves and returns one of
them. It never invents or returns an illegal/generated move.
"""
from pathlib import Path

import torch

from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.neural_fraisse.fight import build_task, load_problem
from logical_gans.modelbuilder.neural_fraisse.game import GameState
from logical_gans.modelbuilder.neural_fraisse.neural_devil import (
    FEATURE_DIM, NeuralActiveDevil, NeuralDevilPolicy)
from logical_gans.modelbuilder.neural_fraisse.players import (
    SymbolicActiveDevil, enumerate_legal_devil_moves)

ROOT = Path(__file__).resolve().parents[1]
CYCLE3 = ROOT / "examples" / "problems" / "cycle3_fight.json"
CYCLE2 = ROOT / "examples" / "problems" / "cycle2_impossible_fight.json"


def _state_and_moves(path):
    task = build_task(load_problem(path))
    structure = PartialStructure.empty(task.signature, task.domain_size)
    state = GameState(structure, task)
    return state, enumerate_legal_devil_moves(state)


def test_policy_has_trainable_parameters():
    policy = NeuralDevilPolicy()
    params = list(policy.parameters())
    assert len(params) > 0
    assert any(p.requires_grad for p in params)
    assert sum(p.numel() for p in params) > 0


def test_receives_nonempty_legal_move_list():
    _state, moves = _state_and_moves(CYCLE3)
    assert len(moves) > 0


def test_returns_one_of_the_provided_legal_moves():
    policy = NeuralDevilPolicy()
    for path in (CYCLE3, CYCLE2):
        state, moves = _state_and_moves(path)
        chosen = policy.choose(state, moves)
        assert any(chosen is m for m in moves)   # identity: an object from the list
        assert chosen in moves


def test_never_returns_illegal_or_generated_move():
    policy = NeuralDevilPolicy()
    state, moves = _state_and_moves(CYCLE3)
    # the chosen move object must be one of the exact provided objects (by identity)
    chosen = policy.choose(state, moves)
    assert any(chosen is m for m in moves)
    # sampling path also only ever returns from the list
    for _ in range(10):
        s = policy.choose(state, moves, sample=True)
        assert any(s is m for m in moves)


def test_scores_have_shape_num_legal_moves():
    policy = NeuralDevilPolicy()
    for path in (CYCLE3, CYCLE2):
        state, moves = _state_and_moves(path)
        scores = policy.score_moves(state, moves)
        assert scores.shape == (len(moves),)
    assert FEATURE_DIM > 0


def test_changing_weights_changes_scores():
    policy = NeuralDevilPolicy()
    state, moves = _state_and_moves(CYCLE3)
    before = policy.score_moves(state, moves).detach().clone()
    with torch.no_grad():
        for p in policy.parameters():
            p.add_(torch.randn_like(p) * 2.0)
    after = policy.score_moves(state, moves).detach().clone()
    assert not torch.allclose(before, after)


def test_symbolic_devil_behavior_intact():
    # The default symbolic Devil is unchanged: it still picks the first legal move.
    for path in (CYCLE3, CYCLE2):
        state, moves = _state_and_moves(path)
        chosen = SymbolicActiveDevil().choose_move(state.structure, state.task)
        assert chosen == moves[0]


def test_neural_active_devil_wrapper_returns_legal_move():
    devil = NeuralActiveDevil(NeuralDevilPolicy())
    state, moves = _state_and_moves(CYCLE3)
    chosen = devil.choose_move(state.structure, state.task)
    assert chosen in moves
