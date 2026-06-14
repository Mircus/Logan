"""Gate: the neural Builder is genuinely arity-parametric (no MAX_ARITY truncation).

Fresh signature with arity > 2:  Σ = {H/4 relation, f/3 function, c constant}.
"""
import json
from pathlib import Path

import torch

from logical_gans.modelbuilder.arena import solve_arena
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.semantic_actions import SetFunction, SetRelation
from logical_gans.modelbuilder.learned.semantic_features import feature_vector
from logical_gans.modelbuilder.learned.semantic_policy_net import TokenSemanticPolicyNet
from logical_gans.modelbuilder.learned.semantic_token_features import (
    TOKEN_DIM,
    edit_tokens,
)

SIG = Signature.build(relations=[("H", 4)], functions=[("f", 3)], constants=["c"])
ARG_TYPE_INDEX = 2  # token type one-hot slot for "arg"


def _struct():
    return PartialStructure.empty(SIG, 2)


def _n_arg_tokens(tokens):
    return sum(1 for t in tokens if t[ARG_TYPE_INDEX] == 1.0)


def test_one_token_per_argument_no_truncation():
    s = _struct()
    h = edit_tokens(SetRelation("H", (0, 0, 0, 0), Truth.TRUE), s)
    f = edit_tokens(SetFunction("f", (0, 0, 0), 1), s)
    assert _n_arg_tokens(h) == 4   # all 4 positions of H/4 represented
    assert _n_arg_tokens(f) == 3   # all 3 positions of f/3 represented
    assert all(len(tok) == TOKEN_DIM for tok in h + f)


def test_high_arity_positions_encode_differently():
    s = _struct()
    base = edit_tokens(SetRelation("H", (0, 0, 0, 0), Truth.TRUE), s)
    diff3 = edit_tokens(SetRelation("H", (0, 0, 1, 0), Truth.TRUE), s)   # 3rd arg
    diff4 = edit_tokens(SetRelation("H", (0, 0, 0, 1), Truth.TRUE), s)   # 4th arg
    assert base != diff3
    assert base != diff4
    assert diff3 != diff4                                                # position matters
    fbase = edit_tokens(SetFunction("f", (0, 0, 0), 1), s)
    fdiff3 = edit_tokens(SetFunction("f", (0, 0, 1), 1), s)              # 3rd arg of f/3
    assert fbase != fdiff3


def test_legacy_featurizer_would_collapse_these():
    # Documents the defect this gate fixes: MAX_ARITY=2 makes high-position edits
    # identical, whereas the token encoder keeps them distinct.
    s = _struct()
    a = SetRelation("H", (0, 0, 0, 0), Truth.TRUE)
    b = SetRelation("H", (0, 0, 0, 1), Truth.TRUE)   # differ only at 4th arg
    assert feature_vector(a, s) == feature_vector(b, s)        # old: collapsed
    assert edit_tokens(a, s) != edit_tokens(b, s)             # new: distinct


def test_policy_learns_high_position_distinction():
    """Label depends ONLY on the 4th argument; train, then check a HELD-OUT pair."""
    s = _struct()

    def pair(prefix):
        a = SetRelation("H", prefix + (0,), Truth.TRUE)
        b = SetRelation("H", prefix + (1,), Truth.TRUE)
        # candidate order [a, b]; correct = the one whose 4th arg is 1 -> target 1
        return [edit_tokens(a, s), edit_tokens(b, s)], 1

    prefixes = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)]
    holdout = [(1, 0, 1), (1, 1, 1)]
    train = [pair(p) for p in prefixes]

    torch.manual_seed(0)
    net = TokenSemanticPolicyNet(TOKEN_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(300):
        for cands, target in train:
            tensors = [torch.tensor(c, dtype=torch.float32) for c in cands]
            logits = net(tensors).unsqueeze(0)
            loss = loss_fn(logits, torch.tensor([target]))
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    for p in holdout:
        cands, _ = pair(p)
        with torch.no_grad():
            scores = net([torch.tensor(c, dtype=torch.float32) for c in cands])
        # the action with 4th arg == 1 (index 1) must outrank the other
        assert scores[1].item() > scores[0].item(), f"held-out {p}: {scores.tolist()}"


def test_arena_cycle3_regression_god_wins():
    root = Path(__file__).resolve().parents[1]
    problem = json.loads((root / "examples" / "problems" / "cycle3_arena.json").read_text())
    res = solve_arena(problem, epochs=150).to_json()
    assert res["outcome"] == "GOD_WINS"
    assert res["builder"]["kind"] == "neural_mcts"
    assert res["builder"]["uses_neural_policy"] is True
    assert res["builder"]["success_via"] == "guided_tree_policy"
    assert res["model_relation"] == "A ⊩_{3,800} T"
    assert res["counterclaim_relation"] == "A ⊭_{3,800} C"
    assert res["structure"] is not None and res["witness"]


def test_arena_tiny_budget_draw():
    root = Path(__file__).resolve().parents[1]
    problem = json.loads((root / "examples" / "problems" / "cycle3_arena_tiny_budget.json").read_text())
    res = solve_arena(problem, epochs=150).to_json()
    assert res["outcome"] == "DRAW"
    assert res["reason"] == "search_budget_exhausted"
    assert res["structure"] is None and res["witness"] is None
