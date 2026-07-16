import pytest
torch = pytest.importorskip("torch")

from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.semantic_actions import (
    SetConstant,
    SetFunction,
    SetRelation,
)
from logical_gans.modelbuilder.learned.semantic_features import FEATURE_DIM, feature_vector
from logical_gans.modelbuilder.learned.semantic_policy_net import SemanticPolicyNet

SIG = Signature.build(relations=[("E", 2)], functions=[("s", 1)], constants=["a"])


def test_forward_shape():
    net = SemanticPolicyNet(FEATURE_DIM)
    x = torch.zeros(5, FEATURE_DIM)
    out = net(x)
    assert out.shape == (5,)


def test_one_optimizer_step():
    net = SemanticPolicyNet(FEATURE_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    s = PartialStructure.empty(SIG, 2)
    feats = torch.tensor([
        feature_vector(SetRelation("E", (0, 0), Truth.FALSE), s),
        feature_vector(SetFunction("s", (0,), 1), s),
        feature_vector(SetConstant("a", 0), s),
    ], dtype=torch.float32)
    target = torch.tensor([0])
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = net(feats).unsqueeze(0)  # (1, 3)
    loss = loss_fn(logits, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_scores_for_each_kind():
    net = SemanticPolicyNet(FEATURE_DIM)
    s = PartialStructure.empty(SIG, 2)
    feats = torch.tensor([
        feature_vector(SetRelation("E", (0, 0), Truth.TRUE), s),
        feature_vector(SetFunction("s", (1,), 0), s),
        feature_vector(SetConstant("a", 1), s),
    ], dtype=torch.float32)
    out = net(feats)
    assert out.shape == (3,) and torch.isfinite(out).all()
