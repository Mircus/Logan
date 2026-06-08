import torch
import torch.nn as nn

from logical_gans.modelbuilder.learned.policy_net import RelationPolicyNet


def test_forward_shape_batched():
    n = 3
    net = RelationPolicyNet(n)
    x = torch.zeros(5, 4, n, n)
    out = net(x)
    assert out.shape == (5, n * n * 2)


def test_forward_shape_unbatched():
    n = 3
    net = RelationPolicyNet(n)
    out = net(torch.zeros(4, n, n))
    assert out.shape == (1, n * n * 2)


def test_loss_and_one_optimizer_step():
    n = 3
    net = RelationPolicyNet(n)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    x = torch.randn(8, 4, n, n)
    y = torch.randint(0, n * n * 2, (8,))
    loss_fn = nn.CrossEntropyLoss()
    loss0 = loss_fn(net(x), y)
    opt.zero_grad()
    loss0.backward()
    opt.step()
    assert torch.isfinite(loss0)
