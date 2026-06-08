"""Tiny torch policy over relation-edit actions."""
from __future__ import annotations

import torch
import torch.nn as nn

from .actions import num_actions


class RelationPolicyNet(nn.Module):
    """Input (batch, 4, n, n) -> logits (batch, n*n*2)."""

    def __init__(self, n: int, hidden_dim: int = 128):
        super().__init__()
        self.n = n
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * n * n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions(n)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.net(x)
