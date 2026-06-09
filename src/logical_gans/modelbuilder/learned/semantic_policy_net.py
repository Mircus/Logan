"""Neural policy over generic semantic-edit action features."""
from __future__ import annotations

import torch
import torch.nn as nn


class SemanticPolicyNet(nn.Module):
    """Input (num_candidate_actions, feature_dim) -> one score per action."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, action_features: torch.Tensor) -> torch.Tensor:
        return self.net(action_features).squeeze(-1)
