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


class TokenSemanticPolicyNet(nn.Module):
    """Arity-parametric policy: each edit is a variable-length token sequence.

    Per-token MLP -> position-aware sum pooling -> per-edit MLP -> one scalar per
    edit. No fixed argument count, so any signature arity is supported.
    """

    def __init__(self, token_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.token_dim = token_dim
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.edit_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_edit(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (num_tokens, token_dim); position is carried inside each token,
        # so order-agnostic sum pooling still distinguishes per-position values.
        return self.token_mlp(tokens).sum(dim=0)

    def forward(self, edits_tokens):
        """edits_tokens: list of (num_tokens_e, token_dim) tensors -> (m,) scores."""
        embs = torch.stack([self.encode_edit(t) for t in edits_tokens], dim=0)
        return self.edit_mlp(embs).squeeze(-1)
