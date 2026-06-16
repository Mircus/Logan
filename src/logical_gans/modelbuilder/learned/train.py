"""Train the relation policy on mined Devil-witness examples."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .data import read_training_examples_jsonl
from .policy_net import RelationPolicyNet


def train_relation_policy(
    data_path: str,
    out_path: str,
    n: int,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    examples = read_training_examples_jsonl(data_path)
    if not examples:
        raise ValueError(f"no training examples in {data_path}")

    X = torch.stack([ex.input_tensor for ex in examples])           # (N, 4, n, n)
    y = torch.tensor([ex.target_action for ex in examples], dtype=torch.long)

    model = RelationPolicyNet(n)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    final_loss = None
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    relation = examples[0].relation
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "n": n,
        "state_dict": model.state_dict(),
        "metadata": {
            "examples": len(examples),
            "epochs": epochs,
            "lr": lr,
            "relation": relation,
            "final_loss": final_loss,
        },
    }
    torch.save(checkpoint, out)
    return {
        "out": str(out),
        "n": n,
        "examples": len(examples),
        "epochs": epochs,
        "final_loss": final_loss,
        "relation": relation,
    }
