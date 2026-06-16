"""Train the neural Builder from active-game winning traces (train tasks only).

For each train task we find a winning trajectory by running the active game with
a generous node budget; along that trajectory each Devil challenge yields a
supervised decision (candidate replies, index of the winning reply). The neural
Builder is trained to imitate the winning replies, so at test time it picks the
right value first more often -> less backtracking.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from ..learned.semantic_search import obligation_edits
from ..learned.semantic_training import train_token_policy, write_semantic_training_jsonl
from .game import play_game
from .players import ObligationFirstBuilder, SymbolicActiveDevil, fraisse_edit_tokens

_ORACLE_BUDGET = 50000   # generous; tasks are feasible so a win is found


def collect_examples(tasks) -> List[dict]:
    devil = SymbolicActiveDevil()
    examples: List[dict] = []
    for task in tasks:
        res = play_game(task, devil, ObligationFirstBuilder(), _ORACLE_BUDGET)
        if res.outcome != "won":
            continue
        for (structure_before, move, edit) in res.decisions:
            candidates = obligation_edits(move.obligation)
            if edit not in candidates:
                continue
            examples.append({
                "edits": [fraisse_edit_tokens(c, structure_before, move.obligation)
                          for c in candidates],
                "target": candidates.index(edit),
            })
    return examples


def train_builder(tasks, data_path, model_path, epochs: int = 120, seed: int = 0) -> dict:
    examples = collect_examples(tasks)
    if not examples:
        raise RuntimeError("no winning oracle traces collected on train tasks")
    Path(data_path).parent.mkdir(parents=True, exist_ok=True)
    write_semantic_training_jsonl(examples, data_path)
    meta = train_token_policy(str(data_path), str(model_path), epochs=epochs, seed=seed)
    meta["examples"] = len(examples)
    return meta
