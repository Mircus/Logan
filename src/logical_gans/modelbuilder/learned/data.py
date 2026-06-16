"""Supervised training examples mined from Devil witnesses.

For random partial structures over a theory's signature, run the bounded
Devil. When it is blocked on an UNKNOWN relation cell, label the target by a
theory-general *closure oracle*: seed the backtracking search from the same
structure and read the blocked cell's value in the model it finds. This
yields forced cells -> TRUE and free cells -> FALSE, which (unlike a pure
SparseHornPolicy heuristic) teaches genuine completion of forced facts such
as transitive closure. The theory is loaded data, not hard-coded.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch

from ..core.backtracking import _decision_cell, backtracking_generate
from ..core.devil import run_devil_bounded
from ..core.partial_structure import PartialStructure
from ..core.theory import Theory
from ..core.types import Truth
from .actions import RelationEdit, decode_action_index, encode_action_index
from .encoding import encode_state


@dataclass
class TrainingExample:
    theory_name: str
    relation: str
    n: int
    input_tensor: torch.Tensor
    target_action: int
    metadata: dict = field(default_factory=dict)


def _random_structure(theory: Theory, relation: str, n: int, rng: random.Random) -> PartialStructure:
    s = PartialStructure.empty(theory.signature, n)
    choices = [Truth.FALSE, Truth.TRUE, Truth.UNKNOWN]
    for i in range(n):
        for j in range(n):
            s.set_relation(relation, (i, j), rng.choice(choices))
    return s


def make_relation_training_examples(
    theory: Theory,
    relation: str,
    n: int,
    num_samples: int,
    seed: int = 0,
    k: Optional[int] = None,
    budget: Optional[int] = None,
) -> List[TrainingExample]:
    rng = random.Random(seed)
    examples: List[TrainingExample] = []
    for _ in range(num_samples):
        structure = _random_structure(theory, relation, n, rng)
        result = run_devil_bounded(structure, theory.clauses, k=k, budget=budget)
        if result.status != "unknown":
            continue
        cell = _decision_cell(structure, result)
        if cell is None or cell[0] != "relation" or cell[1] != relation:
            continue  # only binary-relation obligations in this milestone
        args = cell[2]

        # Closure oracle: complete this structure with the backtracking search
        # and read the blocked cell's value in the model it finds.
        oracle = backtracking_generate(theory, n, seed_structure=structure)
        if oracle.status != "satisfied":
            continue
        value = oracle.structure.get_relation(relation, args)
        if value is Truth.UNKNOWN:
            continue

        edit = RelationEdit(relation, args, value)
        examples.append(TrainingExample(
            theory_name=theory.name,
            relation=relation,
            n=n,
            input_tensor=encode_state(structure, relation, result.witness),
            target_action=encode_action_index(edit, n),
            metadata={
                "args": list(args),
                "value": value.value,
                "conclusion_value": result.witness.conclusion_value,
                "clause": result.witness.clause_name,
            },
        ))
    return examples


def write_training_examples_jsonl(examples: List[TrainingExample], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "theory_name": ex.theory_name,
                "relation": ex.relation,
                "n": ex.n,
                "input_tensor": ex.input_tensor.tolist(),
                "target_action": ex.target_action,
                "metadata": ex.metadata,
            }) + "\n")


def read_training_examples_jsonl(path) -> List[TrainingExample]:
    out: List[TrainingExample] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(TrainingExample(
                theory_name=d["theory_name"],
                relation=d["relation"],
                n=d["n"],
                input_tensor=torch.tensor(d["input_tensor"], dtype=torch.float32),
                target_action=d["target_action"],
                metadata=d.get("metadata", {}),
            ))
    return out
