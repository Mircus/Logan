"""The active Devil/Builder game and its judge.

Devil chooses a bounded challenge (a blocking cell); Builder replies with a
semantic edit filling THAT cell; Judge checks progress. The search is a DFS with
backtracking ordered by the Builder's policy; the node budget is the search
budget. This is Devil-first (challenge -> reply), not Builder-first free search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..core.partial_structure import PartialStructure
from ..learned.semantic_actions import SemanticEdit, apply_semantic_edit, semantic_edit_to_json
from ..learned.semantic_search import classify, obligation_edits


@dataclass(frozen=True)
class GameState:
    structure: PartialStructure
    task: object
    remaining_rounds: int = 0
    trace: tuple = ()


@dataclass(frozen=True)
class DevilMove:
    kind: str
    clause_name: Optional[str]
    assignment: Optional[dict]
    target_cell: Optional[dict]
    reason: str
    obligation: object = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class BuilderReply:
    kind: str                 # SetRelation | SetFunction | SetConstant
    edit: SemanticEdit
    reason: str


@dataclass(frozen=True)
class GameEvent:
    actor: str                # "Devil" | "Builder" | "Judge"
    payload: dict


@dataclass
class Outcome:
    outcome: str              # "won" | "lost" | "draw"
    nodes: int
    decisions: list           # winning path: [(structure_before, DevilMove, edit)]
    trace: List[GameEvent]


def judge(structure, task) -> str:
    status, _, _ = classify(structure, task.theory, task.goal, task.mode,
                            k=task.depth, b=None)
    return status   # "won" | "theory_failed" | "dead_end" | "open"


def legal_replies(move: DevilMove) -> List[SemanticEdit]:
    return obligation_edits(move.obligation) if move.obligation is not None else []


class _BudgetExhausted(Exception):
    pass


def play_game(task, devil, builder, max_nodes: int) -> Outcome:
    counter = {"n": 0}

    def dfs(structure, acc):
        status = judge(structure, task)
        if status == "won":
            return list(acc)
        if status in ("theory_failed", "dead_end"):
            return None
        move = devil.choose_move(structure, task)
        if move is None:
            return None
        replies = legal_replies(move)
        if not replies:
            return None
        for edit in builder.order(structure, move, replies):
            if counter["n"] >= max_nodes:
                raise _BudgetExhausted()
            counter["n"] += 1
            child = apply_semantic_edit(structure, edit)
            res = dfs(child, acc + [(structure, move, edit)])
            if res is not None:
                return res
        return None

    start = PartialStructure.empty(task.signature, task.domain_size)
    try:
        won_path = dfs(start, [])
    except _BudgetExhausted:
        return Outcome("draw", counter["n"], [], [])
    if won_path is None:
        return Outcome("lost", counter["n"], [], [])
    return Outcome("won", counter["n"], won_path, _build_trace(won_path, task))


def _build_trace(decisions, task) -> List[GameEvent]:
    events: List[GameEvent] = []
    for (_struct, move, edit) in decisions:
        events.append(GameEvent("Devil", {
            "move": move.kind, "clause": move.clause_name,
            "assignment": move.assignment, "target_cell": move.target_cell,
            "reason": move.reason,
        }))
        ej = semantic_edit_to_json(edit)
        events.append(GameEvent("Builder", {"reply": ej.get("kind"), **ej}))
        events.append(GameEvent("Judge", {"status": "progress"}))
    events.append(GameEvent("Judge", {"status": "builder_survives"}))
    return events
