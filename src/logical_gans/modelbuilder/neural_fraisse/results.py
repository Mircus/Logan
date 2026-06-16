"""Aggregation + table formatting for the benchmark."""
from __future__ import annotations

import statistics
from typing import Dict, List


def _median(xs: List[int]):
    return int(statistics.median(xs)) if xs else 0


def aggregate(rows: List[dict]) -> Dict[tuple, dict]:
    """rows: {method, kind, outcome, nodes} -> {(method,kind): summary}."""
    keys = sorted({(r["method"], r["kind"]) for r in rows})
    out = {}
    for method, kind in keys:
        sub = [r for r in rows if r["method"] == method and r["kind"] == kind]
        wins = [r for r in sub if r["outcome"] == "won"]
        draws = [r for r in sub if r["outcome"] == "draw"]
        out[(method, kind)] = {
            "n": len(sub),
            "wins": len(wins),
            "draws": len(draws),
            "success_rate": (len(wins) / len(sub)) if sub else 0.0,
            "median_nodes": _median([r["nodes"] for r in sub]),
            "median_nodes_win": _median([r["nodes"] for r in wins]),
        }
    return out


def overall(rows: List[dict], method: str) -> dict:
    sub = [r for r in rows if r["method"] == method]
    wins = [r for r in sub if r["outcome"] == "won"]
    return {
        "n": len(sub),
        "wins": len(wins),
        "draws": len([r for r in sub if r["outcome"] == "draw"]),
        "success_rate": (len(wins) / len(sub)) if sub else 0.0,
        "median_nodes": _median([r["nodes"] for r in sub]),
    }


def format_table(agg: Dict[tuple, dict]) -> str:
    header = f"{'method':<24}{'task_kind':<14}{'success':>9}{'med_nodes':>11}{'draws':>7}{'wins':>6}"
    lines = [header, "-" * len(header)]
    for (method, kind), s in agg.items():
        lines.append(f"{method:<24}{kind:<14}{s['success_rate']:>9.2f}"
                     f"{s['median_nodes']:>11}{s['draws']:>7}{s['wins']:>6}")
    return "\n".join(lines)
