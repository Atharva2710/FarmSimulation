from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class EpisodeRecord:
    task_id:        int
    initial_money:  float
    final_money:    float
    storage_value:  float
    total_reward:   float
    days_elapsed:   int
    max_days:       int
    withered_count: int
    drought_days:   int
    healthy_days:   int
    sell_events:    List[Dict[str, Any]] = field(default_factory=list)


# ── Pure scorer functions ──────────────────────────────────────────────────────

def _score_profit(record: EpisodeRecord, target_mult: float) -> float:
    net_worth = record.final_money + record.storage_value
    ratio = net_worth / max(record.initial_money * target_mult, 1.0)
    return min(0.99, max(0.01, ratio))


def _score_stewardship(record: EpisodeRecord) -> float:
    return min(0.99, record.healthy_days / max(record.max_days, 1))


def _score_efficiency(record: EpisodeRecord) -> float:
    wither_rate = record.withered_count / max(record.days_elapsed, 1)
    return min(0.99, max(0.01, 1.0 - wither_rate * 2.0))


def _score_timing(record: EpisodeRecord) -> float:
    if not record.sell_events:
        return 0.01
    good_revenue = 0.0
    total_revenue = 0.0
    for e in record.sell_events:
        revenue = e["price"] * e["qty"]
        total_revenue += revenue
        if e["price"] > e["base_price"]:
            good_revenue += revenue - e["base_price"] * e["qty"]
    if total_revenue <= 0:
        return 0.01
    return min(0.99, good_revenue / (total_revenue * 0.3))


def _score_resilience(record: EpisodeRecord) -> float:
    if record.max_days <= 0:
        return 0.01
    return min(0.99, record.healthy_days / record.max_days)


def _score_survival(record: EpisodeRecord) -> float:
    if record.final_money > 0 and record.days_elapsed >= record.max_days:
        return 0.99
    if record.final_money > 0:
        return record.days_elapsed / record.max_days
    return 0.01


# ── RubricComposer ─────────────────────────────────────────────────────────────

@dataclass
class Dimension:
    name:    str
    weight:  float
    scorer:  Callable[[EpisodeRecord], float]

    def compute(self, record: EpisodeRecord) -> float:
        return round(self.scorer(record), 4)


@dataclass
class Gate:
    name:          str
    condition:     Callable[[EpisodeRecord], bool]
    on_fail_score: float

    def check(self, record: EpisodeRecord) -> bool:
        return self.condition(record)


class RubricComposer:
    def __init__(self, gates: List[Gate], dimensions: List[Dimension]) -> None:
        self.gates      = gates
        self.dimensions = dimensions

    def grade(self, record: EpisodeRecord) -> dict:
        for gate in self.gates:
            if not gate.check(record):
                return {
                    "score":      gate.on_fail_score,
                    "dimensions": {},
                    "gated":      gate.name,
                }
        dim_scores = {d.name: d.compute(record) for d in self.dimensions}
        weighted = sum(dim_scores[d.name] * d.weight for d in self.dimensions)
        return {
            "score":      round(min(0.99, max(0.01, weighted)), 4),
            "dimensions": dim_scores,
            "gated":      None,
        }


# ── Per-task rubrics ──────────────────────────────────────────────────────────

def _net_worth(r: EpisodeRecord) -> float:
    return r.final_money + r.storage_value

TASK1_RUBRIC = RubricComposer(
    gates=[
        Gate("solvency", lambda r: _net_worth(r) >= r.initial_money, 0.01),
    ],
    dimensions=[
        Dimension("profit",      0.7, lambda r: _score_profit(r, 2.0)),
        Dimension("stewardship", 0.2, _score_stewardship),
        Dimension("efficiency",  0.1, _score_efficiency),
    ],
)

TASK2_RUBRIC = RubricComposer(
    gates=[
        Gate("solvency", lambda r: _net_worth(r) >= r.initial_money, 0.01),
    ],
    dimensions=[
        Dimension("profit",      0.6, lambda r: _score_profit(r, 2.5)),
        Dimension("timing",      0.3, _score_timing),
        Dimension("efficiency",  0.1, _score_efficiency),
    ],
)

TASK3_RUBRIC = RubricComposer(
    gates=[
        Gate("solvency", lambda r: _net_worth(r) >= r.initial_money, 0.01),
    ],
    dimensions=[
        Dimension("profit",      0.5, lambda r: _score_profit(r, 3.0)),
        Dimension("survival",    0.3, _score_survival),
        Dimension("resilience",  0.2, _score_resilience),
    ],
)

_RUBRICS = {1: TASK1_RUBRIC, 2: TASK2_RUBRIC, 3: TASK3_RUBRIC}


# ── Public API ────────────────────────────────────────────────────────────────

def grade_episode_detailed(record: EpisodeRecord) -> dict:
    """Returns full rubric breakdown: {score, dimensions, gated}."""
    rubric = _RUBRICS.get(record.task_id)
    if rubric is None:
        raise ValueError(f"Unknown task_id: {record.task_id}")
    return rubric.grade(record)


def grade_episode(record: EpisodeRecord) -> float:
    """Returns scalar score in (0.01, 0.99) — backward-compatible entry point."""
    return grade_episode_detailed(record)["score"]


# ── Thin wrappers kept for test_phase4.py compatibility ──────────────────────

def grade_task1(record: EpisodeRecord) -> float:
    return grade_episode(record) if record.task_id == 1 else TASK1_RUBRIC.grade(record)["score"]


def grade_task2(record: EpisodeRecord) -> float:
    return grade_episode(record) if record.task_id == 2 else TASK2_RUBRIC.grade(record)["score"]


def grade_task3(record: EpisodeRecord) -> float:
    return grade_episode(record) if record.task_id == 3 else TASK3_RUBRIC.grade(record)["score"]
