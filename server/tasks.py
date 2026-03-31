from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dataclasses import dataclass, field
from typing import Any, Dict, List

from models import SEED_CONFIG

@dataclass
class EpisodeRecord:
    task_id:        int
    initial_money:  float
    final_money:    float
    storage_value:  float          # market value of unsold crops at end
    total_reward:   float
    days_elapsed:   int
    max_days:       int
    withered_count: int            # total crops that withered across episode
    drought_days:   int            # days where drought_active was True
    healthy_days:   int            # plot-days where at least 2 plots were healthy
    sell_events:    List[Dict[str, Any]] = field(default_factory=list)
    # each sell event: {"day": int, "crop": str, "qty": float, "price": float, "base_price": float}

def grade_task1(record: EpisodeRecord) -> float:
    """
    Easy task — single crop, stable climate.
    Perfect score = double your starting money.
    """
    if record.initial_money <= 0:
        return 0.0

    net_worth = record.final_money + record.storage_value
    ratio     = net_worth / (record.initial_money * 2.0)
    score     = min(1.0, max(0.0, ratio))

    # small penalty for any withered crops (shouldn't happen on easy mode)
    wither_penalty = min(0.2, record.withered_count * 0.05)
    score          = max(0.0, score - wither_penalty)

    return round(score, 4)

def grade_task2(record: EpisodeRecord) -> float:
    """
    Medium task — multi-crop, market timing.
    Score = 0.6 × profit_score + 0.4 × timing_score
    """
    if record.initial_money <= 0:
        return 0.0

    # profit component
    net_worth     = record.final_money + record.storage_value
    profit_ratio  = net_worth / (record.initial_money * 2.5)
    profit_score  = min(1.0, max(0.0, profit_ratio))

    # timing component — what fraction of sell revenue came from above-base prices
    if not record.sell_events:
        timing_score = 0.0
    else:
        good_revenue  = 0.0
        total_revenue = 0.0
        for event in record.sell_events:
            revenue      = event["price"] * event["qty"]
            base_revenue = event["base_price"] * event["qty"]
            total_revenue += revenue
            if event["price"] > event["base_price"]:
                good_revenue += revenue - base_revenue   # only the premium

        if total_revenue > 0:
            timing_score = min(1.0, good_revenue / (total_revenue * 0.3))
        else:
            timing_score = 0.0

    # wither penalty is harsher on medium
    wither_penalty = min(0.3, record.withered_count * 0.1)

    score = (0.6 * profit_score) + (0.4 * timing_score)
    score = max(0.0, score - wither_penalty)
    return round(score, 4)

def grade_task3(record: EpisodeRecord) -> float:
    """
    Hard task — drought, spoilage, resource pressure.
    Score = 0.5 × profit_score + 0.3 × survival_score + 0.2 × resilience_score
    """
    if record.initial_money <= 0:
        return 0.0

    # profit component — target is 3× starting money (hard to achieve with drought)
    net_worth    = record.final_money + record.storage_value
    profit_ratio = net_worth / (record.initial_money * 3.0)
    profit_score = min(1.0, max(0.0, profit_ratio))

    # survival component — did the agent survive the full episode without bankruptcy?
    if record.final_money > 0 and record.days_elapsed >= record.max_days:
        survival_score = 1.0
    elif record.final_money > 0:
        # survived but episode ended early (shouldn't happen unless bug)
        survival_score = record.days_elapsed / record.max_days
    else:
        # went bankrupt
        survival_score = 0.0

    # resilience component — fraction of days where at least 2 plots stayed healthy
    if record.max_days > 0:
        resilience_score = min(1.0, record.healthy_days / (record.max_days * 2))
    else:
        resilience_score = 0.0

    # wither penalty is harshest on hard mode
    wither_penalty = min(0.4, record.withered_count * 0.15)

    score = (
        0.5 * profit_score
      + 0.3 * survival_score
      + 0.2 * resilience_score
    )
    score = max(0.0, score - wither_penalty)
    return round(score, 4)

def grade_episode(record: EpisodeRecord) -> float:
    """Route to the correct grader by task_id. Returns float in [0.0, 1.0]."""
    if record.task_id == 1:
        return grade_task1(record)
    elif record.task_id == 2:
        return grade_task2(record)
    elif record.task_id == 3:
        return grade_task3(record)
    else:
        raise ValueError(f"Unknown task_id: {record.task_id}")
