---
title: Farming RL Environment
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
  - rl
  - farming
  - resource-management
---

# Farming RL Environment

A farming simulation environment for training RL agents on multi-step planning,
resource management, and decision-making under uncertainty.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take one action |
| `/state` | GET | Episode metadata |
| `/health` | GET | Liveness check |
| `/schema` | GET | Action / observation schemas |

## Tasks

- **Task 1 (easy)** — 30 days, $200, single crop, stable climate
- **Task 2 (medium)** — 45 days, $150, all 3 crops, market timing
- **Task 3 (hard)** — 60 days, $100, drought events, fast spoilage

## Quickstart
```bash
POST https://your-space.hf.space/reset
POST https://your-space.hf.space/step
{"action_type": "buy_seeds", "seed_type": "wheat", "quantity": 3}
```

## Baseline scores

See `baseline_results.json` for LLM agent benchmark results.
