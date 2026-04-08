<div align="center">

# 🌾 FarmSimulation

**A physics-grounded Reinforcement Learning environment for precision agriculture AI**

*Where LLM agents must manage land, water, pests, and volatile markets — just like real autonomous farming systems.*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white&style=flat-square)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-OpenEnv%20Core-22c55e?style=flat-square)](https://huggingface.co/openenv)
[![License](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white&style=flat-square)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white&style=flat-square)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00?style=flat-square)](https://gradio.app)
[![Meta Hackathon](https://img.shields.io/badge/Meta-PyTorch%20Hackathon-blue?style=flat-square&logo=meta)](https://huggingface.co/meta-llama)

---

[**Live Demo →**](https://huggingface.co/spaces/your-username/FarmSimulation) · [**API Docs →**](#-api-reference) · [**Quick Start →**](#-quick-start) · [**Task Curriculum →**](#-task-curriculum)

</div>

---

## 🧭 What is FarmSimulation?

**FarmSimulation** is a high-fidelity Reinforcement Learning environment built for the Meta PyTorch Hackathon and the [OpenEnv](https://huggingface.co/openenv) framework. It benchmarks whether LLM-based agents can master the multi-step, resource-constrained, economically-grounded decisions of real agricultural management.

Real farmers must simultaneously:
- ⚖️ Balance scarce capital against volatile commodity markets
- 💧 Manage water strategically across droughts and tropical monsoons
- 🦟 Control exponential pest outbreaks before cascading crop failure
- ⏰ Time harvests and market sales to maximize revenue from 20-day price cycles

FarmSimulation encodes all of this into a rigorous, scientifically-grounded RL environment with **11 action types**, **3 rotating climates**, real-world crop physics (FAO-56), Almgren-Chriss market impact economics, and a 3-task curriculum requiring fundamentally different expert strategies per level.

### Why not CartPole?

| Dimension | CartPole / Atari | **FarmSimulation** |
|---|---|---|
| Real-world applicability | ❌ Academic proxy | ✅ $4.1T global agriculture sector |
| Scientific grounding | ❌ Physical toy system | ✅ FAO-56 Penman-Monteith hydrology |
| Economic realism | ❌ None | ✅ Almgren-Chriss market impact model |
| Skill gradient | ❌ Flat trial-and-error | ✅ Seed → growth → timing → liquidation |
| Agent interpretability | ❌ Pixel/vector signals | ✅ Physics-informed text narratives |
| Difficulty curriculum | ❌ Single mode | ✅ 3 tasks: Easy → Medium → Hard |

---

## ⚙️ Environment Specification

| Property | Value |
|---|---|
| **Name** | `farming-env` |
| **Version** | `1.0.0` |
| **Framework** | [OpenEnv Core](https://huggingface.co/openenv) |
| **Runtime** | FastAPI + Uvicorn + Gradio |
| **Protocol** | REST (`/reset`, `/step`, `/state`, `/health`) |
| **Python** | ≥ 3.11 |
| **Concurrent Sessions** | ✅ Supported |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent (LLM / Heuristic)                     │
│      [System Prompt] → [Text Summary] → [JSON Action]               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /step { action }
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FastAPI + OpenEnv Core                        │
│                   /reset  /step  /state  /health                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   FarmingEnvironment Core (1,400+ lines)            │
│                                                                      │
│  ┌─────────────────── 5-Pass Daily Cycle ───────────────────────┐  │
│  │  1. 🌧️  Hydrology     → aquifer + tank recharge from rain   │  │
│  │  2. 🏜️  Pedology      → FAO-56 soil moisture evaporation    │  │
│  │  3. 🦠  Ecology       → exponential pest & weed escalation  │  │
│  │  4. 🌱  Physiology    → crop health update + recovery       │  │
│  │  5. 💹  Economics     → sinusoidal market price drift       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌── 11 Action Handlers ─────────────────────────────────────────┐  │
│  │ buy_seeds · plant · irrigate · harvest · sell · pump_water   │  │
│  │ apply_fertilizer · spray_pesticide · pull_weeds · wait       │  │
│  │ buy_plot · end_day                                           │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌── Task Graders ────────┐  ┌── Reward Engine ──────────────────┐ │
│  │ grade_task1() Easy     │  │ Plant/Harvest/Sell bonuses        │ │
│  │ grade_task2() Medium   │  │ Daily passive rewards             │ │
│  │ grade_task3() Hard     │  │ Withering/waste penalties         │ │
│  └────────────────────────┘  └───────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🎮 Action Space

| Action | Required | Labor Cost | Description |
|---|---|---|---|
| `wait` | — | 1.0h | Rest; context-sensitive reward |
| `end_day` | — | 0.0h | Advance clock; reset 10h labor budget |
| `buy_seeds` | `seed_type`, `quantity` | 0.5h | Purchase crop seeds |
| `plant` | `plot_id`, `seed_type` | 2.0h | Plant seeds in empty plot |
| `irrigate` | `plot_id` | 0.5h | Add +0.2 soil moisture (uses 15L) |
| `pump_water` | — | 1.0h | Aquifer → tank transfer (50L, $5) |
| `apply_fertilizer` | `plot_id` | 1.0h | Boost NPK by +0.3 ($10) |
| `spray_pesticide` | `plot_id` | 1.0h | Clear pests + 3-day protection ($1.50) |
| `pull_weeds` | `plot_id` | 1.5h | Remove weeds (free) |
| `harvest` | `plot_id` | 4.0h | Collect mature crop into storage |
| `sell` | `seed_type`, `quantity` | 0.5h | Liquidate at current market price |
| `clear` | `plot_id` | 0.5h | Remove withered crop from plot |
| `buy_plot` | — | 2.0h | Expand farmland ($100 ± 10%) |

> **Labor System**: Each in-game day = 10 labor hours. Overflow automatically triggers `end_day`. Use `end_day` explicitly when you want crops to grow without performing actions.

---

## 🔬 The 5-Pass Physics Engine

### Pass 1 — Hydrology 💧

```
aquifer     += precip_mm × 2.0 × recharge_mult    # [1mm ≈ 2L; mult=0.5 during drought]
water_tank  += precip_mm × 2.0
```

### Pass 2 — Pedology (FAO-56 Evapotranspiration)

```
ET_c = (temp / 100) × (1.1 - humidity) × Kc_crop
soil_moisture -= ET_c + weed_penalty(0.05)
```

### Pass 3 — Ecology (Exponential Pest Growth)

```python
pest_severity = min(1.0, (pest_severity + 0.1) × 1.5)   # exponential!
health -= 0.05 × pest_severity
```

### Pass 4 — Plant Physiology (Health + Recovery)

```
# Damage sources (compound):
health -= 0.07  if moisture < 0.20        (drought stress)
health -= 0.07  if any NPK < 0.20         (nutrient deficiency)
health -= 0.12  if moisture > 0.85        (waterlogging)

# Recovery (crops can heal!):
if (0.25 ≤ moisture ≤ 0.85) and all(NPK ≥ 0.25) and (no pests):
    health += 0.03                         (+3%/day with optimal conditions)
```

### Pass 5 — Market Economics (Almgren-Chriss)

```python
# Sinusoidal cycle (20 days), phase-offset per crop
sell_price = base × (1 + 0.20 × sin(2π × (day + offset) / 20) + noise)

# Market impact on sell orders
slippage   = (qty / 10kg) × 0.5%         # temporary impact
price_drop = min(50%, qty / 10kg × 1%)   # permanent impact
```

---

## 🌱 Crop Reference

| Crop | Growth | Max Yield | Buy | Base Sell | Water | NPK Drain | Kc |
|---|---|---|---|---|---|---|---|
| `wheat` | 7 days | 10 kg | $5.00 | $8.00 | Low | [0.05, 0.02, 0.03] | 0.80 |
| `rice` | 12 days | 20 kg | $8.00 | $14.00 | **High** | [0.03, 0.04, 0.05] | 1.10 |
| `corn` | 18 days | 35 kg | $12.00 | $20.00 | Medium | [0.08, 0.04, 0.02] | 1.20 |

> ⚠️ **3-Day Harvest Window**: After reaching `mature`, plots wither if not harvested within 3 days.

---

## 🎯 Task Curriculum

### Task 1 — Single Crop Stable 🟢 Easy

- **Start**: $200 · 30 days · temperate climate · ±5% market noise
- **Goal**: Double your starting money
- **Grade**: `clamp(net_worth / 400, 0.01, 0.99) − wither_penalty`

### Task 2 — Multi-Crop Timing 🟡 Medium

- **Start**: $150 · 45 days · full climate rotation · ±10% noise · $0.50/day overhead
- **Goal**: Profit across all crops, sell at price peaks
- **Grade**: `0.6 × profit_score + 0.4 × timing_score`
- **Key**: 40% of score depends on *when* you sell vs. the 7-day moving average

### Task 3 — Drought Survival 🔴 Hard

- **Start**: $100 · 60 days · drought events (−15L tank every 5 days) · ±20% noise · $1.00/day overhead · 3%/day spoilage
- **Goal**: Survive + maintain profitability under extreme resource pressure
- **Grade**: `0.5 × profit + 0.3 × survival + 0.2 × resilience`

---

## 🏆 Reward Table

| Event | Reward | Notes |
|---|---|---|
| `plant` | +0.20 | Commitment bonus |
| `irrigate` (rescue) | +0.50 | Moisture was < 0.25 before action |
| `irrigate` (normal) | +0.10 | Routine care |
| `irrigate` (wasteful) | −0.50 | Over-watering |
| `harvest` | up to +1.00 | Scales with yield/max_yield |
| `sell` | +0.30 + premium | Bonus for above-baseline price |
| `wait` (crops growing) | +0.05/plot | Smart patience |
| `wait` (mature plots) | −0.30/plot | Risk of permanent wither |
| Health maintenance | +0.10–0.15/plot/day | Difficulty-scaled passive reward |
| Crop withers | −2.0 to −5.0 | Easy: −2.0 · Medium: −3.5 · Hard: −5.0 |
| Invalid action | −1.00 | Hard rejection |
| **Terminal bonus** | up to +10.0 | 40% profit + 30% stewardship + 30% efficiency |

---

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Start server + Gradio dashboard
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run LLM inference
export HF_TOKEN="hf_your_token"
python inference.py

# Docker
docker build -t farming-sim . && docker run -p 7860:7860 -e HF_TOKEN=hf_xxx farming-sim
```

---

## 🌐 API Reference

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `GET` | `/health` | — | Liveness check |
| `POST` | `/reset` | `{"task_id": 1\|2\|3}` | Start new episode |
| `POST` | `/step` | `{"action": FarmAction}` | Take one action |
| `GET` | `/state` | — | Full internal state |
| `GET` | `/` | — | Gradio interactive dashboard |

---

## 📁 Repository Structure

```
FarmSimulation/
├── server/
│   ├── app.py                  # FastAPI entry point + Gradio mount
│   ├── farming_environment.py  # 5-pass physics engine (1,400+ lines)
│   ├── gradio_app.py           # Glassmorphic dark-mode dashboard UI
│   ├── tasks.py                # EpisodeRecord + grade_task1/2/3 graders
│   ├── audit_utils.py          # State fidelity & tactical audit helpers
│   └── requirements.txt
├── models.py                   # Pydantic schemas + SEED/CLIMATE constants
├── inference.py                # LLM agent loop (OpenEnv compliance runner)
├── openenv.yaml                # OpenEnv manifest
├── pyproject.toml              # Dependencies (uv-compatible)
├── Dockerfile                  # Python 3.11-slim, exposes :7860
├── baseline_results.json       # Last inference run output
└── test_phase{2,3,4,5,7}.py   # Phase-gated TDD test suite
```

---

## 📊 Performance Baselines

| Task | Difficulty | Baseline (Qwen 2.5 72B) | Expert Target |
|---|---|---|---|
| 1 | 🟢 Easy | 0.42 | ≥ 0.80 |
| 2 | 🟡 Medium | 0.31 | ≥ 0.70 |
| 3 | 🔴 Hard | 0.19 | ≥ 0.55 |

---

## 📚 Citations

1. Allen et al. (1998). *FAO Irrigation and Drainage Paper 56*. FAO, Rome.
2. Almgren & Chriss (2000). *Optimal execution of portfolio transactions*. Journal of Risk.
3. Obizhaeva & Wang (2013). *Optimal trading strategy and supply/demand dynamics*. Journal of Financial Markets.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the Meta PyTorch Hackathon — OpenEnv-compatible environment for benchmarking LLM agents on agricultural planning.**

*If this environment helped your research, consider giving it a ⭐*

</div>
