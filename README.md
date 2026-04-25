---
title: FarmSimulation
emoji: 🌾
colorFrom: green
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🌾 FarmSimulation

**A physics-grounded Reinforcement Learning environment for precision agriculture AI**

*Where LLM agents must manage land, water, pests, and volatile markets — just like real autonomous farming systems.*

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white&style=flat-square)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-OpenEnv%20Core-22c55e?style=flat-square)](https://huggingface.co/openenv)
[![License](https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white&style=flat-square)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white&style=flat-square)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/UI-Gradio-ff7c00?style=flat-square&logo=gradio)](https://gradio.app)
[![HF Space](https://img.shields.io/badge/🤗%20Live%20Demo-Space-orange?style=flat-square)](https://huggingface.co/spaces/your-username/FarmSimulation)

---

[**Live Demo →**](https://huggingface.co/spaces/your-username/FarmSimulation) · [**API Docs →**](#-api-reference) · [**Quick Start →**](#-quick-start) · [**Task Curriculum →**](#-task-curriculum)

</div>

---

## 🧭 Overview

**FarmSimulation** is a high-fidelity Reinforcement Learning environment built for the **Meta PyTorch Hackathon** and the [OpenEnv](https://huggingface.co/openenv) framework. It evaluates whether LLM-based agents can move beyond toy benchmarks and master the multi-step, resource-constrained, economically-grounded decisions of real agricultural management.

Real farmers don't just plant seeds and wait. They must:
- ⚖️ **Balance scarce capital** against volatile commodity markets
- 💧 **Manage water strategically** across droughts and tropical monsoons  
- 🦟 **Control exponential pest outbreaks** before they cause cascading crop failure
- ⏰ **Time harvests and market sales** to maximize revenue from 20-day price cycles

FarmSimulation encodes all of this into a rigorous, **scientifically-grounded** RL environment with 11 action types, 3 rotating climates, real-world crop physics, and a tiered 3-task curriculum that requires fundamentally different expert strategies at each level.

### Why not CartPole?

| Dimension | CartPole / Atari | **FarmSimulation** |
|---|---|---|
| **Real-world applicability** | ❌ Academic proxy | ✅ $4.1T global agriculture sector |
| **Scientific grounding** | ❌ Physical toy system | ✅ FAO-56 Penman-Monteith hydrology |
| **Economic realism** | ❌ None | ✅ Almgren-Chriss market impact model |
| **Skill gradient** | ❌ Flat trial-and-error | ✅ Seed → growth → timing → liquidation |
| **Agent interpretability** | ❌ Pixel/vector signals | ✅ Physics-informed text narratives |
| **Difficulty curriculum** | ❌ Single mode | ✅ 3 tasks: Easy → Medium → Hard |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Agent (LLM / Heuristic)                    │
│       [System Prompt] → [Text Summary] → [JSON Action]              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /step { action }
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI + OpenEnv Core                       │
│                    /reset  /step  /state  /health                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FarmingEnvironment (1,400+ lines)                │
│                                                                      │
│  ┌─────────────────── 5-Pass Daily Cycle ────────────────────────┐  │
│  │  1. 🌧️  Hydrology     → aquifer + tank recharge from rain    │  │
│  │  2. 🏜️  Pedology      → FAO-56 soil moisture evaporation     │  │
│  │  3. 🦠  Ecology       → exponential pest & weed escalation   │  │
│  │  4. 🌱  Physiology    → crop health update + recovery        │  │
│  │  5. 💹  Economics     → sinusoidal market price drift        │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌── Action Handlers ────────────────────────────────────────────┐  │
│  │  buy_seeds · plant · irrigate · harvest · sell                │  │
│  │  pump_water · apply_fertilizer · spray_pesticide              │  │
│  │  pull_weeds · buy_plot · wait · end_day                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌── Task Graders ───────┐  ┌── Reward Engine ──────────────────┐  │
│  │  grade_task1() Easy   │  │  Plant/Harvest/Sell bonuses       │  │
│  │  grade_task2() Medium │  │  Daily passive rewards            │  │
│  │  grade_task3() Hard   │  │  Withering/waste penalties        │  │
│  └───────────────────────┘  └───────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Gradio Dashboard   │
                    │  Glassmorphic Dark   │
                    │  Soil & Leaf Theme   │
                    └──────────────────────┘
```

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

### Action Space (11 Action Types)

| Action | Required Fields | Labor Cost | Description |
|---|---|---|---|
| `wait` | — | 1.0h | Rest; reward scales with farm state |
| `end_day` | — | 0.0h | Advance clock; resets 10h labor budget |
| `buy_seeds` | `seed_type`, `quantity` | 0.5h | Purchase seeds from market |
| `plant` | `plot_id`, `seed_type` | 2.0h | Plant seeds in an empty plot |
| `irrigate` | `plot_id` | 0.5h | Add +0.2 soil moisture (costs 15L water) |
| `pump_water` | — | 1.0h | Transfer aquifer water → tank (50L max, $5) |
| `apply_fertilizer` | `plot_id` | 1.0h | Boost NPK by +0.3 ($10/use) |
| `spray_pesticide` | `plot_id` | 1.0h | Eliminate pests + 3-day protection ($1.50) |
| `pull_weeds` | `plot_id` | 1.5h | Remove weeds (free) |
| `harvest` | `plot_id` | 4.0h | Collect mature crop into storage |
| `sell` | `seed_type`, `quantity` | 0.5h | Liquidate storage at market price |
| `clear` | `plot_id` | 0.5h | Remove withered crop, free the plot |
| `buy_plot` | — | 2.0h | Expand to a new land plot ($100 ± 10%) |

> **Labor Budget**: Each real-world day provides **10 labor hours**. When hours are exhausted, the next action automatically triggers `end_day`. Use `end_day` explicitly to advance time when crops are growing.

### Observation Space

```json
{
  "day": 14,
  "money": 243.50,
  "water_tank": 0.72,
  "aquifer": 412.0,
  "labor_remaining": 6.5,
  "seed_inventory": { "wheat": 3, "rice": 0, "corn": 0 },
  "storage": { "wheat": 8.5, "rice": 0.0, "corn": 0.0 },
  "climate": {
    "climate_type": "arid",
    "temperature": 37.2,
    "humidity": 0.18,
    "precipitation": 0.0
  },
  "market_prices": {
    "wheat": { "sell_price": 9.14, "trend": 0.21, "avg_7d": 8.43 },
    "rice":  { "sell_price": 12.80, "trend": -0.05, "avg_7d": 13.60 },
    "corn":  { "sell_price": 22.10, "trend": 0.48, "avg_7d": 19.20 }
  },
  "plots": [
    {
      "plot_id": 0, "stage": "growing", "crop_type": "wheat",
      "days_planted": 5, "health": 0.91, "soil_moisture": 0.62,
      "nitrogen": 0.75, "phosphorus": 0.82, "potassium": 0.79,
      "has_pests": false, "has_weeds": false, "pest_severity": 0.0
    }
  ],
  "weather_forecast": [ ... ],
  "text_summary": "Day 14 | $243.50 | Climate: Arid (37°C) ...",
  "valid_actions": ["irrigate", "wait", "sell", "end_day"]
}
```

---

## 🔬 Physics Engine: The 5-Pass Daily Cycle

Every call to `end_day` runs the world forward by exactly one simulation day through five sequential passes. **No random walks** — all dynamics are grounded in agricultural science.

### Pass 1 — Hydrology 💧

Precipitation recharges both the underground aquifer and the surface water tank:

```
aquifer   += rain_mm × 2.0 × recharge_mult   [1mm ≈ 2 litres; mult=0.5 during drought]
water_tank += rain_mm × 2.0                   [direct rain catch]
```

Rain probability and intensity are climate-specific:

| Climate | Rain Chance | Precip Range | Temp Range |
|---|---|---|---|
| `temperate` | 40% | 2.5–10 mm | 17–27°C |
| `arid` | 10% | 0.5–2 mm | 33–40°C |
| `tropical` | 70% | 6–24 mm | 23–31°C |

### Pass 2 — Pedology (FAO-56) 🏜️

Soil moisture evaporates daily. The rate is governed by crop coefficients (Kc) inspired by the **FAO-56 Penman-Monteith** standard:

```
ET_c = (temperature / 100) × (1.1 - humidity) × K_crop
soil_moisture -= ET_c + weed_penalty(0.05)
```

Crop coefficients: Wheat Kc=0.80 · Rice Kc=1.10 · Corn Kc=1.20

### Pass 3 — Ecology 🦠

Pests follow **exponential outbreak growth** if untreated:

```python
pest_severity = min(1.0, (pest_severity + 0.1) × 1.5)   # exponential!
health -= 0.05 × pest_severity
```

Spawn probability increases with humidity (>80%) and heat (>30°C). Pesticide provides **3 days of protection** against re-infestation.

### Pass 4 — Plant Physiology 🌱

Crops take compound damage from multiple simultaneous stressors:

```
health -= 0.07  if soil_moisture < 0.20   (drought stress)
health -= 0.07  if any NPK < 0.20         (nutrient deficiency)
health -= 0.12  if soil_moisture > 0.85   (waterlogging)
```

And — critically — **crops *recover* health** when conditions are optimal:

```python
if (0.25 ≤ moisture ≤ 0.85) and all(NPK ≥ 0.25) and (no pests):
    health = min(1.0, health + 0.03)   # +3% per day recovery
```

### Pass 5 — Market Economics 💹

Each crop rides an independent **20-day sinusoidal price cycle**, phase-shifted so peaks never align:

```python
# Offsets: wheat=0d, rice=7d, corn=13d
sell_price = base_sell × (1.0 + 0.2 × sin(2π × (day + offset) / 20) + noise)
```

Large sell orders permanently impact price (Almgren-Chriss model):

```
slippage    = (qty / 10kg) × 0.5%          # temporary execution cost
price_drop  = min(50%, qty / 10kg × 1%)   # permanent market impact
```

---

## 🌱 Crop Reference

| Crop | Growth | Max Yield | Buy | Base Sell | Water Need | NPK Drain [N, P, K] | Kc |
|---|---|---|---|---|---|---|---|
| `wheat` | 7 days | 10 kg | $5.00 | $8.00 | Low (0.3) | [0.05, 0.02, 0.03] | 0.80 |
| `rice` | 12 days | 20 kg | $8.00 | $14.00 | **High** (0.7) | [0.03, 0.04, 0.05] | 1.10 |
| `corn` | 18 days | 35 kg | $12.00 | $20.00 | Medium (0.5) | [0.08, 0.04, 0.02] | 1.20 |

> ⚠️ **Harvest Window**: Once a plot reaches `mature`, you have exactly **3 days** to harvest before permanent wither. `days_mature` in the obs tracks this countdown.

### Crop Pipeline Timeline

```
Plant → [Seedling: 0–33%] → [Growing: 33–100%] → [Mature] → HARVEST (3-day window) → Storage → Sell
                                                              ↳ Wither if missed!
```

---

## 🎯 Task Curriculum

Three tasks with escalating complexity, designed so that the optimal strategy for Task 1 fails catastrophically on Task 3.

### Task 1 — Single Crop Stable 🟢 Easy

| Parameter | Value |
|---|---|
| Starting money | $200 |
| Max days | 30 |
| Climate | Temperate-dominant |
| Market noise | ±5% |
| Daily overhead | $0 |
| **Goal** | Double your starting money |

**Grading formula:**
```
net_worth = final_money + storage_value
score = clamp(net_worth / ($200 × 2.0), 0.01, 0.99)
      − min(0.20, withered_crops × 0.05)
```

**Winning strategy**: Plant wheat (fastest ROI), maintain moisture in 0.35–0.75 range, sell when trend > 0.

---

### Task 2 — Multi-Crop Market Timing 🟡 Medium

| Parameter | Value |
|---|---|
| Starting money | $150 |
| Max days | 45 |
| Climate | Full rotation (temperate → arid → tropical) |
| Market noise | ±10% |
| Daily overhead | $0.50/day |
| **Goal** | Profit across all 3 crops, sell at price peaks |

**Grading formula:**
```
profit_score = clamp(net_worth / ($150 × 2.5), 0.01, 0.99)
timing_score = clamp(premium_revenue / (total_revenue × 0.3), 0.01, 0.99)
score = 0.6 × profit_score + 0.4 × timing_score − min(0.30, withered × 0.10)
```

**40% of your score is timing.** Agents must compare `sell_price vs avg_7d` and hold storage until peak.

---

### Task 3 — Drought Survival 🔴 Hard

| Parameter | Value |
|---|---|
| Starting money | $100 |
| Max days | 60 |
| Drought events | Every 5th day: −15L forced tank drain |
| Market noise | ±20% |
| Daily overhead | $1.00/day |
| Tropical spoilage | 3%/day crop degradation |
| **Goal** | Survive and maintain profitability under extreme resource pressure |

**Grading formula:**
```
profit_score     = clamp(net_worth / ($100 × 3.0), 0.01, 0.99)
survival_score   = 0.99 if survived full 60 days, else days/60
resilience_score = clamp(healthy_days / max_days, 0.01, 0.99)

score = 0.5 × profit_score + 0.3 × survival_score + 0.2 × resilience_score
      − min(0.40, withered_crops × 0.15)
```

**Proactive water management is mandatory.** Agents must `pump_water` regularly and pre-irrigate before drought days.

---

## 🏆 Reward Shaping

A rich, dense reward signal guides agents at every step:

| Event | Reward | Notes |
|---|---|---|
| `plant` | **+0.20** | Commitment bonus — agent started a plan |
| `irrigate` (rescue) | **+0.50** | Moisture was critically low (<0.25) before action |
| `irrigate` (normal) | **+0.10** | Routine maintenance |
| `irrigate` (wasteful) | **−0.50** | Over-watering (>0.80 before action) |
| `harvest` | **up to +1.00** | Scales with `stored_kg / max_yield` |
| `sell` | **+0.30 + premium** | Bonus for above-baseline price premium |
| `wait` (crops growing) | **+0.05/plot** | Smart patience while rows mature |
| `wait` (mature plots) | **−0.30/plot** | Every idle day risks permanent wither loss |
| `wait` (idle empty) | **−0.10/plot** | Wasted opportunity cost |
| Health maintenance | **+0.10–0.15/plot/day** | Passive bonus scaled by task difficulty |
| Crop withers | **−2.0 to −5.0** | Hard penalty. Scaled: Easy −2.0, Med −3.5, Hard −5.0 |
| Spray (no pests) | **−0.20** | Wasteful chemical use |
| Fertilize (already high) | **−0.20** | Resource waste |
| Invalid action | **−1.00** | Hard rejection for out-of-context actions |
| **Terminal bonus** | **up to +10.0** | 3-pillar: 40% profit + 30% stewardship + 30% efficiency |

> **Terminal Bonus Details**: `0.4 × profit_score + 0.3 × stewardship_score + 0.3 × efficiency_score`
> where `efficiency_score = 1 - (wasteful_actions / total_actions)`

---

## 🤖 LLM Agent Architecture

`inference.py` implements a **stateless LLM-as-Agent** loop compatible with any OpenAI-format API endpoint:

```
System Prompt ──► "Chief Economist & Growth Strategist" role
                  + RISK AUDIT / MARKET AUDIT / ACTION structure
       │
       ▼
Observation   ──► text_summary + valid_actions + 7d price history
       │
       ▼
LLM Response  ──► Raw JSON (parsed with fallback regex extractor)
       │
       ▼
Parser        ──► parse_action() → validate_action() → FALLBACK → {"action_type": "wait"}
       │
       ▼
Env Step      ──► env.step(action) → new observation + step reward
       │
       ▼
History Buffer ──► Last 4 steps kept in context window
```

### Inference Configuration

| Parameter | Default | Env Var |
|---|---|---|
| Model | `Qwen/Qwen2.5-72B-Instruct` | `MODEL_NAME` |
| API Endpoint | `https://router.huggingface.co/v1` | `API_BASE_URL` |
| HF Token | — | `HF_TOKEN` |
| Temperature | `0.2` | — |
| Max tokens | `300` | — |
| Max steps/episode | `30` | `MAX_STEPS` |
| Episodes per task | `1` | `EPISODES` |
| Task filter | all (1, 2, 3) | `FARMING_TASK_ID` |

### Reward Clamping (Phase 2 Compliance)

All per-step rewards are strictly clamped to `(0.01, 0.99)` — never `0.0` or `1.0` — to satisfy the OpenEnv Phase 2 validator:

```python
if reward <= 0.0:
    reward = 0.01
elif reward >= 1.0:
    reward = 0.99
```

The same clamping is applied at the episode and grader levels in `tasks.py`.

---

## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.11
- A Hugging Face token (`HF_TOKEN`) with inference access

### Option A — Local Development

```bash
# 1. Clone and install
git clone https://github.com/your-username/FarmSimulation.git
cd FarmSimulation
pip install -e .

# 2. Start the environment server + Gradio dashboard
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 3. Open the interactive dashboard
open http://localhost:7860
```

### Option B — Run LLM Inference

```bash
# Set credentials
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export FARMING_ENV_URL="http://localhost:7860"

# Run all 3 tasks (produces baseline_results.json)
python inference.py

# Run a single task
export FARMING_TASK_ID=2   # 1=easy  2=medium  3=hard
python inference.py
```

### Option C — Docker

```bash
# Build
docker build -t farming-sim .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token_here \
  farming-sim

# Access dashboard at http://localhost:7860
```

### Option D — Instant Validate

```bash
# Run the full 4-layer submission validation
./validate-submission.sh

# Or run individual test phases
pytest test_phase2.py -v   # Score range validation
pytest test_phase3.py -v   # Physics determinism
pytest test_phase4.py -v   # Labor hour system
pytest test_phase5.py -v   # Session persistence
pytest test_phase7.py -v   # Full integration
```

---

## 🌐 API Reference

All endpoints follow the [OpenEnv Core](https://huggingface.co/openenv) protocol.

### `GET /health`

Liveness probe. Returns `200 OK` when the server is ready.

```bash
curl http://localhost:7860/health
# {"status": "ok"}
```

### `POST /reset`

Resets the episode and returns the initial observation.

```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": 2}'
```

**Response:** `FarmObservation` (see [Observation Space](#observation-space))

### `POST /step`

Takes one action and returns the new observation with reward.

```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "action_type": "plant",
      "plot_id": 0,
      "seed_type": "wheat"
    }
  }'
```

**Response:**
```json
{
  "observation": { "day": 1, "money": 195.0, ... },
  "reward": 0.2,
  "done": false
}
```

### `GET /state`

Returns the full internal `FarmState` (includes episode metadata).

### `GET /`

Serves the interactive Gradio dashboard UI.

---

## 📁 Repository Structure

```
FarmSimulation/
│
├── server/                          # Core application package
│   ├── app.py                       # FastAPI + Gradio mount entry point
│   ├── farming_environment.py       # 🧠 Full physics engine (1,400+ lines)
│   ├── gradio_app.py                # Interactive dashboard (glassmorphic dark theme)
│   ├── tasks.py                     # EpisodeRecord + grade_task1/2/3 graders
│   ├── audit_utils.py               # State fidelity & tactical audit helpers
│   ├── scenario_engine.py           # Pre-built scenario runner
│   ├── scenario_definitions.py      # Named scenario configs
│   ├── baseline_inference.py        # Simple heuristic baseline agent
│   └── requirements.txt             # Server dependencies
│
├── models.py                        # 📦 Pydantic schemas + all constants
│                                    #    (SEED_CONFIG, CLIMATE_CONFIG, FarmAction,
│                                    #     FarmObservation, FarmState, PlotState)
│
├── inference.py                     # 🤖 LLM agent loop (OpenEnv compliance runner)
├── dual_llm_inference.py            # Experimental: Two-LLM debate architecture
├── openenv.yaml                     # OpenEnv manifest (name, version, tasks)
├── pyproject.toml                   # uv-compatible project dependencies
├── Dockerfile                       # Python 3.11-slim, exposes :7860
│
├── test_phase2.py                   # Score range (0.01–0.99) validation
├── test_phase3.py                   # Physics determinism tests
├── test_phase4.py                   # Labor hour system tests
├── test_phase5.py                   # WebSocket/session persistence tests
├── test_phase7.py                   # Full end-to-end integration tests
├── validate-submission.sh           # 4-layer submission validation script
│
├── robustness_validation.py         # Determinism + skill gradient prover
├── verify_all.py                    # Complete verification suite runner
├── baseline_results.json            # Last inference run output
│
├── FIXES_APPLIED.md                 # Changelog: bug fixes & balance changes
├── LLM_JUDGE_STRATEGY.md            # Strategy guide for LLM agents
├── MATHEMATICAL_FOUNDATION.md       # Physics/economics derivations
└── docs.html                        # Rich HTML documentation page
```

---

## 📐 Mathematical Foundations

### Soil Hydrology — FAO-56 Penman-Monteith

Evapotranspiration drives the daily moisture loss from each plot:

$$ET_c = \left(\frac{T}{100} \cdot (1.1 - H)\right) \cdot K_{crop}$$

where $T$ = temperature (°C), $H$ = humidity [0,1], $K_{crop}$ = crop coefficient.

Additionally, weed infestations apply a fixed moisture penalty of **−0.05/day** to simulate competition for water resources.

### Market Impact — Almgren-Chriss (2000)

Large sell orders exhibit two-tier price impact:

**Temporary impact** (slippage — affects only current trade):
$$P_{exec} = P_{mid} \cdot \left(1 - \eta \cdot Q\right), \quad \eta = 0.005 / 10\text{kg}$$

**Permanent impact** (shifts baseline for future trades):
$$P_{new} = P_{old} \cdot \left(1 - \gamma \cdot Q\right), \quad \gamma = 0.01 / 10\text{kg}, \quad \text{floor at } 50\%$$

### Sinusoidal Market Cycles

Each crop has a 20-day price wave, phase-offset to prevent synchronization:

$$\text{sell\_price}(d) = P_{base} \cdot \left(1 + 0.20 \cdot \sin\!\left(\frac{2\pi (d + \phi)}{20}\right) + \epsilon\right)$$

| Crop | Phase offset $\phi$ | Peak day |
|---|---|---|
| Wheat | 0 | Day 5, 25, 45… |
| Rice | 7 | Day 12, 32, 52… |
| Corn | 13 | Day 18, 38, 58… |

### Terminal Bonus — Three-Pillar Score

$$B_{terminal} = 0.40 \cdot S_{profit} + 0.30 \cdot S_{steward} + 0.30 \cdot S_{efficiency}$$

where:
- $S_{profit} = \min(10, \max(0, (\text{net\_worth}/\text{initial} - 1) \times 5))$  
- $S_{steward} = (\text{healthy\_days} / \text{episode\_days}) \times 10$  
- $S_{efficiency} = (1 - \text{wasteful\_actions}/\text{total\_actions}) \times 10$

---

## 🧪 Robustness & Compliance

This environment is validated against the OpenEnv Phase 2 certification requirements:

| Requirement | Status | Details |
|---|---|---|
| Score range `(0.01, 0.99)` | ✅ Passed | Triple-clamped: grader → termination → inference |
| Determinism (`seed=X`) | ✅ Passed | Same seed = identical trajectory |
| Skill gradient | ✅ Passed | Naive TWAP ≈ 0.25 < Expert timing ≈ 0.80 |
| Concurrent sessions | ✅ Passed | `SUPPORTS_CONCURRENT_SESSIONS = True` |
| REST compliance | ✅ Passed | `/reset`, `/step`, `/state`, `/health` |
| Reward logging format | ✅ Passed | `[STEP]`, `[END]`, `[START]` regex-compatible |

Run the complete validation suite:

```bash
./validate-submission.sh
```

Expected output:
```
✅ Phase 2: Score range (0.01, 0.99) — PASS
✅ Phase 3: Determinism seed=42 — PASS
✅ Phase 4: Labor hour system — PASS
✅ Phase 5: Session persistence — PASS
✅ Phase 7: Full integration — PASS
🏆 All 5 phases passed. Submission ready.
```

---

## 📊 Performance Baselines

Baseline scores collected using `Qwen/Qwen2.5-72B-Instruct` at temperature 0.2:

| Task | Difficulty | Baseline Score | Expert Target | Notes |
|---|---|---|---|---|
| 1 — Single Crop Stable | 🟢 Easy | **0.42** | ≥ 0.80 | Limited market timing |
| 2 — Multi-Crop Timing | 🟡 Medium | **0.31** | ≥ 0.70 | Misses peak sell windows |
| 3 — Drought Survival | 🔴 Hard | **0.19** | ≥ 0.55 | Reactive vs. proactive water |

> A score ≥ **0.80** across all tasks is considered professional tier.

---

## 🤝 Contributing

We welcome contributions that improve the environment's realism, extend the task curriculum, or enhance the agent strategies.

```bash
# 1. Fork and clone
git clone https://github.com/your-username/FarmSimulation.git

# 2. Create a feature branch
git checkout -b feat/my-improvement

# 3. Make changes and run the test suite
pytest test_phase*.py -v

# 4. Run the robustness validator
python robustness_validation.py

# 5. Submit a pull request
```

### Code Style

- All environment physics lives in `server/farming_environment.py`
- All Pydantic schemas and constants live in `models.py`
- All grading logic lives in `server/tasks.py`
- Keep the API backward-compatible: `/reset`, `/step`, `/state`, `/health`

---

## 📚 Citations

If you use this environment in research, please cite:

```bibtex
@misc{farmsimulation2026,
  title  = {FarmSimulation: A Physics-Grounded RL Environment for Precision Agriculture},
  year   = {2026},
  note   = {Meta PyTorch Hackathon — OpenEnv compatible},
  url    = {https://huggingface.co/spaces/your-username/FarmSimulation}
}
```

**Scientific foundations:**

1. Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration — Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56. Rome: FAO.

2. Almgren, R., & Chriss, N. (2000). *Optimal execution of portfolio transactions*. Journal of Risk, 3(2), 5–40.

3. Obizhaeva, A. A., & Wang, J. (2013). *Optimal trading strategy and supply/demand dynamics*. Journal of Financial Markets, 16(1), 1–32.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the [Meta PyTorch Hackathon](https://huggingface.co/meta-llama) — an OpenEnv-compatible environment for evaluating LLM agents on real-world agricultural planning.**

*If this environment helped your research or project, consider giving it a ⭐*

</div>