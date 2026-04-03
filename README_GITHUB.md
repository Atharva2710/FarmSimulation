# 🌾 FarmSimulation

> A physics-grounded Reinforcement Learning environment where AI agents must manage 4 land plots, balance scarce water and capital, fight pests and drought, and time crop sales to volatile markets — just like real precision-agriculture AI does.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-OpenEnv%20Core-green)](https://huggingface.co/openenv)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](Dockerfile)
[![HF Space](https://img.shields.io/badge/🤗%20Space-Live-orange)](https://huggingface.co/spaces/your-username/FarmSimulation)

---

## 💡 What is FarmSimulation?

Real farmers don't just plant seeds and wait — they balance scarce capital against volatile markets, fight pests and drought, time harvests for peak prices, and never let a plot sit idle. Miss one irrigation in an arid climate and a week of growth dies overnight. Sell too early and you leave money on the table; sell too late and spoilage eats your margin.

This is the **Agricultural Resource Management** problem. **FarmSimulation** is an OpenEnv-compatible RL environment that makes agents solve it — simulating the full crop lifecycle, market economics, and environmental hazards that any autonomous farming agent must navigate to turn a profit.

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

### Action Space

| Field | Type | Range | Description |
|---|---|---|---|
| `action_type` | `str` | 10 valid types | The farming operation to perform this step |
| `plot_id` | `int` | `[0, 3]` | Target land plot (required for plot operations) |
| `seed_type` | `str` | `wheat\|rice\|corn` | Crop variety (required for buy/plant/sell) |
| `quantity` | `int` | `> 0` | Seeds to buy or kilograms to sell |

**Valid action types:** `wait`, `buy_seeds`, `plant`, `irrigate`, `pump_water`, `apply_fertilizer`, `spray_pesticide`, `pull_weeds`, `harvest`, `sell`, `clear`

### Observation Space

| Field | Type | Description |
|---|---|---|
| `day` | `int` | Current simulation day |
| `money` | `float` | Agent cash balance |
| `water_tank` | `float` | Tank fill fraction `[0, 1]` |
| `aquifer` | `float` | Underground reserve in litres |
| `seed_inventory` | `Dict[str, int]` | Seeds on hand |
| `storage` | `Dict[str, float]` | Harvested crop kg |
| `plots` | `List[PlotState]` | 4 independent land plot states |
| `climate` | `ClimateState` | Current temperature, humidity, precipitation |
| `market_prices` | `Dict[str, MarketPrice]` | Live prices with trend signal |
| `text_summary` | `str` | Human/LLM-readable narrative snapshot |
| `valid_actions` | `List[str]` | Context-sensitive legal action hints |

---

## 🧮 The Physics Engine

Every day progression runs five simultaneous simulation passes. No random walks — the model is grounded in agricultural science:

```
1. Precipitation      → aquifer += climate.precipitation × 2   [mm → litres]
2. Moisture Decay     → soil_moisture -= climate.moisture_decay + weed_penalty(0.05)
3. Pest Escalation    → pest_severity = min(1.0, (sev + 0.1) × 1.5)   [exponential growth]
4. Health Damage      → health -= 0.1   [if moisture < 0.2 OR any NPK < 0.2 OR moisture > 0.9]
5. Market Tick        → sell_price = base × (1.0 + 0.2·sin(2π·day/20 + offset) + noise)
```

### Climate Physics

| Climate | Temp | Humidity | Precipitation | Moisture Decay | Spoilage |
|---|---|---|---|---|---|
| `temperate` | 22°C | 60% | 5 mm/day | 0.05/day | 1%/day |
| `arid` | 35°C | 20% | 1 mm/day | **0.12/day** | 1%/day |
| `tropical` | 28°C | 90% | 12 mm/day | 0.03/day | **3%/day** |

Climates rotate every **10 days**: `temperate → arid → tropical`. Extreme temperatures (>32°C or <10°C) freeze crop growth entirely.

### Market Dynamics

Each crop rides its own **20-day sine wave**, desynchronized by a fixed offset so peaks never align:

```python
offset = {"wheat": 0, "rice": 7, "corn": 13}
sell_price = base_sell × (1.0 + 0.20 × sin(2π × (day + offset) / 20) + noise)
```

**Price Elasticity (Market Impact):** Large sell orders crash their own price:
```
price_drop = min(50%, qty_sold / 10kg × 1%)
sell_price *= (1.0 - price_drop)
```

---

## 🌱 Crop Reference

| Crop | Grow Days | Max Yield | Buy Price | Sell Price | Water Need | NPK Drain [N, P, K] |
|---|---|---|---|---|---|---|
| `wheat` | 7 days | 10 kg | $5.00 | $8.00 | Low | [0.05, 0.02, 0.03] |
| `rice` | 12 days | 20 kg | $8.00 | $14.00 | **High** | [0.03, 0.04, 0.05] |
| `corn` | 18 days | 35 kg | $12.00 | $20.00 | Medium | [0.08, 0.04, 0.02] |

> **Critical:** Once a plot reaches `mature`, you have **exactly 3 days** to harvest before it withers and is permanently lost.

---

## 🎯 The 3 Curriculum Tasks

| # | Task | Difficulty | Start $ | Max Days | Challenge |
|---|---|---|---|---|---|
| 1 | Single Crop Stable | 🟢 Easy | $200 | 30 | Double your starting money |
| 2 | Multi-Crop Market Timing | 🟡 Medium | $150 | 45 | Profit across all 3 crops at peak prices |
| 3 | Drought Survival | 🔴 Hard | $100 | 60 | Survive drought events + tropical spoilage |

**Task 2 — Market Timing Detail:**  
The grader measures *when* you sell, not just *that* you sell. Up to 40% of your score comes from selling during price peaks above the base baseline.

**Task 3 — Drought Mechanics:**  
Every 5th day is a drought day: zero precipitation + **−15L forced tank drain**. Agents must actively `pump_water` from the underground aquifer to survive. Tropical spoilage (3%/day) doubles the urgency of liquidating storage fast.

---

## 🏆 Grading Formulas

Every episode produces a final grade in `[0.0, 1.0]` accessible via `observation.metadata["grade"]`.

**Task 1 — Single Crop Stable:**
```
score = clamp(net_worth / (initial_money × 2.0), 0, 1) − min(0.20, withered × 0.05)
```

**Task 2 — Multi-Crop Market Timing:**
```
score = 0.6 × profit_score + 0.4 × timing_score − min(0.30, withered × 0.10)

timing_score = clamp(premium_revenue / (total_revenue × 0.3), 0, 1)
```

**Task 3 — Drought Survival:**
```
score = 0.5 × profit_score + 0.3 × survival_score + 0.2 × resilience_score − min(0.40, withered × 0.15)

resilience_score = healthy_days / max_days   # days where ≥ 2 plots had health ≥ 0.6
```

> A score ≥ **0.80** is professional tier.

---

## 🎁 Reward Shaping

| Action / Event | Reward | Notes |
|---|---|---|
| `plant` | **+0.2** | Commit bonus |
| `irrigate` (rescue) | **+0.5** | Moisture was critically low (<0.25) |
| `irrigate` (normal) | **+0.1** | |
| `harvest` | **up to +1.0** | Scales with `yield / max_yield` |
| `sell` | **+0.3 + premium** | Bonus for above-base price premium |
| `wait` (crops growing) | **+0.05/plot** | Smart patience |
| Health maintenance | **+0.1/plot/day** | Passive per living, healthy plot |
| Crop withers | **−5.0** | Hard penalty, once per wither event |
| Wait with mature plots | **−0.3/plot** | Every idle day risks permanent loss |
| Overwatering | **−0.15 health** | Moisture > 0.9 damages the crop |
| Storage overflow | **−0.3** | Lost kg on harvest |
| Invalid action | **−1.0** | Hard rejection |
| **Terminal bonus** | **up to +10.0** | `max(0, (net_worth/start − 1) × 5)` |

---

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -e .

# Start the environment server + Gradio UI
uvicorn server.app:app --host 0.0.0.0 --port 7860
# Open http://localhost:7860 for the interactive dashboard
```

### Run the LLM Agent (Inference)
```bash
export HF_TOKEN="hf_your_token_here"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export FARMING_ENV_URL="http://localhost:7860"
export MAX_STEPS=60

python inference.py
# → saves baseline_results.json
```

### Run a Single Task
```bash
export FARMING_TASK_ID=3   # 1=easy, 2=medium, 3=hard
python inference.py
```

### Docker
```bash
docker build -t farming-sim .
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx farming-sim
```

---

## 🤖 LLM Agent Architecture

`inference.py` implements a **stateless LLM-as-Agent** loop compatible with any OpenAI-format API:

```
System Prompt  ──►  farming strategy rules (one-shot)
                         │
Observation    ──►  text_summary + valid_actions list
                         │
LLM Response   ──►  raw JSON string: {"action_type": "harvest", "plot_id": 2}
                         │
Parser         ──►  parse_action() → validate_action() → FALLBACK on malformed
                         │
Environment    ──►  env.step(action) → new obs + reward
                         │
History Buffer ──►  last 4 steps kept in context window
```

| Parameter | Default |
|---|---|
| Model | `Qwen/Qwen2.5-72B-Instruct` |
| Temperature | `0.2` |
| Max tokens | `150` |
| Fallback | `{"action_type": "wait"}` |

---

## 🌐 API Reference

| Method | Path | Body | Description |
|---|---|---|---|
| `GET` | `/health` | — | Liveness check |
| `POST` | `/reset` | `{"task_id": 1\|2\|3}` | Reset episode → returns initial `FarmObservation` |
| `POST` | `/step` | `{"action": FarmAction}` | Take one action → returns new `FarmObservation` |
| `GET` | `/state` | — | Full internal `FarmState` |
| `GET` | `/` | — | Gradio interactive dashboard |

---

## 📁 Repository Structure

```
FarmSimulation/
├── server/
│   ├── app.py                  ← OpenEnv create_app() entry + Gradio mount
│   ├── farming_environment.py  ← Full physics engine + 10 action handlers (1026 lines)
│   ├── gradio_app.py           ← Glassmorphic dark-mode dashboard UI
│   ├── tasks.py                ← EpisodeRecord + grade_task1/2/3 graders
│   └── requirements.txt
├── models.py                   ← Pydantic schemas + all SEED_CONFIG / CLIMATE_CONFIG constants
├── inference.py                ← OpenEnv compliance runner (LLM agent loop)
├── openenv.yaml                ← OpenEnv manifest: name, version, tasks, action/obs classes
├── pyproject.toml              ← Dependencies (uv-compatible)
├── Dockerfile                  ← Python 3.11-slim, exposes :7860
├── baseline_results.json       ← Last inference run output
└── test_phase{2-7}.py          ← Phase-gated TDD test suite
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Run the test suite: `pytest test_phase*.py -v`
4. Submit a pull request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for the Meta Hackathon — an OpenEnv-compatible environment for evaluating LLM agents on real-world agricultural planning tasks.*
