# Changelog

All notable changes to FarmSimulation are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2026-04-08

### 🎉 Initial Release — Meta PyTorch Hackathon Submission

#### Added
- **FarmingEnvironment** — Full 5-pass daily physics simulation engine (1,400+ lines)
  - Pass 1: Hydrology (precipitation → aquifer + tank recharge)
  - Pass 2: Pedology (FAO-56 Penman-Monteith soil moisture evapotranspiration)
  - Pass 3: Ecology (exponential pest/weed escalation with pesticide protection system)
  - Pass 4: Physiology (compound health damage + recovery mechanic)
  - Pass 5: Economics (sinusoidal market cycles + Almgren-Chriss impact model)
- **11 Action Types**: `buy_seeds`, `plant`, `irrigate`, `harvest`, `sell`, `pump_water`, `apply_fertilizer`, `spray_pesticide`, `pull_weeds`, `buy_plot`, `wait`, `end_day`, `clear`
- **Labor Hour System**: 10h/day budget; overflow auto-triggers `end_day`
- **3-Task Curriculum**:
  - Task 1 (Easy): Single crop, $200, 30 days, temperate climate
  - Task 2 (Medium): Multi-crop market timing, $150, 45 days, full climate rotation
  - Task 3 (Hard): Drought survival, $100, 60 days, forced tank drains + spoilage
- **3 Rotating Climates**: temperate → arid → tropical (10-day cycle)
- **OpenEnv Core** REST API: `/reset`, `/step`, `/state`, `/health`
- **Gradio Dashboard**: Glassmorphic dark-mode UI ("Soil and Leaf" theme)
- **LLM Inference Script** (`inference.py`): Stateless agent loop with fallback action parsing
- **Dual-LLM Architecture** (`dual_llm_inference.py`): Two-model debate framework (experimental)
- **Pydantic Models** (`models.py`): Full type-safe schemas for Action, Observation, State
- **Phase-Gated Test Suite**: `test_phase2` through `test_phase7`
- **4-Layer Validation Script** (`validate-submission.sh`)
- **Robustness Validator** (`robustness_validation.py`): Determinism + skill gradient proof

#### Physics Details
- **Crop coefficients (Kc)**: Wheat 0.80 · Rice 1.10 · Corn 1.20 (FAO-56 aligned)
- **Harvest window**: 3 days after `mature` before permanent wither
- **Market cycle**: 20-day sine wave, phase-offset per crop (wheat+0d, rice+7d, corn+13d)
- **Price floor**: 30% of base price minimum
- **Aquifer capacity**: 1,000L · Initial: 500L · Pump capacity: 50L/action ($5)
- **Water tank capacity**: 100L · Initial: 80L · Irrigation cost: 15L/plot

---

## [0.9.0] — 2026-04-06

### Phase 2 Compliance Fixes

#### Fixed
- **Critical**: Score `0.0` on bankruptcy triggered Phase 2 validation failure
  - `_handle_episode_termination()` now returns `max(0.01, min(0.99, raw_bonus))`
  - `grade_episode()` now triple-clamps output to `(0.01, 0.99)`
  - `inference.py` per-step reward clamping: `<= 0.0 → 0.01`, `>= 1.0 → 0.99`
- **Critical**: Inference logger format mismatched validator regex
  - `log_step()` now emits exactly: `[STEP] step=N action='...' reward=N.NN done=true|false error='...'`
  - `log_end()` now emits exactly: `[END] success=true|false steps=N score=N.NN rewards=...`
- `parse_action()` now correctly unwraps nested `{"action": {...}}` LLM responses

---

## [0.8.0] — 2026-04-05

### Balance & Realism Overhaul

#### Changed (from FIXES_APPLIED.md)
- **Health Recovery** (new feature): Crops now recover +0.03/day health when conditions are optimal (moisture 0.25–0.85, all NPK ≥ 0.25, no pests)
- **Difficulty-scaled wither penalty**: Easy −2.0 · Medium −3.5 · Hard −5.0 (was flat −5.0)
- **Difficulty-scaled passive reward**: Easy +0.15 · Medium +0.12 · Hard +0.10 per healthy plot
- **Reduced health degradation**: Drought −0.07/day (was −0.10), NPK −0.07/day (was −0.10), overwater −0.12/day (was −0.15)
- **Overwater threshold**: Now triggers at >0.85 moisture (was >0.9)
- **Fertilizer boost**: +0.3 per NPK nutrient (was +0.4) — more strategic timing required
- **Irrigation sweet spot**: 0.25–0.85 (60% safe range)

#### Impact
- Easy mode: Recoverable from 1–2 withered crops; 13-day recovery window (was impossible)
- Medium mode: Requires careful play; forgiving of single errors
- Hard mode: Maintains high challenge level

---

## [0.7.0] — 2026-04-04

### Gradio UI Overhaul

#### Added
- Glassmorphic dark-mode Gradio dashboard with "Soil and Leaf" theme
- Live market price ticker with trend arrows
- 4-panel plot visualization with per-plot health/moisture/pest overlays
- Action history log with reward annotations
- 7-day weather forecast panel
- Documentation tab with interactive HTML docs

---

## [0.6.0] — 2026-04-03

### Labor Hour System

#### Added
- 10-hour daily labor budget per in-game day
- Per-action labor costs (plant=2h, harvest=4h, irrigate=0.5h, etc.)
- Auto-`end_day` trigger when labor budget overflows
- `labor_remaining` field in `FarmObservation`
- `end_day` action for explicit day advancement

---

## [0.5.0] — 2026-04-02

### Market & Economics

#### Added
- Almgren-Chriss market impact model for sell orders
  - Temporary slippage: 0.5% per 10kg
  - Permanent price impact: 1% per 10kg (floor at 50% of mid-price)
- 7-day rolling price average (`avg_7d`) in `MarketPrice`
- `trend` field: normalized price direction signal [-1, 1]
- Per-crop price history tracking for LLM context

---

## [0.4.0] — 2026-04-01

### Task Curriculum

#### Added
- `tasks.py` with `EpisodeRecord`, `grade_task1()`, `grade_task2()`, `grade_task3()`
- Task 2: timing score (40% weight) rewarding above-baseline price sales
- Task 3: resilience score (20% weight) counting days with ≥2 healthy plots
- Wither penalty scaling by task difficulty
- `sell_events` list in `EpisodeRecord` for timing analysis

---

## [0.3.0] — 2026-03-31

### Climate System

#### Added
- 3-climate rotation: temperate → arid → tropical (10-day cycles)
- Stochastic daily weather: rain probability and intensity by climate
- Extreme temperature effects: >32°C or <10°C freezes crop growth
- Natural pest die-off in cold temperatures (<10°C, 30% chance)
- Tropical spoilage (3%/day storage degradation in Task 3)

---

## [0.2.0] — 2026-03-30

### HTTP API

#### Added
- FastAPI server with OpenEnv Core integration
- `/reset`, `/step`, `/state`, `/health` REST endpoints
- Session persistence (singleton environment instance)
- Gradio app mounted at root path `/`
- WebSocket-compatible session management

---

## [0.1.0] — 2026-03-29

### Foundation

#### Added
- Initial `FarmingEnvironment` class with `reset()` and `step()`
- 4 land plots per episode
- Basic crop lifecycle: seedling → growing → mature → harvest
- Resource management: water tank, aquifer, money, seed inventory, storage
- Core action handlers: buy_seeds, plant, irrigate, harvest, sell
- `models.py`: FarmAction, FarmObservation, FarmState, PlotState, MarketPrice Pydantic models
- `openenv.yaml` manifest
- `pyproject.toml` with uv-compatible dependencies
- `Dockerfile` with Python 3.11-slim base
