# Contributing to FarmSimulation

Thank you for considering contributing to FarmSimulation! This environment is built for the Meta PyTorch Hackathon and the OpenEnv framework. We welcome improvements that enhance scientific realism, agent strategy depth, or developer experience.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Project Architecture](#project-architecture)
- [Testing Requirements](#testing-requirements)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Contribution Areas](#contribution-areas)

---

## Code of Conduct

Be respectful and constructive. Focus on ideas, not individuals. We follow the standard open-source community norms.

---

## How to Contribute

1. **Open an issue first** for significant changes — describe the problem and proposed solution before writing code.
2. **Fork the repo** and create a descriptive branch:
   - `feat/crop-rotation-mechanics`
   - `fix/moisture-decay-arid-climate`
   - `docs/api-reference-update`
3. **Make changes** following the architectural rules below.
4. **Run the full test suite** before submitting.
5. **Submit a PR** with a clear description of what changed and why.

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/FarmSimulation.git
cd FarmSimulation

# Install in editable mode with all dev dependencies
pip install -e .

# Start the development server
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Run tests
pytest test_phase*.py -v

# Run the full validation suite
./validate-submission.sh
```

---

## Project Architecture

Understanding the code structure before contributing:

### Core Modules (Strict Ownership)

| Module | Owns | Do NOT touch from other modules |
|---|---|---|
| `models.py` | All Pydantic schemas, all constants | — |
| `server/farming_environment.py` | All physics simulation | API routing |
| `server/tasks.py` | Episode grading only | Physics simulation |
| `server/app.py` | FastAPI/Gradio wiring only | Business logic |
| `inference.py` | LLM agent loop only | Environment internals |

### Key Invariants

These must NEVER be broken:

1. **Score range**: All grades must be strictly in `(0.01, 0.99)` — never `0.0` or `1.0`
2. **Determinism**: Given the same `seed`, `reset()` must produce identical trajectories
3. **API surface**: `/reset`, `/step`, `/state`, `/health` signatures must not break
4. **Concurrent sessions**: `SUPPORTS_CONCURRENT_SESSIONS = True` must remain valid
5. **Reward logging format**: `log_step()` output must match `[STEP] step=N action=... reward=... done=...` regex

---

## Testing Requirements

All PRs must pass the full phase-gated test suite:

```bash
pytest test_phase2.py -v   # Score range (0.01–0.99) validation
pytest test_phase3.py -v   # Physics determinism
pytest test_phase4.py -v   # Labor hour system
pytest test_phase5.py -v   # Session persistence
pytest test_phase7.py -v   # Full end-to-end integration
```

For physics changes, additionally run:

```bash
python robustness_validation.py   # Determinism + skill gradient proof
python verify_all.py              # Extended verification suite
```

---

## Pull Request Guidelines

Your PR description should answer:

1. **What** changed and in which files
2. **Why** — the problem this solves or feature it enables
3. **How** — brief technical explanation of the approach
4. **Test** — which tests you ran and their results
5. **Backward compatibility** — does this change any API, data format, or score formula?

---

## Contribution Areas

### 🔬 Physics & Realism
- New crop types (tomatoes, soybeans, etc.) with realistic Kc values
- Soil pH system affecting NPK absorption efficiency
- Wind speed effects on evapotranspiration
- Crop rotation bonuses (nitrogen fixation from legumes)

### 🎯 Tasks & Curriculum
- Task 4: Multi-farm portfolio management
- Task 5: Climate change scenario (increasing drought frequency)
- Custom scenario API for researchers

### 🤖 Agent Strategies
- Heuristic baseline agents with different strategies (TWAP, momentum, mean-reversion)
- Multi-agent cooperative farming
- Curriculum learning integration

### 📊 Observability
- Richer `text_summary` with trend indicators (↑↓→)
- Plot-level health trajectory charts
- Market regime detection signals

### 🛠️ Developer Experience
- Better error messages for invalid actions
- Replay system for episode recording/playback
- OpenAI Gym-compatible wrapper

---

## Questions?

Open a GitHub Issue with the label `[question]` or start a Discussion.
