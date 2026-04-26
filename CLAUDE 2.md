# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`farming-env` is an **OpenEnv-compatible** RL environment for the Meta PyTorch Hackathon. A FastAPI server exposes `/reset`, `/step`, `/state`, `/health` (the OpenEnv protocol); a Gradio dashboard is mounted at `/` on the same process. Agents (LLMs or heuristics) drive a deterministic farm-management simulation against three graded tasks.

The companion docs `IMPLEMENTATION_PLAN.md`, `TRAINING.md`, and `Planning/` capture the current hackathon strategy — read these before making non-trivial changes to physics, rewards, or the training loop, since several files (`README.md`, `META_HACKATHON_ANALYSIS.md`, `ROUND1_COMPLIANCE_CHECKLIST.md`) describe an earlier compliance milestone and are partially stale.

## Common commands

Server / dashboard (single process, both REST API and UI):
```bash
pip install -e .                                  # editable install via pyproject.toml
uvicorn server.app:app --host 0.0.0.0 --port 7860 # serves /reset /step /state /health + Gradio UI at /
```

Pick a task by env var before starting the server (the singleton picks this up once per process):
```bash
FARMING_TASK_ID=2 uvicorn server.app:app --port 7860   # 1=easy, 2=medium, 3=hard
```

Run the LLM evaluator against a running server:
```bash
export HF_TOKEN=hf_xxx
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export FARMING_ENV_URL="http://localhost:7860"
python inference.py                               # all 3 tasks → baseline_results.json
FARMING_TASK_ID=2 python inference.py             # single task
```

Train (GRPO + LoRA + 4-bit; needs `pip install -e .[train]` or `pip install -r requirements-train.txt`):
```bash
python train.py --tiny --num-iterations 2         # 30s smoke test
python train.py --task-id 1 --num-iterations 50 --group-size 4 --max-steps 30
```
See `TRAINING.md` for the full flag table; the Colab path is `notebooks/train_grpo.ipynb`.

Tests (plain `pytest`, no fixtures runner):
```bash
pytest test_phase2.py -v       # score range (0.01, 0.99) clamping
pytest test_phase3.py -v       # determinism (seed=42 reproducibility)
pytest test_phase4.py -v       # labor-hour budget
pytest test_phase5.py -v       # session persistence
pytest test_phase7.py -v       # full integration
pytest test_phase3.py::test_name -v   # single test
```
Other top-level `verify_*.py` and `test_*.py` scripts are standalone (run with `python …`), not pytest modules.

Submission validator — **note the README is out of date**. Actual usage:
```bash
./validate-submission.sh <hf_space_ping_url> [repo_dir]   # pings Space, docker builds, runs `openenv validate`
```
It does NOT run the pytest phase suite; do that separately when you need it.

Docker (matches the HF Space build):
```bash
docker build -t farming-sim .
docker run -p 7860:7860 -e HF_TOKEN=hf_xxx farming-sim
```

## Architecture

### Process layout — one server, two surfaces

`server/app.py` builds a single ASGI app: `openenv.core.create_app(...)` provides the REST endpoints, then `gr.mount_gradio_app(app, ui, path="/")` overlays the dashboard. A module-level `GLOBAL_ENV` singleton is returned by `make_env()` so that stateless HTTP calls (`/step`) preserve episode state across requests — this is intentional and load-bearing for the OpenEnv runner. Don't replace it with per-request instantiation.

### Where things live

- `models.py` (repo root, **not** under `server/`) — every Pydantic schema (`FarmAction`, `FarmObservation`, `FarmState`, `PlotState`, `MarketPrice`, `ClimateState`) and **every tunable constant** (`SEED_CONFIG`, `CLIMATE_CONFIG`, `STORAGE_CAPACITY`, `IRRIGATION_COST`, `PLOT_BASE_COST`, etc.). Constants are imported by name into the engine; if you add one, export it here.
- `server/farming_environment.py` (~1.4k lines) — the physics engine and action dispatcher. Subclasses `openenv.core.Environment[FarmAction, FarmObservation, FarmState]` and sets `SUPPORTS_CONCURRENT_SESSIONS = True`. The daily clock is the **5-pass cycle** (hydrology → pedology/FAO-56 ET → ecology/pest growth → plant physiology → market drift) executed inside the `end_day` handler. Crops have a 3-day harvest window after `mature` before they wither — `days_mature` tracks the countdown.
- `server/tasks.py` — `EpisodeRecord` dataclass plus `grade_task1/2/3` and `grade_episode` dispatcher. Each grader returns a score clamped to `(0.01, 0.99)`; this clamp is mandatory for OpenEnv Phase 2 compliance and is also reapplied on the inference and termination paths.
- `server/audit_utils.py` — `calculate_state_fidelity` / `calculate_tactical_report` used by the env to surface debugging signals; not on the hot path.
- `server/scenario_engine.py`, `server/scenario_definitions.py` — pre-baked scenarios for demo/validation; separate from the task curriculum.
- `server/agents/heuristic.py`, `server/agents/hybrid.py` — non-LLM baselines used by `server/baseline_inference.py` and (planned) BC warm-start in `train.py`.
- `server/gradio_app.py` — dashboard UI; pure presentation, calls into the same `make_env()` singleton.
- `inference.py` — OpenEnv-compliance evaluator: stateless OpenAI-format LLM loop with a regex JSON extractor and a `wait` fallback on parse failure. Reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `FARMING_ENV_URL`, `FARMING_TASK_ID`, `MAX_STEPS`, `EPISODES`. Output goes to `baseline_results.json`.
- `dual_llm_inference.py` — experimental two-LLM debate variant; do **not** treat as the canonical inference path.
- `train.py` — GRPO over full episodes (group-relative advantage, no value net), LoRA-only updates on a frozen 4-bit base. Per-step `loss.backward()` with one `optimizer.step()` per iteration to keep memory bounded on T4. See `TRAINING.md`.

### Import-path quirk

`models.py` sits at the repo root and is added to `sys.path` separately in both `server/app.py` (via runtime `sys.path.insert`) and the `Dockerfile` (`ENV PYTHONPATH=/app/server:/app`). Modules under `server/` therefore use a try/except dance:

```python
try:
    from tasks import EpisodeRecord, grade_episode          # when run from server/
except ImportError:
    from server.tasks import EpisodeRecord, grade_episode   # when run from repo root
```

If you add new cross-module imports inside `server/`, mirror this pattern — picking only one form will break either the editable install or the Docker image.

### Reward / score clamping invariant

Every reward and grade is forced into the open interval `(0.01, 0.99)` — never `0.0` and never `1.0` — at three layers: the grader (`server/tasks.py`), the environment's terminal/step paths (`server/farming_environment.py`), and the inference runner (`inference.py`). This is an OpenEnv Phase 2 hard requirement; if a test like `test_phase2.py` fails after a change, the regression is almost always a missing clamp on a new reward branch, not a mis-tuned coefficient.

### Action / observation contract

Actions are strict Pydantic — `FarmAction.action_type` is an enum and many actions require companion fields (`plot_id`, `seed_type`, `quantity`). The env returns a `valid_actions` list inside the observation; agents (and graders) treat invalid actions as a hard `−1.0` penalty rather than silently no-op'ing. Each real-world day has a **10-hour labor budget**; exceeding it auto-triggers `end_day`. `end_day` itself costs 0 hours — it's the explicit "advance the simulation" lever.

## Conventions worth preserving

- All physics constants and Pydantic models live in `models.py`. Don't sprinkle magic numbers into `farming_environment.py`.
- Keep the public REST surface (`/reset`, `/step`, `/state`, `/health`) backward-compatible — the OpenEnv runner and the HF Space depend on it.
- The 5-pass daily cycle order matters (rain before evaporation, pests before health update, market last). Reordering changes scores even with the same seed.
- Determinism: `seed=X` must produce identical trajectories. `test_phase3.py` is the canonical guard; if you introduce non-determinism (e.g., a new `random.random()` call without seeding), it will catch it.
- `models.py` is `py-modules = ["models"]` in `pyproject.toml`; it's not packaged under `server/`. Don't move it without updating the build config and the Dockerfile `PYTHONPATH`.

## Important Rules That you have to follow

-RULE 1 — START OF EVERY CLAUDE CODE SESSION:
    git fetch origin
    git checkout main && git pull --ff-only
    git checkout <my-branch>
    git rebase main
    → resolve conflicts immediately if any. If unsure, ping the file's primary owner before resolving.

-RULE 2 — NEVER:

    git push origin main          ← FORBIDDEN
    git push --force origin main  ← FORBIDDEN
    git commit --no-verify        ← FORBIDDEN (fix the failing test instead)
    Edit a file outside your ownership list (see §19 quick-reference table)

-RULE 3 — PR CADENCE:
    Open a PR every 2-3 hours of work. Don't accumulate 8h of changes.
    Title: "<wsN.M>: <one-line summary>"
    Body: bullet list of changes + "I tested with: ..." + "@<other-person> please check: ..."
    Reviewer SLA: 30 minutes. Approve or comment.

-RULE 4 — STATUS.MD UPDATES:
    Both people commit tiny STATUS.md updates every hour even if "still on same task."
    Silence is a smell. If the other person hasn't pinged in 90 minutes, voice-call them.

-RULE 5 — HANDOFFS (§4):
    HANDOFF #1 at H+5  (A's WS1 merge → B can use real narrative_text)
    HANDOFF #2 at H+7  (A's WS2 merge → B trains on hardened economy)
    HANDOFF #3 at H+9  (A's WS3 merge → B uses RubricComposer scores)
    HANDOFF #4 at H+22 (B's training done → A re-runs robustness with trained tier)
    Don't proceed past a HANDOFF until both branches are synced.