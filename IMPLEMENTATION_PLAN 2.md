# FarmSimulation — Finalist Round Implementation Plan

**Project:** Meta PyTorch RL Hackathon 2026 (OpenEnv) — FarmSimulation
**Round:** Finalist (Round 2)
**Repo:** `/mnt/d/FarmSimulation` (branch: `main`)
**Date:** 2026-04-25 (on-site training credits just received)
**Time budget:** 30 hours hard cap. Final 3 hours = README + video.
**Goal:** Top-10 finalist finish.

---

## 0. Theme Alignment (state explicitly in submission)

Per `Planning/Hackathon Themes & Judging Criteria.md`, FarmSimulation primarily targets:

- **Theme #2 — (Super) Long-Horizon Planning & Instruction Following.** "Strategic resource management worlds" is given as an example. 30/45/60-day episodes with delayed rewards, multi-step crop cycles, and context-memory pressure fit perfectly.
- **Theme #3.1 — World Modeling: Professional Tasks.** "Economic simulations with feedback" is given as an example. Dynamic markets, resource feedback loops, and partially observable hydrology/climate fit perfectly.

State this dual-theme alignment **in the README hero block** — judges score Theme fit implicitly under the 40% Innovation axis.

---

## 1. Context & Brutal Reality Check

We advanced past Round 1. The HF Space is already live and passed health checks. The OpenEnv API, physics engine (`server/farming_environment.py`, 1411 lines), 3-task curriculum, structured-log inference (`inference.py`), and a custom GRPO training loop (`train.py`, 598 lines) are implemented. Heuristic baseline + 72B zero-shot baselines are recorded.

But `Planning/Self-Assessment Q&A v2.md` — the canonical "brutal reality check" — flags four blockers we must close in this 30-hour window:

| Blocker | v2 quote | Cost to fix |
|---|---|---|
| **B1. Toy or benchmark?** Clean JSON state is solvable by 2015 tabular RL. "*We are building an environment for a mouse, not a supercomputer.*" | §1 | 4h env pivot |
| **B2. Reward exploits.** Spam-water + HFT exploits will dominate any GRPO run. "*The training curve will represent cheating, not farming.*" | §2 | 2h reward hardening |
| **B3. Wasted potential.** "*Driving a Ferrari in a school zone.*" Quant strengths constrained to 4-plot loop; need volatility + sunk-cost dilemma. | §3 | overlapping with B2 |
| **B4. Meaningless training curves.** "*An upward loss curve just means the agent learned which buttons to click.*" Must train against the pivoted env so curves prove parsing+economic reasoning. | §4 | gated by B1+B2 |

Plus the operational gap surfaced in code review:
- **B5.** No training has actually been run. Baseline 0.01/0.01/0.01 across 3 tasks → zero evidence of learning → forfeits 20% improvement axis.
- **B6.** `README.md` is a 193-byte stub → forfeits storytelling axis (30%).
- **B7.** Existing GRPO loop is hand-built. Rubric explicitly requires **Unsloth or HF TRL** in Colab.

The 30-hour plan closes B1-B7 in the order v2 prescribes.

---

## 2. Strategic Recommendation (the GRPO question)

**Yes, GRPO-first is correct — but only after the v2 pivots land.** Sequence matches v2's "Immediate Action Items":

1. **Refactor Observation Space** (B1) — additive noisy-text layer over structured fields.
2. **Harden the Economy** (B2/B3) — labor friction, Almgren-Chriss slippage, patience cap, binary competence gate.
3. **Composable Rubric** — refactor monolithic graders into `RubricComposer` (cited explicitly by judging criteria: "*composable rubrics > monolithic scoring*").
4. **Run Training** (B4) — Unsloth + TRL `GRPOTrainer` on the pivoted env. Use HF on-site credit ($90 team) for dedicated L4 hardware, NOT free Colab.
5. **Storytelling** (B6) — README + video in last 3h.

**Why GRPO over alternatives** (per `Planning/RL Hackathon Project Analysis Guide.md`): OpenEnv SF #1 (Kube SRE Gym) and #3 (Play-gent) both used GRPO+TRL+LoRA. PPO needs full FT (won't fit reasonable hardware), DPO needs preference pairs we don't have, SFT-on-heuristic forfeits both improvement and pipeline axes. GRPO has no value network (50% VRAM savings), is LoRA-compatible, TRL-integrated, and the canonical winning recipe.

**Backup plan** (only if GRPO smoke shows no signal by H 12): switch headline to **chain-of-thought + scratchpad prompt engineering** — even 0.01→0.15 from prompt changes alone covers the 20% improvement axis. Hold in reserve.

---

## 3. Narrative Reframe (per v2 §3)

The submission is **NOT** "AI plays Stardew Valley."

The submission **IS**: *"A long-horizon economic and resource-management simulation where an agent must synthesize noisy textual data to survive dynamic market and climate volatility."*

This sentence goes verbatim in the README hero, the video opening, and any pitch deck. Every workstream below serves this reframe.

---

## 4. 30-Hour Parallel Timeline (Two People, Two Claude Code Sessions)

Two team members work in parallel on separate machines. Roles locked at H 0 (see §19):

- **Person A — Env Lead.** Owns `server/`, `models.py`, `inference.py`, `openenv.yaml`, `verify_*.py`, `robustness_validation.py`, HF Space deploy.
- **Person B — Training & Story Lead.** Owns `notebooks/`, `train.py`, `assets/*.png`, `README.md`, video, HF Hub upload.

Both work simultaneously throughout the 12-hour blocks. **HANDOFF** rows mark hard sync points (PR merge → both rebase before continuing).

### Block A — env pivot + smoke (Hours 0-12)

| Hour | Person A (Env Lead) | Person B (Training & Story Lead) |
|---|---|---|
| H 0-1 | Repo hygiene: create `STATUS.md`, archive Planning meta-docs, lock pivot spec, branch `a/ws1` | Colab setup: `pip install unsloth trl==0.11.4`, HF + wandb login, verify all 3 team accounts can see $30 credit, branch `b/ws4-notebook` |
| H 1-3 | **WS1.1:** add `narrative_text: str` to `models.py` `FarmObservation`; sketch `_render_narrative_observation()` in `farming_environment.py` | **Notebook cells 1-4:** `!pip install` cell, HF/W&B login, `FastLanguageModel.from_pretrained(unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit, fast_inference=True)`, `get_peft_model(...)` with `use_gradient_checkpointing="unsloth"` |
| H 3-5 | **WS1.2:** 4 narrative components (journal, weather report with 25% lie rate, neighbor locust email, market gossip); switch `inference.py` to consume `narrative_text` with structured fallback; PR `a/ws1` | **Notebook cells 5-7:** `FarmEnvClient` HTTP client (lift from `train.py:124-153`), `parse_action()` helper, **5 reward functions** per §20.1 + §21.2, scored against a stubbed local env (real env hooked at H 9-10) |
| **H 5** | **HANDOFF #1:** A merges `a/ws1` → `main`. B `git pull --rebase main`. Notebook now has real `narrative_text` field to consume. | |
| H 5-7 | **WS2:** 4 reward-hardening fixes (labor decrement, patience cap, Almgren-Chriss superlinear slippage, binary competence gate in `tasks.py`); run `verify_economics.py` + `verify_actions.py`; PR `a/ws2` | **Notebook cells 8-10:** build `prompt_dataset` (200 rows of seeded initial-state prompts), `GRPOConfig` with §20.2 locked hyperparameters (`loss_type="dapo"`, `beta=0.0`, `max_grad_norm=0.1`, etc.), `GRPOTrainer` init |
| **H 7** | **HANDOFF #2:** A merges `a/ws2` → `main`. B rebases. Reward exploits closed before training touches the env. | |
| H 7-9 | **WS3:** `RubricComposer` + `Dimension` + `Gate` classes in `tasks.py`; extend `openenv.yaml` with composite grader blocks per task; update `test_phase*.py` for new return shape; PR `a/ws3` | **Notebook cells 11-13:** plot generation scaffolding, `model.save_pretrained_merged()` + `huggingface_hub.upload_folder()` skeleton; metadata-printing final cell; smoke-test full notebook on T4 with stubbed env; commit `b/ws4-notebook` (don't merge yet) |
| **H 9** | **HANDOFF #3:** A merges `a/ws3` → `main`. Both rebase. Pivoted env + composable rubric live on `main`. | |
| H 9-10 | Re-baseline on pivoted env: random agent + heuristic agent + 72B zero-shot via `inference.py`; save `baseline_results_pivoted.json`; commit | Pull `main`; swap `FarmEnvClient` from stub to live HF Space URL; verify `narrative_text` field is non-empty in env response |
| H 10-12 | Run `robustness_validation.py` on pivoted env; confirm skill gradient (random < heuristic < 72B) holds; if inverted → dial down forecast lie rate 25%→15% per §13 risk register | **GRPO smoke run:** 10 iters × K=2 on Task 1, free Colab T4 OR $1-2 on dedicated L4 if account-gating issues; bisect per §20.6 if flat (LR 5e-6→2e-6, then K 2→4, then check advantages); commit smoke logs |

### SLEEP (Hours 12-18, 6h hard block, both team members)

### Block B — full training + evidence + packaging (Hours 18-30)

| Hour | Person A (Env Lead) | Person B (Training & Story Lead) |
|---|---|---|
| H 18-19 | Pull `main`; redeploy HF Space with pivoted env; run `validate-submission.sh <space-url>` + `openenv validate`; verify `/reset` and `/step` endpoints; check no MCP reserved-name collisions (`reset/step/state/close`) | Pull main + smoke results; spin dedicated **L4 Space** ($0.80/h) for training; pre-warm vLLM cache; verify $90 HF credit intact and not gated |
| H 19-21 | Heuristic agent baseline (10 episodes × 3 tasks) + 72B zero-shot baseline via `inference.py` against live pivoted Space; save `baselines_pivoted.csv` | **Full GRPO run:** 50 iters × K=4 × Task 1 on L4 (~1.5h compute); monitor reward + loss curves in W&B; checkpoint adapter every 10 iters |
| H 21-22 | Pre-submit cleanup: move `STRATEGIC_SUMMARY.md`, `META_HACKATHON_ANALYSIS.md`, `ROUND1_COMPLIANCE_CHECKLIST.md`, `FIXES_APPLIED.md`, `LLM_JUDGE_STRATEGY.md`, `MATHEMATICAL_FOUNDATION.md`, `farming_realism_analysis.md` to `Planning/archive/`; delete `main.py` + `honey_app.py`; PR `a/cleanup` | While training finishes: scaffold `scripts/make_plots.py` (5 matplotlib plots from W&B run history); draft README outline using §9 story arc |
| **H 22** | **HANDOFF #4:** B's training run completes. Adapter saved locally. Both pull main. | |
| H 22-24 | Pull main; with the trained adapter present, re-run `robustness_validation.py` adding the "0.5B trained" tier; produce skill-gradient summary (random < heuristic < 72B zero-shot < 0.5B trained) | **Evidence harvest:** generate 5 PNGs in `assets/` — `reward_curve.png`, `loss_curve.png`, `baseline_comparison.png` (5 bars on **same axes**), `action_shift.png`, `trajectory_diff.png`; commit |
| H 24-25 | Hand A's robustness summary + annotated observation example + action JSON sample to B for README inclusion; run `verify_all.py` final pass | `huggingface_hub.upload_folder()` → push LoRA adapter to `<team>/farm-grpo-qwen0.5b-task1`; add Hub model card with usage snippet pointing back to FarmSim env |
| H 25-27 | Review B's README draft with focus on env description accuracy, physics claims, and reward design paragraph; commit minor wording edits via PR comments only | **Full README rewrite** (replace 193-byte stub): hero block with Theme #2 + #3.1 tags + verbatim narrative reframe; problem; environment with annotated screenshot; Reward design paragraph from §21.3; Training section; Results with all 5 plots embedded; Try-it-yourself; Future work citing PRMs/PBRS/BiPaRS/RUNE/IRD per §21 |
| H 27-29 | Test the Colab notebook **on a fresh runtime** end-to-end (one full re-run); verify no ImportError, NameError, or runtime crash; if broken — page B immediately | Record <2:00 video (OBS/Loom screen capture + pre-written VO); upload unlisted to YouTube; add link to README hero (NOT committed as file). HF blog post fallback if video overruns 1.5h |
| H 29-30 | Final engineering pass: `validate-submission.sh` + `openenv validate` + `verify_all.py` against live Space; sign off on §15 engineering checklist | Final story pass: confirm all 5 plots embed inline on GitHub, all cross-links resolve (Space, Hub, Colab, video). Submit GitHub URL to hackathon portal |

### Hard rules

- **Both rebase on `main` at the start of every Claude Code session** (`git fetch origin && git checkout main && git pull --ff-only && git checkout <my-branch> && git rebase main`).
- HANDOFF rows = mandatory PR merge + rebase. Don't proceed past a HANDOFF until both branches are synced.
- If anything in Block A overruns by >2h, drop pivot scope to journal-only (cut market gossip + neighbor email noise) and proceed.
- Block B cannot start late.
- Sleep block is non-negotiable; if behind schedule, cut video polish before cutting sleep.
- Status pings every hour in the team channel — silence is a smell.

### Critical-path dependencies

```
A: WS1 ──> WS2 ──> WS3 ──> redeploy ──> robustness ──> README review ──> final eng checklist
                              │                            ▲
                              ▼                            │
B: notebook scaffold ──> smoke ──> SLEEP ──> full train ──> evidence ──> Hub push ──> README ──> video ──> submit
```

A's WS1+WS2+WS3 must merge before B's smoke can run on real env (H 9). B's training must finish before A's final robustness re-run (H 22). All other work is parallelizable.

---

## 5. Workstream 1 — Refactor Observation Space (Hours 1-5) — closes B1 — **Owner: Person A**

**v2 §1 mandate:** *"Kill the clean arrays. The environment must be partially observable through natural language."*

**Decision: ADDITIVE LAYER, not a rewrite.** Keep all structured fields in `FarmObservation` (heuristic agent + graders + tests depend on them). Add a single `narrative_text: str` field. The LLM consumes the narrative; structured fields remain for graders, tests, and a parse-failure fallback.

### Files to modify

- **`server/farming_environment.py`**
  - Add `_render_narrative_observation(obs, day, rng) -> str` near the existing summary helpers.
  - Call from `_build_observation()`; attach to `obs.narrative_text`.
  - Add `noise_seed: int | None = None` parameter to `reset()` so judges can reproduce noise.
- **`models.py`** — add `narrative_text: str = ""` to `FarmObservation`.
- **`inference.py`** — switch the user-message construction (~line 229-242) to use `narrative_text` if present; structured fallback on parse failure.

### Narrative content (v2's three explicit examples + a fourth)

Per v2 §1 the narrative must include:

1. **Textual State Updates** — daily weather report snippet ("*Patchy clouds rolled in this morning. Temperature is mild, but the soil's holding moisture from yesterday's drizzle.*") replacing `climate_state = TEMPERATE`.
2. **Noisy Alerts** — neighbor email or radio-alert prose ("*Email from the Singh farm next door — they spotted a locust swarm heading west. Your fields might be in the path within 2-3 days.*") replacing `pest_level = HIGH`.
3. **Daily Farmer's Journal** — paragraph form derived from existing health/moisture/stage but expressed vaguely. health 0.4 → "looking rough today", moisture 0.3 → "soil's gone thirsty".
4. **Casual Resource & Market Summary** — "*About a third of a tank of water left. Roughly $40 in pocket. Storage shed has a few sacks of last harvest's wheat. Co-op newsletter mentioned drought worries for corn — Bob next door reckons wheat's climbing.*" Two-source market gossip aligns with true `MarketPrice.trend` 70% of the time; weather forecast lies 25% of the time. The agent must learn source-reliability weighting.

### What NOT to pivot

- **Action space stays structured JSON.** Free-text actions destroy GRPO parsing reliability — bad ROI for a 30h finals sprint. Defer to Round 3.
- **Don't rename fields.** Only add `narrative_text`; structured fields keep their names.
- **MCP reserved-name check:** judging criteria forbids `reset/step/state/close` as MCP tool names. Verify our action space (currently uses `wait`, `end_day`, `buy_seeds`, `plant`, `irrigate`, `pump_water`, `apply_fertilizer`, `spray_pesticide`, `pull_weeds`, `harvest`, `sell`, `clear`, `buy_plot`) doesn't collide. None of these are reserved, but add an automated test asserting it.

### Validation

```bash
python verify_all.py        # all phase tests must still pass (structured fields unchanged)
python inference.py         # confirm LLM still parses on Task 1 with narrative_text
```

---

## 6. Workstream 2 — Harden the Economy (Hours 5-7) — closes B2 & B3 — **Owner: Person A**

**v2 §2 + §3 mandate:** *"Resource Friction. Market Impact. Dynamic Volatility. Sunk Cost Fallacy."*

Four must-fix patches before GRPO touches the env:

### FIX 1 — Enforce labor hours (Spam-Water exploit close)

`farming_environment.py:81` defines `labor_hours_remaining` but no step handler decrements it. The agent has effectively infinite labor → can spam irrigate forever. Audit all `_handle_*` functions; deduct labor:

```python
LABOR_COSTS = {
    "irrigate": 1, "plant": 2, "harvest": 2, "spray_pesticide": 1,
    "pull_weeds": 2, "sell": 0.5, "apply_fertilizer": 1,
    "pump_water": 1, "buy_seeds": 0.5, "clear": 1, "buy_plot": 0,
    "wait": 0, "end_day": 0,
}
```

Reject action if `labor_hours_remaining < cost`. Reset to `10 / day` on `end_day`. No carryover.

### FIX 2 — Cap patience-farming (passive-reward exploit close)

`_daily_passive_reward()` (~line 799) grants 0.10-0.15/day per healthy plot uncapped. Across 30 days × 4 plots = up to 18 reward points for inaction. Patch:

```python
self._consecutive_wait_days += 1 if action == "wait" else 0; self._consecutive_wait_days = 0 if action != "wait" else ...
passive_multiplier = 1.0 if waits < 2 else 0.5 if waits < 4 else 0.0
```

### FIX 3 — Superlinear Almgren-Chriss slippage (HFT exploit close)

`farming_environment.py:638-639` has linear slippage (`0.005 * qty/10`). v2 §2 requires Almgren-Chriss style impact. Replace:

```python
slippage_pct = 0.005 * (qty / 10.0) ** 1.5
slippage_pct = min(slippage_pct, 0.30)  # cap at 30%
```

Prevents one-shot dump-everything strategies.

### FIX 4 — Binary competence gate (in graders)

In each `grade_task*` in `server/tasks.py`, add at top:

```python
if final_net_worth < initial_money:
    return 0.01  # bankrupt -> floor, regardless of stewardship
```

The README headline is *"agent learns to make money."* The grader must enforce that — otherwise the stewardship floor lifts a bankrupt agent to ~0.3.

### Sunk-Cost-Fallacy hook (v2 §3)

The pivoted env naturally surfaces v2's headline scenario: *"cut its losses on a dying, water-starved crop, or double down based on projected market yields?"* Make sure Task 2 or Task 3 includes at least one hand-engineered scenario where the optimal strategy involves abandoning a half-grown crop on a high-water-cost plot when corn futures spike. Use this as the README's marquee example trajectory.

### Validation

```bash
python verify_economics.py      # economy regressions
python verify_actions.py        # action validity
python robustness_validation.py # confirm skill gradient: random < heuristic < zero-shot LLM
```

If skill gradient inverts on the new env, the noise injection or slippage is too aggressive — dial down forecast lie rate from 25% to 15% and slippage exponent from 1.5 to 1.3.

---

## 7. Workstream 3 — Composable RubricComposer (Hours 7-9) — **Owner: Person A**

**Judging criteria mandate:** *"Uses OpenEnv's Rubric system thoughtfully (composable rubrics > monolithic scoring)."*

Refactor `server/tasks.py` from 3 monolithic graders into one rubric composer with named, weighted dimensions and explicit gates.

### `server/tasks.py`

```python
class Dimension:
    def __init__(self, name, weight, scorer): ...
    def compute(self, record) -> float: ...  # returns [0, 1]

class Gate:
    def __init__(self, name, condition, on_fail_score): ...
    def check(self, record) -> bool: ...

class RubricComposer:
    def grade(self, record, task_config) -> dict:
        for gate in self.gates:
            if not gate.check(record):
                return {"score": gate.on_fail_score, "dimensions": {}, "gated": gate.name}
        dim_scores = {d.name: d.compute(record) for d in self.dimensions}
        weighted = sum(dim_scores[d.name] * d.weight for d in self.dimensions)
        return {"score": clamp(weighted, 0.01, 0.99), "dimensions": dim_scores, "gated": None}
```

Per-dimension pure functions (each ~10 lines):
- `score_profit(record)` — `(net_worth / target) ** 1.0` clamped
- `score_stewardship(record)` — `healthy_day_fraction`
- `score_efficiency(record)` — `1.0 - waste_action_ratio`
- `score_resilience(record)` — `min_health_observed` (hard mode only)

### `openenv.yaml` extension

Per task:
```yaml
tasks:
  - id: task_1
    name: "Single Crop Stable"
    grader:
      type: composite
      dimensions:
        - name: profit
          weight: 0.7
          metric: net_worth_ratio
          target: 2.0
        - name: stewardship
          weight: 0.2
          metric: healthy_day_fraction
        - name: efficiency
          weight: 0.1
          metric: action_waste_inverse
      gates:
        - condition: "net_worth >= initial_money"
          on_fail_score: 0.01
```

### Validation

Run all 3 tasks under heuristic agent and 72B zero-shot; confirm dimension-level scores look sensible and composite lands in [0.01, 0.99]. Update `test_phase*.py` for new return shape (~30 min).

The dimension breakdown becomes a free **spider/radar chart** for the README — visually richer than a single number.

---

## 8. Workstream 4 — Unsloth + TRL GRPO Training (Hours 10-24) — closes B4 & B7 — **Owner: Person B** (env-side support: Person A)

### Why Unsloth + TRL specifically

`Planning/Hackathon Themes & Judging Criteria.md` explicitly mandates: *"A working training script using Unsloth or Hugging Face TRL, ideally as a Colab notebook so judges can re-run it."* The existing hand-built `train.py` works but isn't Unsloth or TRL — judges likely downgrade. We migrate.

### Path

- **Path A (recommended):** `notebooks/train_grpo_unsloth.ipynb` — `unsloth.FastLanguageModel` for model loading + `trl.GRPOTrainer` for the training loop. ~80 lines of cells. Reuses the `FarmEnvClient` HTTP client from `train.py:124-153` to roll out episodes.
- **Path B (1h fallback):** keep `train.py`; swap model loading to `unsloth.FastLanguageModel.from_pretrained` (10-line swap). Doesn't satisfy TRL, but at least claims Unsloth.

Plan: A primary, B as 1h fallback if `GRPOTrainer` integration with HTTP rollouts breaks.

### Compute plan — use HF on-site credit, NOT free Colab

The hackathon brief says: *"Post-training can be done onsite on 25th & 26th when you receive compute credits for HuggingFace."* Today is 2026-04-25 — credits are live. Use them as the primary training compute, not insurance.

| Item | Cost | Use |
|---|---|---|
| HF Space dedicated **L4** (24GB VRAM) | ~$0.80/h | Primary training: 3-4h of GRPO + smoke runs ≈ $3-4 |
| HF Space dedicated **A10G** | ~$1.30/h | If we need bigger model or curriculum runs ≈ $4-6 |
| HF Inference Endpoint (Llama-3.1-70B) | ~$2-3/h | Optional: stronger baseline bar in comparison plot ≈ $1-2 |
| **Total budget used** | **$8-12** | **Reserve $78+ untouched** |

Free Colab T4 with click-to-stay-alive is now the **fallback** if HF credit runs into account-gating issues.

### Hyperparameters (locked)

- **Model:** `unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit` (Unsloth 4-bit, fits T4 trivially, fits L4 with massive headroom)
- **LoRA:** r=16, alpha=32, dropout=0.05, target = `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`
- **GRPO:** num_iterations=50, group_size=4, max_steps=30, lr=5e-5, kl_coef=0.05, max_new_tokens=80
- **Curriculum:** Task 1 only. Evaluate **zero-shot** on Tasks 2 and 3 — generalization is itself a result.
- **Seed:** torch + env both seeded for reproducibility.

**Compute estimate:** 50 iters × K=4 × 30 steps × ~20k tokens/episode ≈ 1.0-1.5h on L4. Comfortable in H 19-21 budget.

### Smoke test (H 10-12)

10 iters × K=2 on Task 1. Pass criteria: episode-reward median moves up by ≥0.5 OR gradient norm non-zero/non-NaN. If smoke fails:

- Bisect 1: drop LR 5e-5 → 2e-5
- Bisect 2: disable KL term
- Bisect 3: bump K=2 → K=4 (advantages all-zero means env too deterministic at small K)

If still flat after all 3 bisects → trigger CoT prompt-engineering backup plan (don't announce until forced).

### Five pieces of evidence (H 22-24)

Per judging criteria: *"If you have multiple runs (baseline vs. trained, ablations, etc.), put them on the same axes so the comparison is obvious."*

1. **Reward curve** — episode reward mean ± std across 50 iters → `assets/reward_curve.png`
2. **Loss curve** — policy loss + KL term, dual y-axis → `assets/loss_curve.png`
3. **Baseline-vs-trained bar chart** — 5 bars on **same axes**: random / heuristic / 72B zero-shot / 0.5B untrained / 0.5B trained, mean ± std over 10 episodes per bar → `assets/baseline_comparison.png`. Headline: *"0.5B trained ≥ 72B zero-shot."*
4. **Action distribution shift** — pie or stacked-bar of action frequencies before vs after training → `assets/action_shift.png`. Expect: less `wait`, more `irrigate`/`harvest`/`sell`.
5. **Trajectory diff** — same seed, day-by-day actions of untrained vs trained, side-by-side table embedded in README markdown.

All plots: 1200px wide, axes labeled with units, title, legend. matplotlib defaults — do NOT spend cycles on theme polish. Save as PNG, **commit to repo** (judging criteria explicitly forbids leaving plots only in Colab cells or deleted W&B runs).

### Push checkpoint to HF Hub

Final adapter only. One `huggingface_hub.upload_folder` call. Repo: `<team-username>/farm-grpo-qwen0.5b-task1`. Add a one-page model card with usage snippet pointing back to the FarmSim env.

---

## 9. Workstream 5 — Storytelling & Submission (Hours 21-30) — **Owner: Person B** (review + engineering sign-off: Person A)

### README rewrite (H 25-27)

Replace `/mnt/d/FarmSimulation/README.md` (193-byte stub). Source material: `README_GITHUB.md` + new training results. Story arc per judging criteria's explicit prescription (Problem / Environment / Results / Why does it matter):

1. **Hero block (1 paragraph + theme tags)**
   *"FarmSimulation: a long-horizon economic and resource-management simulation where an LLM agent must synthesize noisy textual data — weather reports, neighbor emails, market gossip — to survive dynamic market and climate volatility. Theme #2 (Long-Horizon Planning) + Theme #3.1 (Professional Tasks). Headline result: a 0.5B GRPO-trained model outperforms 72B zero-shot on Task 1, [0.01 → X grade, Yx improvement]."*
2. **The problem.** Most RL benchmarks are clean grids. Real LLM agents face messy text and incomplete information. Cite the v2 reframe verbatim.
3. **The environment.** Annotated Gradio screenshot, one example noisy observation block (showing daily weather report + neighbor email + market gossip), one annotated action JSON. State the 3-task curriculum.
4. **The reward.** RubricComposer diagram with profit / stewardship / efficiency dimensions + binary competence gate. Mention the three exploits we explicitly closed (spam-water → labor friction; HFT → Almgren-Chriss slippage; patience-farming → consecutive-wait cap).
5. **The training.** *"GRPO via Unsloth + TRL on Qwen2.5-0.5B-Instruct with LoRA r=16. Group size 4, 50 iterations, full episodes against the live env."* One paragraph + algorithm box.
6. **Results.** Embed the 5 plots from Workstream 4. Headline number prominent (e.g., "0.01 → 0.34 grade, 34× improvement"). Include the **Sunk-Cost-Fallacy trajectory diff** as the marquee qualitative example.
7. **Try it yourself.** `pip install`, link to live HF Space, link to Colab notebook badge, link to HF Hub checkpoint.
8. **Limitations & future work** (1 short paragraph) — multi-season persistence, crop rotation soil-fertility, dual-LLM judge, the **Reward 2.0** roadmap (see §11).

### Required cross-links (in README and HF Space description)

- HF Space URL (env)
- HF Hub model URL (checkpoint)
- Colab notebook badge (Unsloth+TRL trainer)
- YouTube video URL (unlisted, <2:00) **OR** HF blog post URL
- Optional: HF blog post URL (additive)

### Required pre-submit cleanup

Move to `Planning/archive/` (don't let judges open self-assessment docs):
- `STRATEGIC_SUMMARY.md`, `META_HACKATHON_ANALYSIS.md`, `ROUND1_COMPLIANCE_CHECKLIST.md`, `FIXES_APPLIED.md`, `LLM_JUDGE_STRATEGY.md`, `MATHEMATICAL_FOUNDATION.md`, `farming_realism_analysis.md`

Delete:
- `main.py` (empty)
- `honey_app.py` (leftover from another project — TradeExecGym)
- `dual_llm_inference.py` — keep but ensure not referenced from README (we'll surface it as future work)

### Video (H 27-29)

**Rule from judging criteria:** *"Please do not include big video files in your Env submission on HF Hub… Please use url as reference link to additional materials."* So: upload to YouTube unlisted, **link only**. Do NOT commit the .mp4 file.

Script (<2:00 total):

| Time | Content |
|---|---|
| 0:00-0:15 | Hook: "What if your RL agent had to read **this** to farm?" → show noisy obs block on screen |
| 0:15-0:45 | Gradio dashboard walkthrough; point at narrative text + reward dimensions |
| 0:45-1:15 | Reward curve animation + baseline-comparison bar (same-axes plot) |
| 1:15-1:45 | Side-by-side trajectory: untrained spams water + ignores neighbor warning; trained pre-irrigates and pivots when locust email arrives |
| 1:45-2:00 | Links + thanks; theme tags on screen |

Recording rules: OBS or Loom. Pre-write VO script. Record VO separately, layer over screen capture. Single edit pass max.

**HF blog fallback:** if video eats >2h, ship a 600-word HF blog post with the 5 plots embedded. Rubric explicitly accepts either.

---

## 10. Critical Files Touched

| File | Change | Workstream |
|---|---|---|
| `server/farming_environment.py` | Add `_render_narrative_observation()`; fix labor decrement; cap patience reward; superlinear slippage | 1, 2 |
| `server/tasks.py` | Replace 3 monolithic graders with `RubricComposer` + `Dimension` + `Gate`; add binary competence gate | 2, 3 |
| `models.py` | Add `narrative_text: str` to `FarmObservation` | 1 |
| `inference.py` | Switch prompt to consume `narrative_text` with structured fallback | 1 |
| `openenv.yaml` | Extend each task with composite `grader:` block | 3 |
| `notebooks/train_grpo_unsloth.ipynb` | NEW: Unsloth + TRL `GRPOTrainer` | 4 |
| `train.py` | Path B fallback only: swap model loading to Unsloth | 4 |
| `README.md` | Full rewrite from 193-byte stub; theme tags; 5 embedded plots; cross-links | 5 |
| `assets/reward_curve.png`, `assets/loss_curve.png`, `assets/baseline_comparison.png`, `assets/action_shift.png` | NEW: 4 PNG plots committed | 4 |

**Files to delete or archive:** see Workstream 5 cleanup list.

---

## 11. Reward 2.0 — Future Direction (Round 3 / Post-Submission)

`Planning/reward engg.md` outlines the state-of-the-art reward architecture for production RL systems. **DO NOT implement in this 30h** — but include as a concrete future-work section in the README so judges see we know where this is going. Maps each principle to FarmSim:

| Principle from `reward engg.md` | FarmSim mapping | Round-3 work |
|---|---|---|
| **Hybrid Multi-Objective** (Execution + Similarity + Model-Based) | Profit/grade (anchor) + heuristic-trajectory similarity (guide) + LLM-judge on action rationale (refiner) | Add CodeBLEU-analog action-sequence similarity to expert trajectories |
| **Process Reward Models (PRMs)** | Per-step partial-trajectory scoring | Train a small PRM on (state, action, future-reward) tuples |
| **Potential-Based Reward Shaping (PBRS)** | Potential function: distance to optimal market timing | Add `Φ(state) = projected_yield_at_optimal_sell_day - current_yield` |
| **Curriculum Schedules** | Tasks 1→2→3 by difficulty (already partial) | Add Task 4 (Expert) with multi-season persistence + crop rotation |
| **Exploration-Guided Reward Shaping (EXPLORS)** | Intrinsic bonus for novel observations | Bonus for observing rare weather/market events |
| **Bi-Level Optimization (BiPaRS)** | Adaptive learning of dimension weights | Outer loop optimizes (profit, stewardship, efficiency) weights against held-out task suite |
| **Reward Uncertainty (RUNE)** | Ensemble LLM-judges over action rationale | Multi-judge variance as exploration bonus |
| **Gated Thinking Rewards** | Intermediate reasoning rewarded only if final profit > 0 | Already partly implemented via binary competence gate; extend to step-level CoT |
| **Inverse Reward Design (IRD)** | Treat handcrafted reward as proxy, not truth | Infer true intent from heuristic-agent trajectories; apply risk-averse policy correction |

In the README's "Future Work" paragraph, name-check at least 3 of these (PRMs, PBRS, BiPaRS) — signals to judges we have a research roadmap, not a one-off submission.

---

## 12. HF $90 Team Credit Strategy

Reframed: this is the **on-site post-training compute budget**, given for Apr 25-26 specifically.

| Allocation | Cost | Purpose | Workstream |
|---|---|---|---|
| Dedicated **L4 Space** for primary GRPO training | ~$3-4 | Stable training environment, no Colab disconnects | WS4 |
| Dedicated L4 Space for smoke runs + ablations | ~$1-2 | Faster iteration than Colab | WS4 |
| HF Inference Endpoint (Llama-3.1-70B) for stronger baseline bar | ~$2-3 | Optional polish, only if H 24+ has slack | WS4 |
| **Reserve untouched** | **~$78+** | Buffer for re-runs, new ablations, longer training if smoke promising | — |

**Decision points:**
- H 0-1: verify all 3 team accounts can see credits. Pool to one account if any are gated.
- H 12: if smoke run survived 10 iters cleanly → continue on L4. If account-gating issues → fall back to free Colab.
- H 22+: if reward curve looks great and we have 6+ hours of buffer → spend $2-3 on Llama-70B baseline endpoint to widen the comparison plot.

---

## 13. Risk Register & Backup Plans

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GRPO no signal by iter 10 | Medium | High | 3 bisects (LR, KL, K). If still flat: switch headline to CoT prompt engineering, ship 0.01→~0.15 prompt-only improvement. Do NOT announce until forced. |
| HF account credit-gating issues | Low-Med | High | Verify all 3 accounts BEFORE H 12. Pool credit. Fallback to free Colab T4 with click-to-stay-alive. |
| Colab disconnect mid-run (fallback path) | Medium-High | High | Checkpoint every 5 iters in the notebook. |
| Pivot bug breaks graders or tests | Medium | High | Keep all structured fields. Run `verify_all.py` immediately after Workstream 1. If broken: revert to journal-only narrative (drop forecast lies + neighbor emails); re-test. |
| HF Space deploy breaks after env changes | Medium | Medium | Test container locally with `docker build` BEFORE pushing. Last-known-good image stays on HF as fallback. H 18-19 reserved. |
| MCP reserved-name collision | Very Low | High | Add a unit test that asserts no action name in `{reset, step, state, close}`. Run in CI / pre-submit. |
| openenv.yaml schema rejection from new composite grader | Low-Med | High | Validate locally with `openenv validate` before pushing. One-line rollback to monolithic grader on hand. |
| Robustness skill-gradient inverts on pivoted env | Low | Medium | Re-run `robustness_validation.py` at H 9 and H 24. If inverted: dial down forecast lie rate 25%→15% and slippage exponent 1.5→1.3. |
| Video eats >2h | Medium | Low-Med | Pre-write script. Hard cap 1.5h. If overrun → ship HF blog post instead. |
| Sleep deficit causes errors at H 24+ | High | Medium | 6h sleep block 12-18 is non-negotiable. Cut video polish before sleep. |

---

## 14. Explicit DEFER / CUT List (do NOT do in this 30h)

Per `Planning/Self-Assessment Q&A.md` (v1) and the inventory, these are tempting but Round-3 territory:

- Multi-season soil persistence
- Crop rotation soil-fertility mechanics
- Equipment degradation
- Phenological staging beyond seedling/growing/mature
- Weather forecast confidence decay (use fixed lie rates instead)
- Risk management / crop insurance options
- Task 4 (Expert tier) — keep curriculum at 3 tasks
- Dual-LLM judge (`dual_llm_inference.py`) — keep disabled, surface as future work
- Free-text action parsing — actions stay structured JSON
- Full curriculum 1→2→3 GRPO training — Task 1 only, evaluate zero-shot on 2/3
- Bumping to Qwen 1.5B / 1.7B — VRAM trivially fits but weakens "small model wins" story
- Refactoring 1411-line `farming_environment.py` into modules — works as-is
- Adding new unit tests for the pivot — manual smoke acceptable for finals
- `honey_app.py` investigation — one-commit deletion only
- `main.py` content — leave empty or delete
- Theme/CSS polish on Gradio dashboard — not judged on aesthetics
- PRMs / PBRS / BiPaRS / EXPLORS implementation — README mentions them as Reward 2.0 future work; do NOT implement

---

## 15. Verification & Submission Checklist

Run at H 29 — every box checked or do not submit:

**Engineering table-stakes (per judging criteria §"Engineer it cleanly"):**
- [ ] Uses OpenEnv `Environment` / `MCPEnvironment` base classes
- [ ] Client/server separation respected (no client-side imports of server internals)
- [ ] Standard Gym-style API (reset, step, state)
- [ ] Valid `openenv.yaml` (`openenv validate` passes)
- [ ] No reserved tool names in MCP action space (`reset/step/state/close`)

**Mandatory deliverables (per judging criteria §"Minimum Submission Requirements"):**
- [ ] `bash validate-submission.sh <hf-space-url>` passes (Space ping + Docker build + openenv validate)
- [ ] HF Space loads, `/reset` and `/step` endpoints respond
- [ ] HF Hub checkpoint URL resolves and downloads with `from_pretrained()`
- [ ] Unsloth + TRL Colab notebook runs end-to-end on a fresh runtime (test once at H 28-29)
- [ ] Loss + reward plots saved as PNG, **committed to repo**, embedded in README
- [ ] Mini-blog (HF) **OR** <2 min YouTube video — linked from README, NOT committed as file
- [ ] README links to: HF Space, HF Hub model, Colab notebook, video/blog

**Quality gates:**
- [ ] `python verify_all.py` passes
- [ ] `python robustness_validation.py` shows skill gradient: random < heuristic < 72B zero-shot < 0.5B trained
- [ ] Baseline + trained scores reproducible from documented seeds
- [ ] Plot axes labeled with units, baseline-vs-trained bar uses **same axes**
- [ ] README readable in 3-5 minutes; story arc (Problem / Environment / Results / Why)
- [ ] Theme alignment statement (Theme #2 + #3.1) in README hero
- [ ] Reward 2.0 future-work section name-checks PRMs / PBRS / BiPaRS

**Hygiene:**
- [ ] No "we tried but…" lines or mention of cut features in README
- [ ] `Planning/archive/` houses self-assessment docs; root has only forward-facing files
- [ ] `git log --oneline` clean (squash messy WIP commits if any)
- [ ] No HF Space env-var secrets committed (re-check `.env*` files)
- [ ] No big video files in repo
- [ ] Submission link points to public GitHub repo + lists Space URL, video URL, HF Hub URL

---

## 16. Cross-References to Planning Folder

This plan is anchored on, in priority order:

1. `Planning/Self-Assessment Q&A v2.md` — **most authoritative**; the brutal reality check + Immediate Action Items list. Every workstream traces back to a v2 directive.
2. `Planning/Hackathon Themes & Judging Criteria.md` — the 40/30/20/10 rubric, Unsloth/TRL mandate, theme alignment, plot rules, story-arc prescription.
3. `Planning/RL Hackathon Project Analysis Guide.md` — competitive intel: GRPO+TRL+LoRA winning recipe, composable rubrics over monolithic.
4. `Planning/reward engg.md` — Reward 2.0 roadmap (PRMs, PBRS, BiPaRS, EXPLORS, Gated Thinking, IRD); DO NOT implement now, but cite in README future work.
5. `Planning/Self-Assessment Q&A.md` (v1) — original gap checklist; superseded by v2 for pivots, but useful for the cut list.

If any Planning doc conflicts with this plan during execution, defer to the Planning doc and flag it.

---

## 17. End-to-End Verification (How a judge will see it)

After submission, a judge runs:

1. **Open GitHub README** → reads the story arc in 3-5 min, sees theme tags, sees 5 plots inline, clicks the video, understands the project.
2. **Click the HF Space link** → loads Gradio dashboard → triggers `/reset` and `/step` → sees noisy narrative observations rendered.
3. **Open the Colab notebook** → runs all cells → sees `unsloth.FastLanguageModel` load Qwen + `trl.GRPOTrainer` train + reward curve regenerate.
4. **Run `bash validate-submission.sh <space-url>`** → all checks pass.
5. **Optional:** `huggingface_hub` downloads the checkpoint, runs `inference.py` against the live Space, sees the 0.5B trained model outscore the 72B zero-shot baseline on Task 1.

Each path must work on a single attempt, on a fresh machine, with documented commands. That's what H 29-30 buffer is for.

---

## 18. Concrete Unsloth + HuggingFace Utilization Plan

Judging criteria mandate verbatim: *"A working training script using **Unsloth or Hugging Face TRL**, ideally as a Colab notebook so judges can re-run it."* This section makes that requirement concrete.

### 18.1 Unsloth — what to use and how

Unsloth is the canonical Colab/T4-friendly fine-tuning stack and **explicitly named in the rubric**. Use it for three things:

1. **Model loading + 4-bit quant + LoRA wiring** — replaces ~30 lines of HF transformers + bitsandbytes + peft boilerplate with 5 lines:
   ```python
   from unsloth import FastLanguageModel
   model, tokenizer = FastLanguageModel.from_pretrained(
       model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
       max_seq_length=2048,
       load_in_4bit=True,
   )
   model = FastLanguageModel.get_peft_model(
       model, r=16, lora_alpha=32, lora_dropout=0.05,
       target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
   )
   ```
2. **2-5x training throughput** on T4/L4 vs vanilla HF (Unsloth's custom Triton kernels). Critical for finishing 50 GRPO iters in <2h on L4.
3. **Recognizable to judges.** The first cell of the Colab notebook should be `!pip install unsloth` and the visible imports — judges scan for this.

### 18.2 TRL — GRPOTrainer integration

Unsloth ships with a TRL-compatible patch path. Use **`trl.GRPOTrainer`** (not the hand-built loop in `train.py`) so the rubric requirement is satisfied:

```python
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    output_dir="grpo_farm_qwen_0_5b",
    num_train_epochs=1,
    per_device_train_batch_size=4,        # = group size K
    num_generations=4,                     # GRPO group size
    max_prompt_length=1024,
    max_completion_length=80,
    learning_rate=5e-5,
    beta=0.05,                             # KL coefficient
    logging_steps=1,
    save_steps=10,
    report_to="wandb",
)

def reward_fn(completions, **kwargs):
    # Roll completion into FarmEnvClient.step(), return reward per completion
    return [run_episode_and_get_reward(c) for c in completions]

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=prompt_dataset,          # initial farm-state prompts
    tokenizer=tokenizer,
)
trainer.train()
```

**Adapter pattern for env-as-reward:** TRL's `GRPOTrainer` expects `reward_funcs` to take completions and return scalar rewards. Wrap the existing `FarmEnvClient` (`train.py:124-153`) in a `run_episode_and_get_reward(completion: str) -> float` adapter that rolls the LLM's full episode against the live HF Space and returns the final episode reward. ~40 lines.

**Version pinning** (avoid breakage in Colab): `trl==0.11.4`, `unsloth` latest, `transformers>=4.45`, `peft>=0.13`. Test in a throwaway notebook at H 1.

### 18.3 HuggingFace platform — five concrete uses

| Use | What | Cost | Workstream |
|---|---|---|---|
| **HF Spaces (env hosting)** | Already live; redeploy after pivot. Use Docker SDK. | Free for shared CPU; **upgrade to dedicated L4 ($0.80/h) for the post-training run only**. | WS5 redeploy, WS4 training |
| **HF Hub (checkpoint)** | Push trained LoRA adapter via `huggingface_hub.upload_folder()`. Repo: `<team>/farm-grpo-qwen0.5b-task1`. | Free | WS4 |
| **HF Hub (dataset, optional)** | Push baseline + trained trajectories as a `datasets` repo for reproducibility. | Free | WS4 polish |
| **HF Inference Endpoints (optional baseline)** | Spin up Llama-3.1-70B for ~30 min ($1-2) → adds a stronger baseline bar to the comparison plot. | $1-2 | WS4 polish |
| **HF Blog post (storytelling fallback)** | If the YouTube video runs over budget, ship a 600-word HF blog post with embedded plots. Rubric accepts either. | Free | WS5 |

### 18.4 Colab notebook structure (judges re-run this)

`notebooks/train_grpo_unsloth.ipynb` cells in order:

1. `!pip install unsloth trl==0.11.4 huggingface_hub openenv-core` (one-shot install)
2. HF login + W&B login
3. `FastLanguageModel.from_pretrained(...)` — model load
4. `FastLanguageModel.get_peft_model(...)` — LoRA wiring
5. `FarmEnvClient` HTTP client to the live HF Space
6. `reward_fn` adapter (rollout + scalar reward)
7. `GRPOConfig` + `GRPOTrainer` instantiation
8. `trainer.train()` — main loop
9. Plot generation (matplotlib) — saves to `assets/`
10. `huggingface_hub.upload_folder()` — push adapter to Hub
11. Final cell: print reproducibility metadata (seeds, package versions, Hub URL)

Notebook must run end-to-end on a fresh Colab runtime. Test once at H 28-29.

### 18.5 What this gets us per rubric axis

- **Innovation (40%)** — Unsloth/TRL aren't innovation; they're table stakes. The pivot is the innovation.
- **Storytelling (30%)** — Colab badge + HF Hub link + HF blog post = three legible "try it yourself" surfaces.
- **Improvement (20%)** — TRL's `GRPOTrainer` ships with built-in reward + loss logging; the curves are auto-generated.
- **Pipeline (10%)** — Using the **named** stack (Unsloth + TRL) instead of hand-built GRPO directly addresses this axis.

---

## 19. Two-Person Claude Code Collaboration Workflow

Goal: two team members on separate PCs, each running Claude Code, ship the 30-hour plan together without merge conflicts, broken `main`, or duplicated work.

> **The full hour-by-hour parallel timeline is in §4.** This section is the operational playbook (branches, merge rules, ownership matrix, comms protocol). Read §4 for "what to do at hour H"; read this section for "how to do it without breaking each other."

### Quick "who-runs-which-command" reference

| Command / action | Person A | Person B |
|---|---|---|
| `git push` to `server/`, `models.py`, `inference.py`, `openenv.yaml` | ✅ | ❌ |
| `git push` to `notebooks/`, `assets/`, `README.md` | ❌ | ✅ |
| `git push` to `tasks.py`, `verify_*.py`, `test_phase*.py` | ✅ | ❌ |
| `git push` to `train.py` | ❌ | ✅ (or delete) |
| Run `validate-submission.sh` | ✅ (primary) | ✅ (sanity check at H 29) |
| Run `verify_all.py` / `verify_economics.py` / `verify_actions.py` | ✅ | ❌ |
| Run `robustness_validation.py` | ✅ | ❌ |
| Deploy/redeploy HF Space (env container) | ✅ | ❌ |
| Spin up dedicated L4 Space for training | ❌ | ✅ |
| `huggingface_hub.upload_folder()` (push adapter to Hub) | ❌ | ✅ |
| Trigger Colab training run | ❌ | ✅ |
| Record video / write blog post | ❌ | ✅ |
| Submit final URL to hackathon portal | ❌ | ✅ (with A's engineering sign-off) |
| Update `STATUS.md` | ✅ + ✅ (both, hourly) | ✅ + ✅ |

### 19.1 Roles & ownership (lock at H 0)

Pick once, do not swap:

| Role | Owns | Workstreams |
|---|---|---|
| **Person A — Env Lead** | `server/`, `models.py`, `inference.py`, `openenv.yaml`, `verify_*.py`, HF Space deploy | WS1, WS2, WS3, WS4 (env-side) |
| **Person B — Training & Story Lead** | `notebooks/`, `train.py`, `assets/*.png`, `README.md`, video script, HF Hub upload | WS4 (training-side), WS5 |

This minimizes file overlap to near-zero. The only shared files both touch: `IMPLEMENTATION_PLAN.md` (read-only after lock), `README.md` (B writes; A reviews), and the `Planning/archive/` move (one person executes, both verify).

### 19.2 Branch strategy

- **`main` is sacred.** Never push directly. HF Space auto-deploys from `main`; broken `main` = broken Space.
- One **feature branch per workstream**, prefixed with owner initial:
  - `a/ws1-narrative-obs`, `a/ws2-reward-hardening`, `a/ws3-rubric-composer`
  - `b/ws4-unsloth-grpo`, `b/ws5-readme-rewrite`
- Each branch is short-lived (≤4h). Merge to `main` via PR after each workstream, not at the end.
- After a merge, the OTHER person runs `git pull --rebase origin main` immediately on their active branches.

### 19.3 PR cadence (3 PRs minimum per person, target ~2-3h each)

- PR opens with a one-line title (`ws1: add narrative_text to FarmObservation`).
- Body: bullet list of changes + "I tested with: …" + "@reviewer please check: …".
- Reviewer has a 30-minute SLA. Approve or comment within that window. No silent stalls.
- **No squash-merge of WIP commits** — keep small commits visible so blame and bisect work.

### 19.4 File ownership matrix (avoid simultaneous edits)

| File | Primary owner | Other person rule |
|---|---|---|
| `server/farming_environment.py` | A | Never edit; suggest changes via PR comment |
| `server/tasks.py` | A | Never edit; needs A's WS3 done before B uses |
| `models.py` | A | Add fields only via A; B can read freely |
| `inference.py` | A | Never edit |
| `openenv.yaml` | A | Never edit |
| `train.py` | B | A doesn't touch; B may delete entirely if Path A succeeds |
| `notebooks/*` | B | A doesn't touch |
| `assets/*.png` | B | A doesn't touch |
| `README.md` | B | A reviews only |
| `IMPLEMENTATION_PLAN.md` | both read | Read-only after H 0 lock |
| `Planning/*` | both read | Read-only |
| Test files (`test_phase*.py`, `verify_*.py`) | A | If B's work breaks tests, B opens a PR to A with the fix; A approves |

### 19.5 Sync protocol — what each person does at the top of every Claude Code session

```bash
git fetch origin
git checkout main && git pull --ff-only
git checkout <my-branch>
git rebase main          # resolve conflicts immediately if any
```

Then in Claude Code: paste a short status update of what the OTHER person merged since last session. This keeps Claude's context current without scanning git log.

### 19.6 Handoffs between workstreams (dependencies)

```
WS1 (A) ──┐
WS2 (A) ──┼──> WS3 (A) ──┐
          │              │
          └─> baseline ──┴──> WS4 smoke (B) ──> WS4 full (B) ──> WS5 README (B)
                                                           │
                                                  HF deploy (A)─┘
```

Concretely:
- **A finishes WS1+WS2** (H 7) → tags B in PR. B starts notebook scaffolding (cells 1-5) on a frozen API contract.
- **A finishes WS3** (H 9) → B runs first smoke against the merged `main`.
- **A redeploys HF Space** (H 19) while B kicks off the full training run. Parallel.
- **B finishes WS4 evidence** (H 24) → A runs the final robustness validation. Parallel: B starts README.

### 19.7 Communication channel

Pick ONE channel for the 30 hours (Slack / Discord / Telegram / WhatsApp — doesn't matter, just one). Post:

- **Status pings every 1 hour.** "H+3, on WS1 step 2/4, no blockers" — even if nothing changed. Silence is a smell.
- **Branch-switch notices.** "About to push `a/ws2`, please don't touch `farming_environment.py` for the next 30 min."
- **PR-ready ping.** "PR #5 open, needs review by H+5 or I'm blocked."
- **Blocker calls.** Anything that stops you for >15 min → escalate to voice call. Don't grind alone.

### 19.8 Conflict-resolution rules

- **No force-push to main, ever.** Force-push to your own feature branch is fine.
- **Conflict during rebase:** the person doing the rebase resolves it. If unsure, ping the file's primary owner before resolving.
- **`uv.lock` / `pyproject.toml` conflicts** are common when both add deps. Resolution: A owns `pyproject.toml`. B requests deps via PR comment; A adds them in a single bundled commit.
- **`assets/*.png` conflicts:** B owns. If A regenerates plots somehow, A overwrites only via B's approval.

### 19.9 Pre-merge gate (every PR must pass)

```bash
python verify_all.py                    # never broken
python -m pytest test_phase*.py -q      # all green
bash validate-submission.sh <space-url> # only required for PRs touching server/ or openenv.yaml
```

Any red → PR is not mergeable. Fix on the branch, push again. No `--no-verify`.

### 19.10 Shared status board (single source of truth)

Use a `STATUS.md` file at the repo root that BOTH people update via tiny commits:

```markdown
# Live Status (last updated H+5 by A)

## Person A
- Branch: a/ws3-rubric-composer
- In progress: refactoring grade_task1 to RubricComposer
- Blocked on: nothing
- Next: openenv.yaml extension

## Person B
- Branch: b/ws4-unsloth-notebook
- In progress: GRPOConfig + reward_fn adapter
- Blocked on: A's WS3 merge (need RubricComposer return shape)
- Next: smoke run on L4 Space
```

Update on every status ping. `STATUS.md` is git-tracked but excluded from the final submission via `.gitignore` add at H 28 (or just leave it — judges won't penalize).

### 19.11 Backup persons-of-failure

- If A's PC crashes → B has read access to the HF Space + repo, can deploy from local clone, but should NOT touch `server/` files mid-flight.
- If B's PC crashes → A has the HF account credentials and can push to Hub from their machine. README finalization can shift to A.
- Pre-share: HF token (read-only fine for B, write for A), Colab Pro account (if any), HF Spaces deploy key.

### 19.12 What both people promise at H 0

Lock these three things in writing in `STATUS.md`:

1. "I will not push to `main` directly. PR + merge only."
2. "I will rebase on `main` at the start of every session and after every PR I see merged."
3. "I will post a status ping every hour."

Any drift from these = the plan slips. Discipline carries this project.

---

## 20. RL Engineering Deep-Dive (from Unsloth GRPO docs)

This section translates Unsloth's official GRPO/RL guide into FarmSim-specific engineering decisions. Source: `unsloth.ai/docs/get-started/reinforcement-learning-rl-guide` and `unsloth.ai/docs/models/gpt-oss-reinforcement-learning`. Read this section before writing a single line of `train_grpo_unsloth.ipynb`.

### 20.1 Reward decomposition — list of reward functions, NOT one monolithic fn

**Critical insight from the Unsloth + GPT-OSS RL tutorials:** TRL's `GRPOTrainer` accepts `reward_funcs` as a **list**. Each function is called per generation; total reward = **sum** of all functions. The official GPT-OSS example uses `reward_funcs=[function_works, no_cheating, strategy_succeeds]`.

This is the right pattern for FarmSim. Decompose into 5 reward functions instead of one giant `reward_fn`:

```python
def task_completion_reward(completions, **kwargs) -> list[float]:
    """Final episode profit / target_profit, clamped [0, 2]."""
    return [run_episode(c)["grade"] * 2.0 for c in completions]

def format_reward(completions, **kwargs) -> list[float]:
    """+0.3 for valid JSON action on every step, else 0."""
    return [0.3 if all_steps_parse(c) else 0.0 for c in completions]

def no_invalid_action_reward(completions, **kwargs) -> list[float]:
    """-0.5 per invalid action attempt (caught by env)."""
    return [-0.5 * count_invalid(c) for c in completions]

def anti_exploit_reward(completions, **kwargs) -> list[float]:
    """-1.0 if action distribution is >60% irrigate or >60% wait (cheating signature)."""
    return [-1.0 if exploit_pattern(c) else 0.0 for c in completions]

def stewardship_reward(completions, **kwargs) -> list[float]:
    """+0.5 × healthy_day_fraction (rewards keeping plots alive)."""
    return [0.5 * healthy_fraction(c) for c in completions]

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        task_completion_reward,    # primary (the anchor)
        format_reward,              # cheap shaping signal
        no_invalid_action_reward,   # discourage invalid actions
        anti_exploit_reward,        # close v2 spam-water/HFT exploits
        stewardship_reward,         # multi-objective
    ],
    args=training_args,
    train_dataset=prompt_dataset,
)
```

**Why this matters:**
- Unsloth logs each reward function separately in W&B → judges see a per-component breakdown automatically. Free storytelling.
- Maps 1:1 to the RubricComposer dimensions in §7 (profit / format / efficiency / anti-exploit / stewardship).
- v2's "spam-water" and "HFT" exploits get their own dedicated `anti_exploit_reward` fn that judges can name-check.

**Reward-function signature (locked):**
```python
def reward_fn(prompts, completions, **kwargs) -> list[float]: ...
```
- `prompts`: list of message dicts (the initial farm-state prompt per group member)
- `completions`: list of `[{"role":"assistant","content":"..."}]` — one per group member
- `**kwargs`: any extra dataset columns (use this to pass `task_id`, `seed`, `noise_seed`)
- Return: `list[float]`, one scalar per completion. Unsloth handles group-relative normalization internally.

### 20.2 GRPO hyperparameter map (FarmSim-locked values)

Pulled from Unsloth's `tutorial-train-your-own-reasoning-model-with-grpo` and `advanced-rl-documentation`, adapted for our env:

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    # === Generation ===
    temperature=1.0,                    # NOT 0.7 — GRPO needs exploration
    max_prompt_length=1024,             # narrative_text + system + history fits
    max_completion_length=80,           # JSON action only
    num_generations=4,                  # group size K; drop to 2 if T4 OOMs

    # === Optimizer ===
    learning_rate=5e-6,                 # Unsloth's tutorial default; 5e-5 too high for 0.5B
    optim="adamw_8bit",                 # 8-bit Adam saves ~25% VRAM on T4
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,                   # strong regularization for small model
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=0.1,                  # tight clipping — GRPO stability requires this

    # === Batching ===
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,      # smoother reward signal vs default 1
    max_steps=50,                       # ≈ 50 GRPO updates; one episode per group per step

    # === GRPO algorithm ===
    beta=0.0,                           # 0.0 = no reference model loaded → saves VRAM + 30% speed
                                        # bump to 0.04 if policy starts drifting
    epsilon=0.2,                        # standard PPO clip
    epsilon_high=0.28,                  # DAPO upper bound (Unsloth recommends)
    loss_type="dapo",                   # DAPO loss > vanilla grpo on small models
    mask_truncated_completions=True,    # don't penalize when env episode hits step cap
    importance_sampling_level="token",  # token-level (default; sequence is for reasoning models)

    # === Logging & checkpointing ===
    logging_steps=1,                    # log every step → dense reward curve
    save_steps=10,                      # adapter checkpoint every 10 steps
    output_dir="grpo_farm_qwen_0_5b",
    report_to="wandb",                  # → reward curve auto-uploaded
)
```

**Key callouts from Unsloth docs:**
- `beta=0.0` removes the reference model entirely. **Saves ~30% VRAM and 30% speed.** Only enable (set to 0.04) if smoke run shows policy collapse.
- `loss_type="dapo"` is the newer DAPO-paper loss — Unsloth's docs cite it as more stable for small models. Defaults to `"bnpo"` if not specified.
- `epsilon_high=0.28` (DAPO recommendation) widens the upper clip and helps when reward signal is sparse.
- `max_grad_norm=0.1` is **much tighter** than typical SFT (1.0). GRPO gradients are spiky; this prevents collapse.
- `gradient_accumulation_steps=4` simulates larger batch without VRAM cost — Unsloth's tutorials repeatedly recommend bumping from 1 to 4 for "smoother training."

### 20.3 Unsloth model loading — vLLM fast_inference + grad checkpointing

Generation cost dominates GRPO wall-clock (~80% of step time). Two Unsloth flags slash this:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
    max_seq_length=1104,                # = max_prompt_length + max_completion_length
    load_in_4bit=True,                  # 4-bit base, LoRA stays bf16
    fast_inference=True,                # vLLM backend → ~10× faster generation
    max_lora_rank=16,
    gpu_memory_utilization=0.7,         # leave 30% for activations + KV cache
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,                      # alpha = 2 × r is Unsloth's default rule
    lora_dropout=0.05,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
    ],
    use_gradient_checkpointing="unsloth",   # NOT True/False — the string "unsloth" enables their custom impl
    random_state=3407,
)
```

**Three flags worth understanding:**
1. `fast_inference=True` — Unsloth swaps in vLLM for the rollout phase. ~10× generation speedup. Only available when `load_in_4bit=True`. **Use this.**
2. `use_gradient_checkpointing="unsloth"` — string, not bool. Unsloth's custom checkpointing impl saves 30% more VRAM than HF's `True`. Worth ~30% longer max-seq-length.
3. `gpu_memory_utilization=0.7` — vLLM grabs 70% of VRAM upfront for KV cache. Drop to 0.5 if OOM during rollout.

### 20.4 Multi-turn rollout adapter (FarmSim-specific)

TRL's `GRPOTrainer` is single-turn by default: one prompt → K completions → K rewards → update. FarmSim is multi-turn (30-60 steps per episode). Two viable patterns; we use **Pattern B**:

**Pattern A — Episode-as-completion (rejected):** concat all 30 actions into one giant completion. Loses per-step diversity in the group; group_size=4 → only 4 trajectories explored per step. Bad for GRPO advantage estimation.

**Pattern B — Reward-fn-as-rollout (chosen):** the "completion" is just the first action; the reward function rolls out the rest of the episode internally and returns final episode reward. Each generation creates an independent rollout via the same model.

```python
def task_completion_reward(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        first_action = parse_action(completion[0]["content"])
        env = FarmEnvClient(SPACE_URL)
        obs = env.reset(task_id=kwargs.get("task_id", 1),
                        noise_seed=kwargs.get("noise_seed", 42))
        total_reward = 0.0
        # apply the GRPO-sampled first action
        obs, r, done, _ = env.step(first_action); total_reward += r
        # roll the rest with greedy model.generate (NOT GRPO-sampled)
        while not done:
            action_text = model_generate_greedy(obs.narrative_text)
            action = parse_action(action_text)
            obs, r, done, _ = env.step(action); total_reward += r
        rewards.append(total_reward)
    return rewards
```

**Why this works:** GRPO only needs *gradient* on the first action's tokens. The rest of the episode just gives us a final reward signal. As training progresses, the policy improves first-action quality, which compounds across the episode.

**Trade-off:** less efficient than full-trajectory PPO (we only update on the first step), but matches TRL's single-turn API and avoids rewriting `GRPOTrainer`. Acceptable for a 30h sprint.

**If we have spare time after H 24:** swap to a "step-as-completion" pattern where each row in `train_dataset` is `(state_at_step_t, history_so_far)` and we collect ~1000 such (state, action) pairs per epoch. This gives per-step gradient. Defer unless the basic Pattern B converges and we have buffer.

### 20.5 Loss type & two-sided clipping (don't ship the default)

Unsloth's advanced docs list four `loss_type` options:
- `"bnpo"` — basic, default. Works but plateaus on small models.
- `"grpo"` — original DeepSeek-R1 formulation.
- `"dr_grpo"` — Dr. GRPO, simplified.
- `"dapo"` — newer, two-sided clipping per the DAPO paper. **Recommended for FarmSim.**

Use:
```python
loss_type="dapo",
epsilon=0.2,
epsilon_high=0.28,
delta=1.5,                           # enables two-sided clipping (INTELLECT-2 paper)
mask_truncated_completions=True,
```

Two-sided clipping (`delta` set + `epsilon_high` distinct) is the single biggest stability win over default GRPO. Worth the 5 lines.

### 20.6 Common pitfalls (from Unsloth's RL guide)

1. **Reward variance collapse.** If all K=4 generations get identical reward, GRPO advantages are all zero → no gradient. Symptoms: flat loss, no movement after iter 5. **Fix:** raise `temperature` from 1.0 to 1.2; if still flat, K=4 → K=8; if still flat, the reward function isn't discriminating (your reward landscape is too flat — add per-step shaping signals).

2. **Reward hacking via format gaming.** If `format_reward` is too high relative to `task_completion_reward`, the model spams perfectly-formatted no-ops. **Fix:** keep format reward ≤ 30% of task reward magnitude. Rule of thumb: `max(format_reward) ≤ 0.3 × max(task_completion_reward)`.

3. **Truncated completions counted as failures.** If `mask_truncated_completions=False` (default), a generation that hits `max_completion_length` gets a 0 reward even if it would've been correct. **Fix:** always set `mask_truncated_completions=True`.

4. **vLLM cache OOM on group_size bump.** `num_generations=8` on T4 OOMs because vLLM allocates K parallel sequences. **Fix:** drop `gpu_memory_utilization` to 0.5, or stay at K=4.

5. **8-bit optimizer × LoRA mismatch.** Some `bitsandbytes` versions don't play with `optim="adamw_8bit"` + LoRA. **Fix:** if you see `RuntimeError: Tensor is not contiguous` during optimizer.step, fall back to `optim="adamw_torch"` (slightly more VRAM but stable).

6. **`processing_class` vs `tokenizer` arg name.** TRL ≥0.11 renamed `tokenizer=tokenizer` to `processing_class=tokenizer` in `GRPOTrainer.__init__`. **Use `processing_class=tokenizer`** to match Unsloth's current docs.

7. **`apply_chat_template` mismatch.** GRPO operates on chat-formatted prompts. Run prompts through `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` BEFORE feeding to the dataset. Skipping this means the model sees raw `[{"role":"user"...}]` strings and never learns.

### 20.7 Final notebook cell-list (locked)

`notebooks/train_grpo_unsloth.ipynb` — 13 cells:

1. `!pip install -U unsloth trl==0.11.4 huggingface_hub openenv-core wandb` (one-shot)
2. `huggingface_hub.login()` + `wandb.login()`
3. `FastLanguageModel.from_pretrained(...)` with `fast_inference=True, load_in_4bit=True`
4. `FastLanguageModel.get_peft_model(...)` with `use_gradient_checkpointing="unsloth"`
5. Define `FarmEnvClient` HTTP client (copy from `train.py:124-153`)
6. Define `parse_action()` helper (lift from `inference.py`)
7. Define the **5 reward functions** (§20.1)
8. Build `prompt_dataset`: 200 rows of (system_prompt, narrative_observation_at_day_0) — sample initial states with different `noise_seed`s
9. `GRPOConfig(...)` — locked hyperparameters from §20.2
10. `GRPOTrainer(model=model, processing_class=tokenizer, reward_funcs=[...], args=config, train_dataset=...)` + `trainer.train()`
11. Plot generation: load `trainer.state.log_history` → save 4 PNGs to `assets/`
12. `model.save_pretrained_merged(...)` then `huggingface_hub.upload_folder(...)` to push adapter
13. Final cell: print reproducibility metadata (`torch.__version__`, `unsloth.__version__`, `trl.__version__`, all seeds, Hub URL, W&B URL)

Test the notebook end-to-end on a fresh Colab runtime at H 28-29. Judges will rerun cell-by-cell — any ImportError or NameError disqualifies the submission from the 10% pipeline axis.

### 20.8 What this changes in earlier sections

The plan's earlier hyperparameters (§8) are now **superseded** by §20.2. Specifically:
- `lr` 5e-5 → **5e-6** (Unsloth's tutorial value; 5e-5 was too aggressive for 0.5B + GRPO).
- `kl_coef` 0.05 → **`beta=0.0`** (no reference model — saves 30% VRAM/speed; fall back to 0.04 only if drift).
- `max_steps` is now 50 GRPO updates, NOT 50 env iterations. Each step = one batch of K=4 episodes.
- Add `loss_type="dapo"`, `epsilon_high=0.28`, `delta=1.5`, `mask_truncated_completions=True`.
- Reward structure: 5 functions in a list, not one `reward_fn`. Each maps to a RubricComposer dimension.

The training compute estimate stays the same (~1.5h on L4 / ~3-4h on T4), since the bottleneck is rollout latency, not the optimizer.

---

## 21. Research Paper Techniques — Integrating `Planning/reward engg.md`

`Planning/reward engg.md` distills 9 named techniques from the reward-engineering literature. This section maps each technique to a concrete decision in our 30h plan: either **USE NOW** (already wired into a workstream, with the implementation hook) or **DEFER** (cite in README future work with a ready-to-paste sentence). Citing the techniques by name in the README is what turns our submission from "we built an RL env" into "we built an RL env grounded in current research" — directly serving the 30% Storytelling axis.

### 21.1 Technique-to-implementation map

| # | Technique (from `reward engg.md`) | Status | FarmSim implementation hook |
|---|---|---|---|
| 1 | **Hybrid Multi-Objective Reward** (Execution + Similarity + Preference) | ✅ USE NOW | §20.1 5-function reward list. Tag explicitly: `task_completion_reward` = **Execution anchor**; `format_reward` + heuristic-distance = **Similarity guide**; `anti_exploit_reward` = **Preference/Model-based refiner** |
| 2 | **Execution-Based Rewards** (verifiable anchor) | ✅ USE NOW | `task_completion_reward` returns the env's deterministic grade. The env IS the verifiable oracle — directly analogous to "compiler/test-runner" in the source paper |
| 3 | **Similarity-Based Rewards** (CodeBLEU/AST analog) | ✅ USE NOW (cheap version) | Add `heuristic_similarity_reward(c)`: edit-distance between agent action sequence and `HeuristicAgent` action sequence on same seed. Returns +0.2 × (1 - normalized_edit_distance). Cap contribution at 30% of task reward (§20.6 pitfall #2) |
| 4 | **Process Reward Models (PRMs)** | 🔜 DEFER | Round-3 work. README cite: *"Future work: train a PRM over (state, partial-action-sequence, future-reward) tuples for dense per-step credit assignment, following Lightman et al. (2024)."* |
| 5 | **Potential-Based Reward Shaping (PBRS)** | ⚙️ STRETCH | If H 22-24 has slack, add `Φ(state) = market_value_of_inventory_at_projected_optimal_sell_day - current_inventory_value`. Per-step shaping reward = `γ·Φ(s') - Φ(s)`. Theory guarantee: doesn't change optimal policy (Ng et al. 1999). 40-line addition to `farming_environment.py` |
| 6 | **Curriculum Schedules** (syntactic → compilation → tests) | ✅ USE NOW | Tasks 1 → 2 → 3 already exist (Easy 30d → Medium 45d climate-rotation → Hard 60d droughts). README must explicitly call this out as a curriculum schedule. Train on Task 1, evaluate zero-shot on 2/3 — shows curriculum generalization |
| 7 | **Exploration-Guided Reward Shaping (EXPLORS)** | 🔜 DEFER | Round-3. README cite: *"Future work: inject intrinsic exploration bonuses for novel observations (rare market or weather events) following EXPLORS-style shaping, to prevent stuck states under sparse rewards."* |
| 8 | **Bi-Level Optimization (BiPaRS)** | 🔜 DEFER | Round-3. README cite: *"Future work: outer-loop adaptive learning of (profit, stewardship, efficiency, anti-exploit) weights against a held-out task suite, replacing the hand-tuned weights in our RubricComposer."* |
| 9 | **Reward Uncertainty for Exploration (RUNE)** | 🔜 DEFER | Round-3. README cite: *"Future work: ensemble of 3 LLM-judges over action rationale; their disagreement variance acts as an exploration bonus."* |
| 10 | **Gated Thinking Rewards** | ✅ USE NOW | Already implemented as the **binary competence gate** in WS2/WS3: if `final_net_worth < initial_money`, all dimension scores collapse to 0.01 regardless of intermediate stewardship/efficiency. This is structurally identical to Gated Thinking (DeepSeek-R1 pattern: reasoning rewarded only if final answer correct). Name-check it in the README |
| 11 | **Inverse Reward Design (IRD)** | 🔜 DEFER | Round-3. README cite: *"Future work: treat the RubricComposer's hand-coded weights as proxy observations and infer the true objective from heuristic-agent demonstrations, building a risk-averse policy following Hadfield-Menell et al. (2017)."* |
| 12 | **Text2Reward** (LLM-generated reward fns) | 🔜 DEFER | Round-3. README cite: *"Future work: bootstrap new task graders via Text2Reward — an LLM writes the reward function from a natural-language task description."* |

### 21.2 Concrete code hooks for the USE-NOW techniques

**(1) (2) (3) (10)** — extend §20.1's reward function list:

```python
def task_completion_reward(prompts, completions, **kwargs):
    """Execution-based reward (verifiable anchor) — Hybrid Multi-Objective principle."""
    return [run_episode(c)["grade"] * 2.0 for c in completions]

def format_reward(prompts, completions, **kwargs):
    """Similarity-style format guide — keeps reward dense."""
    return [0.3 if all_steps_parse(c) else 0.0 for c in completions]

def heuristic_similarity_reward(prompts, completions, **kwargs):
    """Similarity-based reward (Guide) — analog of CodeBLEU/AST similarity from reward_engg.md.
    Compares agent action sequence to HeuristicAgent's policy on the same noise_seed."""
    rewards = []
    for prompt, completion, seed in zip(prompts, completions, kwargs["noise_seed"]):
        agent_actions = extract_action_sequence(completion)
        heuristic_actions = run_heuristic_episode(seed)
        sim = 1.0 - normalized_edit_distance(agent_actions, heuristic_actions)
        rewards.append(0.2 * sim)         # ≤30% of max task_completion_reward (pitfall #2)
    return rewards

def anti_exploit_reward(prompts, completions, **kwargs):
    """Preference/Model-based refiner — penalizes spam-water + HFT exploits from v2 §2."""
    return [-1.0 if exploit_pattern(c) else 0.0 for c in completions]

def stewardship_reward(prompts, completions, **kwargs):
    """Multi-objective component — long-term soil health."""
    return [0.5 * healthy_fraction(c) for c in completions]

# Gated Thinking — implemented as binary competence gate inside RubricComposer (§7)
# (already in tasks.py — no separate reward function needed)
```

**(6) Curriculum Schedule** — no code change; explicit README framing:

> *"FarmSim ships with a 3-task curriculum schedule (Easy 30-day stable climate → Medium 45-day climate rotation → Hard 60-day drought events), following the curriculum-schedule principle from `Planning/reward engg.md`. We train GRPO on Task 1 only and evaluate zero-shot on Tasks 2 and 3 to demonstrate curriculum generalization."*

**(5) PBRS — STRETCH if H 22-24 has slack:**

```python
# In server/farming_environment.py
def _potential(self, state) -> float:
    """Φ(s) = projected inventory value at optimal sell day, minus current inventory."""
    inv_value = sum(qty * self.market.spot_price(crop) for crop, qty in state.storage.items())
    optimal_sell_value = sum(
        qty * self.market.projected_max_price_in_window(crop, days_remaining=self.days_left)
        for crop, qty in state.storage.items()
    )
    return optimal_sell_value - inv_value

def _pbrs_shaping(self, prev_state, curr_state, gamma=0.99) -> float:
    """Per-step shaping reward — Ng et al. 1999 PBRS guarantees policy invariance."""
    return gamma * self._potential(curr_state) - self._potential(prev_state)
```

Add `_pbrs_shaping(...)` to the per-step reward in `step()`. README must cite Ng et al. 1999 if shipped.

### 21.3 Ready-to-paste README paragraphs (citing the techniques)

Drop these verbatim into the README `Reward design` section (Workstream 5, README rewrite step 4):

> **Reward design — research-grounded, multi-objective.**
>
> Our reward signal is a 5-function decomposition implementing the **Hybrid Multi-Objective Reward Architecture** from the reward-engineering literature:
>
> - **Execution-based anchor** (`task_completion_reward`): the verifiable, deterministic episode grade.
> - **Similarity guide** (`heuristic_similarity_reward`): edit-distance between agent action sequence and a hand-crafted heuristic baseline — a CodeBLEU-analog for action sequences.
> - **Format guide** (`format_reward`): cheap dense signal for valid JSON action structure.
> - **Preference/refiner** (`anti_exploit_reward`): explicitly penalizes the spam-water and high-frequency-trading patterns flagged in our self-assessment.
> - **Multi-objective stewardship** (`stewardship_reward`): rewards long-term soil and crop health.
>
> A **binary competence gate** (Gated Thinking pattern, à la DeepSeek-R1) collapses all dimension scores to the floor if the agent ends bankrupt — preventing reward hacking via stewardship-without-profit.
>
> The 3-task curriculum (Easy → Medium → Hard) is a deliberate **curriculum schedule**; we train on the easiest task and evaluate zero-shot generalization on the harder ones.
>
> **Future work** extends this with Process Reward Models for per-step credit assignment, Potential-Based Reward Shaping for dense temporal guidance, Bi-Level Optimization for adaptive weight learning, and Reward Uncertainty as an exploration signal.

### 21.4 Standing prompt for future Claude sessions

When any teammate (or future Claude session) is editing reward logic in this repo, they should consult `Planning/reward engg.md` BEFORE writing code. Add this to `STATUS.md` as a permanent note:

> **Reward-engineering rule:** Before adding, removing, or reweighting any reward component, re-read `Planning/reward engg.md` and confirm the change is consistent with one of the 9 named techniques. If the change doesn't map to a named technique, write a one-line justification in the PR description.

This keeps the reward function aligned with research practice and gives the README a continuously-cited source.

### 21.5 What to add to `STATUS.md` at H 0

```markdown
## Reward Engineering Reference
- Source: Planning/reward engg.md (9 named techniques + future-work roadmap)
- USE NOW: Hybrid Multi-Objective (Exec+Sim+Pref), Curriculum Schedule, Gated Thinking
- STRETCH (H 22-24 slack only): PBRS via Φ(state) = optimal_sell_value - current_inventory_value
- DEFER (cite in README future work): PRMs, EXPLORS, BiPaRS, RUNE, IRD, Text2Reward
- Rule: any reward edit must map to a named technique; cite in PR description
```

---

## 22. Bottom Line

GRPO-first is correct, but only after the v2 noisy-text pivot transforms the env from grid-world to LLM-native — otherwise the 40% Innovation slot evaporates and training curves are meaningless (B4). The 0.01 catastrophic baseline is our biggest asset: any modest training gain produces a dominant reward curve. README + video held to the last 3h block, with theme tags and the verbatim narrative reframe up top. HF $90 credit is the **primary** training compute, not insurance. Reward 2.0 (PRMs / PBRS / BiPaRS) named in future work but not implemented. Cut everything else.

*"A messy but ambitious environment with real training evidence beats a polished but boring one."* — Hackathon judging criteria, verbatim. That's the bar.
