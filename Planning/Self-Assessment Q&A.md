# FarmSimulation — Self-Assessment Q&A

Answering every question from the "What Makes a Submission Stand Out" section of the judging guide, honestly assessed against our current FarmSimulation project.

---

## Section 1: Pick an Ambitious, Original Problem

### Q: Does this environment exist to teach an LLM something it currently can't do well?

**Yes.** LLMs currently fail at multi-step resource management under uncertainty. Specifically:

1. **Temporal planning under compound constraints** — LLMs struggle when a single decision (e.g., planting rice) locks in consequences across 12 future days (water demand, nutrient drain, harvest timing) while external conditions (climate rotation, market cycles, pest outbreaks) change independently. FarmSimulation forces the agent to reason about *delayed, interleaved consequences* — not just the next token.

2. **Quantitative resource balancing** — The agent must simultaneously track money, water tank level, aquifer reserves, soil moisture per plot, NPK nutrients per plot, seed inventory, storage capacity, labor hours, and market prices. LLMs are notoriously poor at maintaining and updating multiple numerical state variables across long episodes. Our environment exposes this weakness directly.

3. **Market timing under information asymmetry** — Task 2 explicitly grades 40% on whether the agent sold at *above-average* prices. The agent sees a 7-day rolling average and a trend indicator, but must decide whether to hold or sell. This is a non-trivial sequential decision problem that zero-shot LLMs consistently fail at (our baseline Qwen-72B scores only 0.31 on Task 2).

**What LLMs can't do well today that this environment teaches:**
- Hold inventory when a greedy sell would give immediate reward, because waiting yields higher future revenue
- Pre-emptively pump water and irrigate *before* a drought arrives (proactive vs reactive reasoning)
- Recognize that planting corn ($12 investment, 18-day wait) is only worth it if the agent can sustain moisture for the full growth period given current water reserves

---

### Q: Is the domain underexplored in RL/LLM training?

**Yes, strongly.** Agricultural decision-making is massively underexplored in LLM/RL research:

- **No existing OpenEnv farming environments** — Grid-world clones, chess, and coding tasks dominate. Agricultural resource management is a $4.1 trillion global sector with zero dedicated LLM training environments.

- **Existing RL farming work is tabular/discrete** — Academic RL papers on agriculture (e.g., crop scheduling) use small discrete state spaces with tabular Q-learning. None use LLM agents with natural language observations and tool-calling actions.

- **Real-world relevance** — The FAO estimates that 30% of global food production is lost to poor resource management. An LLM that can reason about irrigation timing, pest response, and market timing has direct applicability to autonomous farming systems, precision agriculture advisors, and agricultural extension services in developing countries.

- **Scientific grounding is rare in RL envs** — Most RL environments use arbitrary game mechanics. Ours uses real agricultural science: FAO-56 Penman-Monteith evapotranspiration, Almgren-Chriss market impact models, and crop coefficients from agronomic literature. This is the kind of environment a researcher *would* cite.

---

### Q: Could a researcher write a paper about training on this?

**Yes.** Several paper-worthy angles exist:

1. **"Teaching LLMs Temporal Resource Management via Agricultural Simulation"** — Demonstrating that GRPO fine-tuning on FarmSimulation improves an LLM's ability to plan across 30-60 step horizons with compound resource constraints.

2. **"Dense Reward Shaping for Multi-Domain Sequential Decision Making"** — Our reward function spans 5 domains (hydrology, ecology, economics, agronomy, logistics) with ~15 distinct reward signals. A paper could study which reward components contribute most to learning.

3. **"Transfer from Simulation to Real Agricultural Advisory"** — If a model trained on FarmSimulation improves at answering real agricultural planning questions (e.g., "Given current soil moisture and a drought forecast, should I irrigate now or wait?"), that's a publication-worthy finding.

4. **"Curriculum Difficulty Scaling for LLM RL: Easy → Medium → Hard Task Progression"** — Our 3-task curriculum (stable climate → market timing → drought survival) is designed so optimal Task 1 strategies *fail* on Task 3. This progressive failure is an interesting training signal.

---

## Section 2: Design a Reward Signal That Actually Teaches

### Q: Does your reward function provide a rich, informative signal (not just 0/1 at the end)?

**Yes.** Our reward is dense and multi-dimensional:

| Category | Signal | Frequency |
|---|---|---|
| Planting | +0.20 per plant action | Per action |
| Irrigation rescue | +0.50 when saving a critically dry crop | Per action |
| Routine irrigation | +0.10 for normal watering | Per action |
| Wasteful irrigation | −0.50 for overwatering (moisture > 0.8) | Per action |
| Harvest | Up to +1.00, proportional to yield/max_yield | Per action |
| Smart sell | +0.30 + premium bonus for above-base pricing | Per action |
| Smart wait | +0.05 per growing plot (patience reward) | Per action |
| Dangerous wait | −0.30 per mature plot (wither risk) | Per action |
| Idle wait | −0.50 when nothing is happening | Per action |
| Health maintenance | +0.10–0.15 per healthy plot per day | Daily |
| Crop wither | −2.0 to −5.0 (scaled by difficulty) | On event |
| Terminal bonus | Up to +10.0 (profit + stewardship + efficiency) | End of episode |

**Total: 15+ distinct reward signals**, firing at action, daily, and episode levels. This is dramatically richer than a binary end-of-episode score.

---

### Q: Does your reward capture something hard to measure in a clever way?

**Yes, in three ways:**

1. **Market timing reward** — We don't just reward selling. We reward selling *above the 7-day average price*. The `price_premium` scalar in the sell handler computes `(execution_price - base_price) / base_price`, rewarding agents who learn to read the sinusoidal price cycles. This captures *strategic timing* — a skill LLMs currently lack.

2. **Stewardship score** — The terminal bonus includes a "stewardship" component: `healthy_days / total_days`. This captures sustained good management over time, not just a lucky end-state. An agent that lets crops wither on day 5 but recovers by day 30 gets a lower stewardship score than one that maintained health throughout.

3. **Efficiency score** — `1 - (wasteful_actions / total_actions)` penalizes agents who spray pesticide when there are no pests, irrigate when soil is saturated, or fertilize already-rich soil. This captures *precision* — doing the right thing at the right time, not just doing things.

---

### Q: Does your reward use OpenEnv's Rubric system thoughtfully?

**Partially — needs improvement.** Currently our grading is monolithic (one `grade_taskN()` function per task). To align with the judges' preference for composable rubrics, we should refactor into separate rubric dimensions:

- **Rubric 1: Profit** — `net_worth / target`
- **Rubric 2: Timing** — fraction of sells at above-average prices
- **Rubric 3: Stewardship** — healthy plot-days / total plot-days
- **Rubric 4: Efficiency** — 1 − wasteful actions ratio
- **Rubric 5: Resilience** — survived without bankruptcy

**Action needed:** Refactor `tasks.py` graders into composable rubric components that OpenEnv's Rubric system can display independently.

---

### Q: Is your reward hard to game?

**Mostly yes.** Several anti-gaming mechanisms are built in:

1. **Almgren-Chriss market impact** — An agent that tries to "game" by selling massive quantities gets progressively worse prices. Selling 50kg of corn in one go triggers 2.5% slippage and 5% permanent price depression. This prevents brute-force dump strategies.

2. **Wasteful action tracking** — Irrigating saturated soil, spraying pest-free plots, or fertilizing nutrient-rich soil all get negative rewards AND count toward the efficiency penalty. An agent can't spam actions to farm positive rewards.

3. **Wither penalties scale with difficulty** — On Hard mode, each withered crop costs −5.0 reward. This is 5× the maximum harvest reward (+1.0). You can't offset neglect with volume.

4. **Labor budget** — Only 10 hours per day. An agent can't take unlimited actions per day to brute-force outcomes.

**Potential gaming weakness:** An agent could learn to just `wait` repeatedly with growing crops to collect the +0.05/plot patience reward without ever engaging with the harder sell-timing mechanics. **Mitigation:** The terminal bonus (up to +10.0) heavily outweighs the cumulative patience rewards (~0.6 total), so a wait-only strategy would score poorly overall.

---

## Section 3: Show Real Training, End to End

### Q: Does your training loop connect to your environment (not a static dataset)?

**NOT YET — this is our #1 gap.** We have `inference.py` which runs zero-shot LLM evaluation against the environment, but no `train.py` that performs actual RL training.

**What we need to build:**
- `train.py` using Unsloth + TRL GRPOTrainer
- The training loop must call our `POST /reset` and `POST /step` endpoints live
- Each GRPO sample generates multiple action candidates, steps the env, and uses the env reward as the GRPO reward signal

---

### Q: Did you train long enough that the curves mean something?

**NOT YET.** We have zero training runs. We need to:
- Run at minimum 100 episodes (ideally 500+) on Task 1
- Show a clear upward trend in episode reward over training
- The curve should plateau or show diminishing returns (proving convergence, not noise)

---

### Q: Did you compare a trained agent vs a random/untrained baseline?

**Partially.** We have zero-shot baselines from Qwen-72B:
- Task 1: 0.42
- Task 2: 0.31
- Task 3: 0.19

We need to add:
- A **random baseline** (uniformly random valid actions) — expected score ~0.05
- A **trained model baseline** (after GRPO on TinyLlama) — target: >0.55 on Task 1
- Side-by-side plots showing all three on the same axes

---

### Q: Are the plots and numbers in your README and writeup?

**NOT YET.** The README has baseline scores in a table but no embedded plot images. We need:
- `reward_curve.png` — episode reward over training steps
- `baseline_comparison.png` — random vs zero-shot vs trained
- Embed both in README with one-line captions

---

## Section 4: Make Your Plots Readable

### Q: Are both axes labeled with units?

**N/A — no plots exist yet.** When we create them:
- X-axis: "Training Episode" or "Training Step"
- Y-axis: "Episode Reward (sum)" or "Task Grade (0-1)"
- Title: "FarmSimulation Task 1 — GRPO Training Progress"

---

### Q: Are plots saved as .png and committed to the repo?

**No.** We need to save all plots to `/Planning/plots/` or root directory and commit them.

---

### Q: Are key plots embedded in the README with captions?

**No.** We need to add:
```markdown
![Reward curve showing improvement from 0.42 to 0.65 over 200 episodes](reward_curve.png)
*Figure 1: Episode reward on Task 1 during GRPO training (TinyLlama 1.1B, 4-bit). Baseline zero-shot: 0.42.*
```

---

### Q: If you have multiple runs, are they on the same axes?

**N/A.** When we do comparisons, we'll overlay:
- Random agent (flat line at ~0.05)
- Zero-shot Qwen-72B (flat line at ~0.42)
- GRPO-trained TinyLlama (climbing curve)

All on one plot with a legend.

---

## Section 5: Tell a Story, Not an API Doc

### Q: What capability gap or interesting domain are you targeting?

**Capability gap:** LLMs cannot perform multi-step resource management under temporal constraints and market uncertainty. They make greedy short-term decisions instead of planning across 30-60 day horizons with interleaved resource dependencies.

**Domain:** Precision agriculture — a $4.1 trillion sector where autonomous decision-making could reduce the 30% global food waste caused by poor resource management.

---

### Q: What does the agent see, do, and get rewarded for?

**See:** A rich text summary of their farm: money, water levels, soil moisture per plot, crop health, growth stage, market prices with 7-day averages and trends, weather forecast, available labor hours.

**Do:** Choose from 12 actions — buy seeds, plant, irrigate, harvest, sell, pump water, fertilize, spray pesticide, weed, buy new plots, wait, or end the day. Each costs labor hours; 10 hours per day.

**Rewarded for:** Making the *right* action at the *right* time — irrigating dry crops (+0.5), selling at peak prices (+0.3 + premium), harvesting at full yield (+1.0). Punished for waste (−0.5 overwatering), neglect (−5.0 wither), and idle inaction (−0.5).

---

### Q: What changed after training? Show it.

**NOT YET DEMONSTRATED.** This is the critical gap. After we run GRPO training, we expect to show:

1. **Quantitative:** Task 1 grade improves from 0.42 (zero-shot) to >0.60 (trained)
2. **Qualitative:** The trained agent learns to:
   - Wait for price peaks before selling (instead of selling immediately)
   - Pre-irrigate before drought days (instead of reacting after crop damage)
   - Plant high-value corn only when it has enough water reserves for the 18-day growth period
3. **Behavioral:** Action distribution shifts from ~60% wait (zero-shot) to a more balanced mix of proactive actions

---

### Q: Why does it matter? Who would care?

1. **AI researchers** — This environment provides a benchmark for LLM temporal reasoning that goes beyond toy tasks. If GRPO training on FarmSimulation generalizes to other resource-management domains, that's a significant finding.

2. **Agricultural technologists** — Precision agriculture companies (John Deere, Bayer CropScience, Indigo Ag) are actively developing AI advisory systems. An LLM fine-tuned on realistic farming scenarios could power the next generation of crop management assistants.

3. **Development organizations** — The FAO, World Bank, and CGIAR fund digital agriculture tools for smallholder farmers in developing countries. An LLM that can advise on irrigation timing and pest response in resource-constrained settings addresses a real human need.

4. **RL/LLM training researchers** — The 3-task curriculum (easy → medium → hard) where optimal strategies are mutually incompatible provides a natural testbed for curriculum learning and transfer studies.

---

### Q: Can a reviewer read your README in 3-5 minutes and want to try your environment?

**Partially.** The README is comprehensive (755 lines) but currently reads more like an API doc than a story. It needs:

- [ ] A **2-paragraph hook** at the top that answers: "Why should I care?"
- [ ] A **Results section** (currently missing entirely) with embedded plots
- [ ] **Less technical jargon** in the overview — speak to a non-technical audience for the storytelling score
- [ ] **Links to video/blog** — currently missing

**Recommended README structure for judges:**
1. One-line pitch (what is this?)
2. Why it matters (2 paragraphs)
3. Quick demo screenshot/GIF
4. Results (reward curves, before/after table)
5. Try it yourself (HF Space link)
6. How it works (technical details for interested readers)
7. Links (video, blog, training notebook)

---

## Section 6: Engineer It Cleanly

### Q: Do you use OpenEnv's Environment / MCPEnvironment base classes properly?

**Yes.** `FarmingEnvironment` extends `Environment[FarmAction, FarmObservation, FarmState]` from `openenv.core`. It implements `reset()`, `step()`, `state()`, and `get_observation()` correctly.

**Note:** The judging docs mention `MCPEnvironment` — we should verify whether the latest OpenEnv release requires this instead of `Environment`.

---

### Q: Do you respect client/server separation?

**Yes.** `inference.py` uses `FarmEnvClient` which communicates via HTTP (`requests.Session`). It never imports `farming_environment`, `models`, or any server-side code. Clean separation.

---

### Q: Do you follow the standard Gym-style API?

**Yes.** Endpoints: `POST /reset`, `POST /step`, `GET /state`, `GET /health`. All implemented via `create_app()` from `openenv.core`.

---

### Q: Do you have a valid openenv.yaml?

**Mostly.** Current `openenv.yaml` is valid but missing `grader:` fields for each task. Fix needed:

```yaml
tasks:
  - id: task_1
    grader: "server.tasks:grade_task1"   # ← ADD THIS
```

---

### Q: Do you avoid reserved tool names?

**Yes.** Our 12 actions are: `buy_seeds`, `plant`, `irrigate`, `harvest`, `sell`, `pump_water`, `apply_fertilizer`, `spray_pesticide`, `pull_weeds`, `clear`, `buy_plot`, `wait`, `end_day`. None conflict with reserved names (`reset`, `step`, `state`, `close`).

---

## Summary: Gaps to Close Before Submission

| Priority | Gap | Effort | Impact on Score |
|---|---|---|---|
| 🔴 Critical | No training script (`train.py` / Colab) | 3 hours | 30% (pipeline + evidence) |
| 🔴 Critical | No reward curves or training evidence | 4 hours (needs GPU) | 20% (improvement evidence) |
| 🔴 Critical | No video or blog post | 2 hours | 30% (storytelling) |
| 🟡 Important | README reads like API doc, not a story | 1 hour | 30% (storytelling) |
| 🟡 Important | No `grader:` paths in openenv.yaml | 5 minutes | Submission validity |
| 🟡 Important | Placeholder `your-username` in README | 5 minutes | Professionalism |
| 🟢 Nice-to-have | Composable rubrics (OpenEnv Rubric system) | 2 hours | 10% (pipeline quality) |
| 🟢 Nice-to-have | Adversarial curriculum controller | 3 hours | 40% (innovation bonus) |

**Best theme fit:** Theme #2 (Long-Horizon Planning) or Theme #3.1 (World Modeling — Professional Tasks). FarmSimulation is a 30-60 step planning problem with a dynamic partially-observable world model.
