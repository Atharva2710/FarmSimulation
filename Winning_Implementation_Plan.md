# 🏆 FarmSimulation: The Winning "3-Layer" Implementation Plan (v2)

This plan has been refined to move from a simple "grid-world" to an **LLM-Native Strategic Benchmark**. We will build an environment that forces the agent to use its linguistic superpower: reading messy, ambiguous, and human-style reports to infer a hidden world state.

---

## 🎯 The Core Mission (20-Hour Submission Deadline)
**"Your simulator thinks in numbers. Your agent reads in language. They never directly talk to each other — there is a translation layer in between."**

The goal is to have a **Submission-Ready** project (Env + Training + Rewards + README) within **20 hours**. The remaining 10 hours will be for hyper-tuning and "Top 10" feature additions like Adversarial Curricula.

---

## 🏗️ The 3-Layer Architecture

### Layer 1: The Simulator (Hidden Source of Truth)
*   **What it is:** The existing `FarmingEnvironment` and `models.py`.
*   **Role:** Tracks precise floats (Moisture: 0.4, Pest: 0.7, Price: 142).
*   **Integrity:** This layer is the **only** source for reward calculation. The reward never touches the "messy text".

### Layer 2: The Text Renderer (The Translation Layer)
*   **What it is:** A new logic block in `_build_text_summary()` that converts Layer 1 numbers into Layer 3 text.
*   **Role:** Acts like a "Farm Journalist" or "Neighbor Gossip". 
    *   *Example (Moisture 0.4):* "The regional bulletin warns of critical soil dehydration; the ground is cracking in several districts."
    *   *Example (Pests 0.7):* "Old man Ramesh across the road says he saw locusts in his maize. Might be nothing, might be trouble."

### Layer 3: The Agent (The Strategic Planner)
*   **What it is:** The LLM (Trained via GRPO).
*   **Role:** It *only* sees Layer 2 text. It must synthesize reports, cross-reference market rumors, and manage its own **Journal**.

---

## 📅 Refined 30-Hour Timeline

### Phase 1: Engineering & Layer 2 (Hours 0-6)
*   **Action:** Refactor `models.py` and `FarmingEnvironment`.
*   **Feature: Agent-Managed Memory (`write_journal`)**
    *   Add `WRITE_JOURNAL` to `ActionType`.
    *   Add `journal_entry: Optional[str]` to `FarmAction`.
    *   Store the journal in `FarmState` and append it to the `text_summary` in `get_observation()`.
    *   *Why:* Tests Theme #2 (Long-Horizon Planning) by seeing if the agent can learn to use its own notes to stay on track.
*   **Action:** Build the "Text Renderer" in `_build_text_summary`. Use 10-15 templates per variable with injected randomness.

### Phase 2: GRPO Training Pipeline (Hours 6-12)
*   **Algorithm:** **GRPO is Mandatory.** Use Unsloth + TRL `GRPOTrainer`.
*   **Goal:** Connect `train.py` to the live OpenEnv server.
*   **Verification:** Ensure the model's reward curve starts rising as it learns to decode the "messy text" reports.

### Phase 3: Evaluation & Baseline Plots (Hours 12-16)
*   **Action:** Generate irrefutable proof of improvement.
*   **Baselines:** 
    1.  **Random:** Clicks buttons randomly.
    2.  **Zero-Shot:** Base model gets confused by messy text.
    3.  **GRPO-Trained:** Model parses text correctly and optimizes resource ROI.
*   **Deliverable:** `plots/reward_curve.png` and `plots/baseline_comparison.png`.

### Phase 4: Submission Sprint (Hours 16-20)
*   **Action:** Finalize `README.md` and `openenv.yaml`.
*   **Storytelling:** Focus on the "Noisy Text" innovation.
*   **Deployment:** Push to Hugging Face Spaces.
*   **Status:** **SUBMISSION READY.**

### Phase 5: "Top 10" Feature Expansion (Hours 20-30)
*   **Action:** Implement the **Adversarial Co-Evolution Loop**.
*   **Action:** Add the **Market Volatility Engine** (Global events like "Fertilizer Shortage" affecting prices).
*   **Action:** Record the 2-minute "Wow" video.

---

## 🛠️ Specific Code Implementation Guide

### 1. `models.py` Updates
```python
class ActionType(str, Enum):
    ...
    WRITE_JOURNAL = "write_journal"

class FarmAction(Action):
    ...
    journal_entry: Optional[str] = Field(None, description="Notes to remember for future days.")
```

### 2. `FarmingEnvironment` Observation Refactor
The environment should maintain a `_journal: str` internal state.
`_build_text_summary` should now look like this:
1.  **Read numbers** from PlotState and ClimateState.
2.  **Select templates** for each (Drought, Pests, Market).
3.  **Append `self._journal`** so the agent sees its own thoughts.
4.  **Return** a single block of natural language text.

### 3. Training Script Strategy
*   Use `trl.GRPOTrainer` with a prompt template that encourages the use of the `write_journal` action for multi-day planning.
*   Reward signal remains the hard Layer 1 numbers (Yield, Profit, Health).

---

## 🏆 Innovation Argument for Judges
"FarmSimulation isn't just a simulator; it's a test of **Cross-Modal Reasoning**. We force an LLM to take unstructured, human-style 'noise' (Layer 2) and map it back to a hidden numerical 'truth' (Layer 1) to optimize long-horizon outcomes. This addresses **Theme #3.1 (World Modeling)** and **Theme #2 (Long-Horizon Planning)** simultaneously."

---
**Ready to execute. Start by refactoring `models.py` and adding the `write_journal` action.**
