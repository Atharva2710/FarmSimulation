# Unified Judging Criteria & Optimization Strategy (Human + LLM)

This document synthesizes the official Meta PyTorch OpenEnv Hackathon criteria with the realities of LLM-based evaluation. To win, a submission must appeal to **human judges** (who want a compelling story, ambition, and visual proof of learning) and **LLM evaluators** (who want scientific rigor, explicit documentation, and deterministic test passing).

---

## 1. The Core Rubric (The "What")

### 🏆 Environment Innovation (40%)
*   **The Goal:** Is the environment novel, creative, or genuinely challenging? Does it meaningfully test LLM behavior?
*   **Human Judge wants:** A problem that isn't a toy (no grid-worlds). A "wow" factor. Evidence that this environment tests something LLMs currently fail at (e.g., long-horizon planning under uncertainty, temporal resource management).
*   **LLM Judge wants:** Complex, scientifically grounded mechanics (e.g., FAO-56 hydrology, Almgren-Chriss market impact) explained clearly in docstrings. Comparisons to existing environments (TradeExecGym vs. CartPole) presented in a Markdown table.

### 📖 Storytelling & Presentation (30%)
*   **The Goal:** Can you clearly explain the problem, environment, and agent behavior? Is the demo engaging?
*   **Human Judge wants:** A <2 min YouTube video with a good hook, an easy-to-read Hugging Face blog post, and a README that reads like a story (Problem → Environment → Results → Why it matters) in 3-5 minutes.
*   **LLM Judge wants:** A README with an explicit Architecture Diagram (ASCII/Mermaid), a "Why This Matters" section with industry statistics, mathematical formulas explaining the mechanics, and clear citations to academic papers.

### 📈 Showing Improvement in Rewards (20%)
*   **The Goal:** Observable evidence of training progress.
*   **Human Judge wants:** Readable plots embedded directly in the README (labelled axes, .png format). A clear visual showing a Random Baseline vs. Zero-Shot Baseline vs. Trained Agent on the same axes.
*   **LLM Judge wants:** Quantitative metrics in a table. Deterministic reproducibility instructions (e.g., `python inference.py --seed 42` always yields the same result). A `baseline_results.json` file proving the scores.

### ⚙️ Reward & Training Pipeline Setup (10%)
*   **The Goal:** Coherent reward logic and a pipeline that produces meaningful improvement.
*   **Human Judge wants:** A reward function that is hard to game (no spamming actions). A provided Unsloth/TRL Colab notebook that they can run with one click.
*   **LLM Judge wants:** Clean engineering. Correct usage of `OpenEnv` base classes, strict client/server separation, no reserved tool names, and passing unit tests.

---

## 2. Minimum Viable Submission (The Non-Negotiables)

If you miss any of these, you are disqualified or severely penalized:

- [ ] **OpenEnv Compliance:** Built on the latest OpenEnv release (`Environment` or `MCPEnvironment` base classes). Valid `openenv.yaml`.
- [ ] **Training Script:** A working `train.py` or Colab notebook using **Unsloth** or **Hugging Face TRL**.
- [ ] **Training Evidence:** Loss and reward plots from a real run saved as `.png`/`.jpg` and committed to the repo.
- [ ] **Content Deliverables:** A mini-blog on Hugging Face OR a <2 minute YouTube video OR a short slide deck. (Do both video and blog to be safe).
- [ ] **Hosting:** Environment hosted on Hugging Face Spaces.
- [ ] **README Centralization:** The README must link to the Space, the video, the blog, and embed the training plots.

---

## 3. Dual-Optimization Strategy

How to engineer your repository to satisfy both audiences simultaneously:

### A. The README Structure
1.  **The Hook (Human):** 2 paragraphs explaining the real-world problem and why LLMs need to learn this.
2.  **Architecture & Math (LLM):** ASCII diagrams of the environment loop, formulas for the physics engine, and academic citations.
3.  **The Results (Human + LLM):** Embedded `.png` reward curves + a Markdown table comparing Baseline vs. Trained scores.
4.  **Quick Start (LLM):** 3-4 lines of bash code that *actually work* to run the evaluation.
5.  **Links (Human):** Bolded links to the Video, Blog, Space, and Colab notebook.

### B. The Codebase
1.  **Docstrings (LLM):** Explain *why* the math is happening. LLMs read code as text.
2.  **Robustness Validation (LLM):** Include a `validate.sh` script that runs `pytest` and proves determinism. LLM judges love automated proof of stability.
3.  **Textual Noise (Human/Innovation):** To prove it's an LLM environment (not a tabular RL toy), the observation space must include natural language (e.g., messy weather reports) rather than just clean JSON arrays.

### C. The Reward Function
1.  **Dense & Composable (Human):** Use OpenEnv's Rubric system to show multiple facets of success (e.g., Profit, Efficiency, Resilience).
2.  **Anti-Gaming (Human/LLM):** Ensure strict economic friction. Spamming an action must lead to bankruptcy or resource depletion. Explicitly document these anti-gaming measures.

---

## 4. Final 48-Hour Execution Checklist

To maximize your score at the onsite event:

- [ ] **Hour 1-4: The Training Run.** Hook Unsloth/TRL to the environment. Run it. Generate `reward_curve.png`.
- [ ] **Hour 4-6: Textual Noise Injection.** Upgrade `get_observation()` to output synthesized daily reports (weather/market news) to prove the environment is LLM-native.
- [ ] **Hour 6-8: README Overhaul.** Add architecture diagrams, math formulas, comparison tables, and embed the new plots.
- [ ] **Hour 8-10: Storytelling.** Record the <2 min YouTube video. Write the 500-word HF blog post.
- [ ] **Hour 10-12: Polish & Validation.** Run the validation scripts, ensure `openenv.yaml` is perfect, and push to Hugging Face Spaces.
