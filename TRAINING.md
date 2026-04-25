# Training the FarmSim Agent (GRPO)

This document is the runbook for `train.py`, which trains a small LLM on
`farming-env` using a **GRPO-style policy gradient** over rolled-out episodes.

It addresses two of the four hackathon judging axes directly:

| Judging axis | How `train.py` hits it |
|---|---|
| Showing improvement in rewards (20%) | Loss + reward + grade curves emitted to CSV / W&B |
| Reward & training pipeline setup (10%) | GRPO + LoRA + 4-bit, OpenEnv-compliant client |

The other two axes (Innovation 40%, Storytelling 30%) are handled by the
Three-Layer text rendering and the README/video — see the plan file for those.

---

## 1. Algorithm

We do **value-net-free policy gradient with group-relative advantages** —
i.e. GRPO, applied at the episode level.

For each iteration:

1. Roll out **K** episodes (default K=4) with sampling. Same task, different
   action sequences.
2. Score each by its **total episode reward** R_k.
3. Compute group-relative advantage **A_k = (R_k − mean R) / (std R + ε)**.
4. For every (state, action) pair across all K episodes, recover
   differentiable log-probs via a teacher-forced forward pass.
5. Loss = − mean_step( A_k · Σ log π(a_t | s_t) ).
   Optionally + β · KL(policy ‖ frozen-reference).
6. Per-step `loss.backward()` (gradient accumulation), one `optimizer.step()`
   at the iteration end. This keeps memory bounded — important on T4 / 1650.

Only LoRA adapters get updated; the base model is frozen (and 4-bit on GPU).

---

## 2. Hardware reality

| GPU | Verdict |
|---|---|
| **Colab T4 free (15 GB)** | ✅ recommended target; default config (Qwen 0.5B + 4-bit + LoRA r=16) fits comfortably |
| **Colab Pro A100 (40 GB)** | ✅ can step up to Qwen 1.7B with K=8 |
| **GTX 1650 (4 GB)** | ⚠️ won't fit even Qwen 0.5B + 4-bit reliably. Use `--tiny` for code-path sanity only; do not expect a real training run. |
| **CPU** | Painful but technically works for `--tiny`. |

If you have any GPU at the venue, prefer it over Colab to avoid the runtime
disconnect risk on long runs. Otherwise, Colab T4 is fine.

---

## 3. Run it — Colab (recommended)

Open `notebooks/train_grpo.ipynb` in Colab with a T4 runtime.

Set `HF_TOKEN` in Colab's secrets pane (left sidebar → 🔑) — the notebook reads
it via `google.colab.userdata`.

The notebook will:
1. Install training deps.
2. Pull the repo (edit `REPO_URL` first).
3. Start the env server in the background on `localhost:7860`.
4. Run `train.py` with default config.
5. Plot curves and save `training_curves.png`.

A 50-iteration run with K=4, max_steps=30 takes ~30–60 min on T4 for Qwen 0.5B.

---

## 4. Run it — CLI

After `pip install -r requirements-train.txt`:

```bash
# Terminal 1 — start the env
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Terminal 2 — train
export HF_TOKEN=hf_xxx
python train.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task-id 1 \
    --num-iterations 50 \
    --group-size 4 \
    --max-steps 30 \
    --lr 5e-5 \
    --output-dir ./grpo_checkpoints
```

For a 30-second smoke test:
```bash
python train.py --tiny --num-iterations 2
```

---

## 5. Outputs

| Path | What |
|---|---|
| `grpo_checkpoints/training_log.csv` | Per-iteration metrics: `iter, mean_reward, std_reward, mean_grade, loss, kl, elapsed_s` |
| `grpo_checkpoints/iter_<N>/` | LoRA adapter checkpoints every `--save-every` iterations |
| W&B run | If `WANDB_PROJECT` env var is set |

Plot the curves with the cell at the end of the notebook, or this snippet:

```python
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('grpo_checkpoints/training_log.csv')
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(df['iter'], df['loss']);        ax[0].set_title('Loss')
ax[1].plot(df['iter'], df['mean_reward']); ax[1].set_title('Mean episode reward')
ax[2].plot(df['iter'], df['mean_grade']);  ax[2].set_title('Mean grade')
plt.tight_layout(); plt.savefig('assets/training_curves.png', dpi=140)
```

---

## 6. Configuration reference

All flags accept env-var overrides too. Defaults shown.

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--model` | `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Base model to fine-tune |
| `--env-url` | `FARMING_ENV_URL` | `http://localhost:7860` | OpenEnv server URL |
| `--task-id` | `TASK_ID` | `1` | Curriculum task (1=easy, 2=medium, 3=hard) |
| `--num-iterations` | `NUM_ITERATIONS` | `50` | Outer training iterations |
| `--group-size` | `GROUP_SIZE` | `4` | K rollouts per iteration |
| `--max-steps` | `MAX_STEPS` | `30` | Env steps per episode |
| `--lr` | `LR` | `5e-5` | Adam LR (LoRA only) |
| `--kl-coef` | `KL_COEF` | `0.0` | KL penalty against frozen ref (set >0 to enable) |
| `--no-4bit` |  | (off) | Disable 4-bit; use fp16 |
| `--output-dir` | `OUTPUT_DIR` | `./grpo_checkpoints` | Where to write checkpoints + log |
| `--wandb-project` | `WANDB_PROJECT` | none | W&B project name |
| `--seed` | `SEED` | `42` | RNG seed |
| `--tiny` |  | (off) | Sanity-check mode: forces K=2, max_steps=10, lora_r=8 |

Required: `HF_TOKEN` for gated models on the HF hub.

---

## 7. What to expect

**Healthy curves** (Task 1, Qwen 0.5B):

- **Mean episode reward** trends upward over 30–50 iters; large variance early
  is normal because the policy's JSON output is unstable until LoRA learns
  the format.
- **Mean grade** climbs from ~0.05 (random/babbling) toward 0.3+ within 50
  iters. Beating zero-shot Qwen 2.5-72B (target 0.42 per the README) on a 0.5B
  model is unlikely — but the *improvement curve* is what the judges score, not
  the absolute number.
- **Loss** is policy-gradient loss; absolute value isn't meaningful, but it
  should stay bounded (no NaNs, no monotonic explosions).

**Unhealthy signals**:

- Mean reward stuck at the lowest possible value → model never produces valid
  JSON. Add a few-shot example to `SYSTEM_PROMPT`, or BC-warmstart on heuristic
  trajectories (see "Future work" below).
- Loss explodes / NaN → reduce `--lr` to `1e-5`, add `--kl-coef 0.05`.
- All episodes get identical rewards → group-relative advantage is 0,
  no learning signal. Increase `--temperature` to 1.0 to encourage exploration.

---

## 8. Future work (post-finale)

- BC warm-start: pretrain on heuristic-agent rollouts (`server/agents/heuristic.py`)
  for 1 epoch before GRPO, like MyoChallenge's winning recipe.
- Process Reward Models — score each step's *reasoning* quality, not just the
  final outcome (per `Planning/reward engg.md`).
- Adversarial curriculum — ramp `market_noise` and `overhead` as `mean_grade`
  rises (Kube SRE Gym's pattern).
- Train against the **narrative** observation mode once the structured-mode
  curve is in the bag (see plan §5b for why we deliberately do NOT train on
  text observations in 30 hours).
