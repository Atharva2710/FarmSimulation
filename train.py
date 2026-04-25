#!/usr/bin/env python
"""
GRPO Training for FarmSimulation
================================

Trains a small LLM (default: Qwen2.5-0.5B-Instruct) on the farming environment
using a GRPO-style policy gradient over rolled-out episodes.

Algorithm (GRPO over full episodes, value-net-free):
  For each iteration:
    1. Roll out K episodes (same task, sampled actions); collect per-step
       (prompt_ids, action_ids) and the episode total reward.
    2. Group-relative advantage:  A_k = (R_k - mean(R)) / (std(R) + eps)
    3. Teacher-forced forward pass on the concatenated (prompt + action) tokens
       to recover differentiable log-probs of the chosen action tokens.
    4. Loss = - mean_k( A_k * sum(log_probs over action tokens) )
              + beta * KL(policy || ref)         (optional, beta=0 by default)
    5. Backward + AdamW step on LoRA adapters only.

Hardware
--------
  - Recommended: Colab T4 (15 GB) or better.
  - Local sanity-check on a 4 GB GPU works with --tiny:
      qwen2.5-0.5B + 4-bit + LoRA r=8 + K=2 + max_steps=10.

Required env vars
-----------------
  HF_TOKEN          (read access for gated models)
  FARMING_ENV_URL   (default: http://localhost:7860)

Selected env vars (overridden by CLI flags)
-------------------------------------------
  MODEL_NAME        (default: Qwen/Qwen2.5-0.5B-Instruct)
  TASK_ID           1|2|3 (default 1)
  NUM_ITERATIONS    (default 50)
  GROUP_SIZE        K     (default 4)
  MAX_STEPS         per episode (default 30)
  LR                (default 5e-5)
  KL_COEF           (default 0.0)
  USE_4BIT          (default 1)
  WANDB_PROJECT     (if set, log to W&B)
  OUTPUT_DIR        (default ./grpo_checkpoints)

Output
------
  - LoRA adapter checkpoints under OUTPUT_DIR/iter_<N>/
  - Training log CSV at OUTPUT_DIR/training_log.csv
  - Optional W&B run if WANDB_PROJECT is set
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    env_url: str = "http://localhost:7860"
    task_id: int = 1
    num_iterations: int = 50
    group_size: int = 4
    max_steps: int = 30
    lr: float = 5e-5
    kl_coef: float = 0.0
    temperature: float = 0.9
    max_new_tokens: int = 80
    lora_r: int = 16
    lora_alpha: int = 32
    use_4bit: bool = True
    output_dir: Path = Path("./grpo_checkpoints")
    eval_every: int = 5
    save_every: int = 10
    wandb_project: Optional[str] = None
    seed: int = 42

    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> "TrainConfig":
        cfg = cls()
        cfg.model_name = args.model or os.getenv("MODEL_NAME", cfg.model_name)
        cfg.env_url = args.env_url or os.getenv("FARMING_ENV_URL", cfg.env_url)
        cfg.task_id = args.task_id or int(os.getenv("TASK_ID", cfg.task_id))
        cfg.num_iterations = args.num_iterations or int(os.getenv("NUM_ITERATIONS", cfg.num_iterations))
        cfg.group_size = args.group_size or int(os.getenv("GROUP_SIZE", cfg.group_size))
        cfg.max_steps = args.max_steps or int(os.getenv("MAX_STEPS", cfg.max_steps))
        cfg.lr = args.lr or float(os.getenv("LR", cfg.lr))
        cfg.kl_coef = args.kl_coef if args.kl_coef is not None else float(os.getenv("KL_COEF", cfg.kl_coef))
        cfg.use_4bit = bool(int(os.getenv("USE_4BIT", "1"))) and not args.no_4bit
        cfg.output_dir = Path(args.output_dir or os.getenv("OUTPUT_DIR", cfg.output_dir))
        cfg.wandb_project = os.getenv("WANDB_PROJECT") or args.wandb_project
        cfg.seed = args.seed or int(os.getenv("SEED", cfg.seed))
        if args.tiny:
            cfg.group_size = min(cfg.group_size, 2)
            cfg.max_steps = min(cfg.max_steps, 10)
            cfg.lora_r = 8
            cfg.lora_alpha = 16
            cfg.max_new_tokens = 48
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Env client (mirrors inference.py:FarmEnvClient — same wire format)
# ─────────────────────────────────────────────────────────────────────────────


class FarmEnvClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._s = requests.Session()

    def health(self) -> bool:
        try:
            return self._s.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except Exception:
            return False

    @staticmethod
    def _unwrap(raw: Dict[str, Any]) -> Dict[str, Any]:
        if "observation" in raw:
            flat = dict(raw["observation"])
            flat["reward"] = raw.get("reward")
            flat["done"] = raw.get("done", False)
            flat["metadata"] = raw.get("metadata", {})
            return flat
        return raw

    def reset(self, task_id: int = 1) -> Dict[str, Any]:
        r = self._s.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return self._unwrap(r.json())

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self._s.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        return self._unwrap(r.json())


# ─────────────────────────────────────────────────────────────────────────────
# Action parsing — mirrors inference.py
# ─────────────────────────────────────────────────────────────────────────────


_VALID_ACTION_TYPES = {
    "wait", "buy_seeds", "plant", "irrigate", "harvest", "sell",
    "pump_water", "apply_fertilizer", "spray_pesticide", "pull_weeds",
    "buy_plot", "clear", "end_day",
}
_FALLBACK_ACTION: Dict[str, Any] = {"action_type": "wait"}
_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_action(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return dict(_FALLBACK_ACTION)
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj.get("action", obj) if "action" in obj else obj
    except json.JSONDecodeError:
        pass
    for m in _JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj.get("action", obj) if "action" in obj else obj
        except json.JSONDecodeError:
            continue
    return dict(_FALLBACK_ACTION)


def validate_action(a: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(a, dict):
        return dict(_FALLBACK_ACTION)
    if a.get("action_type") not in _VALID_ACTION_TYPES:
        return dict(_FALLBACK_ACTION)
    out = {"action_type": a["action_type"]}
    if "plot_id" in a:
        try:
            pid = int(a["plot_id"])
            if 0 <= pid <= 7:
                out["plot_id"] = pid
        except (ValueError, TypeError):
            pass
    if "quantity" in a:
        try:
            q = int(a["quantity"])
            if q > 0:
                out["quantity"] = q
        except (ValueError, TypeError):
            pass
    if "seed_type" in a and a["seed_type"] in {"wheat", "rice", "corn"}:
        out["seed_type"] = a["seed_type"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = textwrap.dedent("""\
    You are an autonomous farm manager. Each turn you observe the farm state and pick ONE action.
    Goal: maximize net worth via survival, growth, and well-timed market sales.
    Priorities: keep crops alive (irrigate when moisture is low), plant when you have seeds,
    harvest when mature, sell when prices are above the 7-day average.
    Reply with EXACTLY one JSON object: {"action_type": "...", "plot_id": 0, "seed_type": "...", "quantity": 1}.
    Omit fields that don't apply. Valid action_type values: wait, buy_seeds, plant, irrigate, harvest,
    sell, pump_water, apply_fertilizer, spray_pesticide, pull_weeds, buy_plot, clear, end_day.""")


def build_user_message(obs: Dict[str, Any], history: List[str]) -> str:
    summary = obs.get("text_summary", "(no summary)")
    valid = obs.get("valid_actions", [])
    hist = "\n".join(history[-3:]) if history else "(none)"
    return textwrap.dedent(f"""\
        STATE:
        {summary}

        VALID ACTIONS THIS STEP: {', '.join(valid) if valid else '(any)'}

        RECENT HISTORY:
        {hist}

        Reply with one JSON action.""")


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    prompt_ids: "torch.Tensor"   # [P]
    action_ids: "torch.Tensor"   # [A]
    reward: float


@dataclass
class EpisodeRecord:
    steps: List[StepRecord]
    total_reward: float
    final_grade: float
    episode_len: int


def rollout_episode(
    cfg: TrainConfig,
    env: FarmEnvClient,
    model,
    tokenizer,
    device,
    sample: bool = True,
) -> EpisodeRecord:
    """One full episode. Generation runs under no_grad; we keep the prompt and
    sampled action token ids so we can re-score them with grad later."""
    import torch

    obs = env.reset(task_id=cfg.task_id)
    history: List[str] = []
    steps: List[StepRecord] = []
    total_reward = 0.0

    for t in range(cfg.max_steps):
        if obs.get("done"):
            break

        user_msg = build_user_message(obs, history)
        chat = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(chat, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        with torch.no_grad():
            gen = model.generate(
                prompt_ids,
                max_new_tokens=cfg.max_new_tokens,
                do_sample=sample,
                temperature=cfg.temperature if sample else 1.0,
                top_p=0.95 if sample else 1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        action_ids = gen[0, prompt_ids.shape[1]:]
        action_text = tokenizer.decode(action_ids, skip_special_tokens=True)

        action = validate_action(parse_action(action_text))
        try:
            obs = env.step(action)
        except Exception as exc:
            print(f"  [env err] {exc}", file=sys.stderr)
            obs = {"done": True, "reward": -1.0, "metadata": {}}

        r = float(obs.get("reward") or 0.0)
        total_reward += r
        history.append(f"t={t} {json.dumps(action)} -> r={r:+.2f}")

        steps.append(StepRecord(
            prompt_ids=prompt_ids[0].detach().cpu(),
            action_ids=action_ids.detach().cpu(),
            reward=r,
        ))
        if obs.get("done"):
            break

    final_grade = float(obs.get("metadata", {}).get("grade", 0.01))
    return EpisodeRecord(steps=steps, total_reward=total_reward, final_grade=final_grade, episode_len=len(steps))


# ─────────────────────────────────────────────────────────────────────────────
# Loss: teacher-forced log-prob of action tokens × group-relative advantage.
# Memory-efficient: forward + backward per step, accumulate grads, single
# optimizer.step() at the end of the iteration. Avoids holding K×T autograd
# graphs simultaneously (which OOMs on small GPUs).
# ─────────────────────────────────────────────────────────────────────────────


def grpo_step(
    episodes: List[EpisodeRecord],
    model,
    device,
    optimizer,
    ref_model=None,
    kl_coef: float = 0.0,
    grad_clip: float = 1.0,
) -> Tuple[float, float, float, float]:
    """Compute group-relative advantages, accumulate gradients across all
    (state, action) pairs in the K rollouts, then apply one optimizer step.
    Returns (mean_loss, mean_kl, mean_reward, std_reward)."""
    import torch
    import torch.nn.functional as F

    rewards = np.array([ep.total_reward for ep in episodes], dtype=np.float32)
    mean_r, std_r = float(rewards.mean()), float(rewards.std())
    if std_r < 1e-6:
        advantages = np.zeros_like(rewards)
    else:
        advantages = (rewards - mean_r) / (std_r + 1e-6)

    # Flatten all active (step, advantage) pairs
    active: List[Tuple[StepRecord, float]] = []
    for ep, adv in zip(episodes, advantages):
        if abs(adv) < 1e-6:
            continue
        for sr in ep.steps:
            if sr.action_ids.numel() > 0:
                active.append((sr, float(adv)))

    if not active:
        return 0.0, 0.0, mean_r, std_r

    N = len(active)
    optimizer.zero_grad()
    total_loss = 0.0
    total_kl = 0.0

    for sr, adv in active:
        prompt_ids = sr.prompt_ids.to(device)
        action_ids = sr.action_ids.to(device)
        full = torch.cat([prompt_ids, action_ids], dim=0).unsqueeze(0)
        attn = torch.ones_like(full)

        logits = model(full, attention_mask=attn).logits[0]            # [L, V]
        p_len = prompt_ids.shape[0]
        shift = logits[p_len - 1 : p_len - 1 + action_ids.shape[0]]    # predicts action tokens
        logp = F.log_softmax(shift, dim=-1)
        chosen_logp = logp.gather(-1, action_ids.unsqueeze(-1)).squeeze(-1)

        loss = -(adv * chosen_logp.sum()) / N

        if ref_model is not None and kl_coef > 0:
            with torch.no_grad():
                ref_logits = ref_model(full, attention_mask=attn).logits[0]
                ref_logp = F.log_softmax(
                    ref_logits[p_len - 1 : p_len - 1 + action_ids.shape[0]], dim=-1,
                )
            p_dist = logp.exp()
            kl = (p_dist * (logp - ref_logp)).sum(-1).sum() / N
            loss = loss + kl_coef * kl
            total_kl += float(kl.detach().cpu())

        loss.backward()
        total_loss += float(loss.detach().cpu())

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip,
    )
    optimizer.step()
    return total_loss, total_kl, mean_r, std_r


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────


def train(cfg: TrainConfig) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.output_dir / "training_log.csv"

    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Env
    env = FarmEnvClient(cfg.env_url)
    if not env.health():
        print(f"[fatal] env not reachable at {cfg.env_url}. Start it with:\n"
              f"  uvicorn server.app:app --host 0.0.0.0 --port 7860", file=sys.stderr)
        sys.exit(1)

    # Tokenizer / model
    print(f"[init] loading {cfg.model_name} (4bit={cfg.use_4bit}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.use_4bit and device == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("[warn] bitsandbytes unavailable; falling back to fp16.", flush=True)
            model_kwargs["torch_dtype"] = torch.float16
    elif device == "cuda":
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
    if device == "cpu":
        model = model.to(device)

    # LoRA
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if cfg.use_4bit and device == "cuda":
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Reference model (frozen) — only if kl_coef > 0
    ref_model = None
    if cfg.kl_coef > 0:
        print("[init] loading reference model (frozen) for KL ...", flush=True)
        ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **model_kwargs)
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    # W&B (optional)
    wandb_run = None
    if cfg.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg.wandb_project, config=vars(cfg))
        except Exception as exc:
            print(f"[warn] wandb init failed: {exc}", flush=True)

    # CSV log header
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "iter", "mean_reward", "std_reward", "mean_grade", "loss", "kl",
            "elapsed_s",
        ])

    # ── Training loop ────────────────────────────────────────────────────────
    t0 = time.time()
    for it in range(1, cfg.num_iterations + 1):
        it_start = time.time()
        episodes: List[EpisodeRecord] = []

        # Roll out K episodes
        for k in range(cfg.group_size):
            ep = rollout_episode(cfg, env, model, tokenizer, device, sample=True)
            episodes.append(ep)
            print(f"  [iter {it} rollout {k+1}/{cfg.group_size}] "
                  f"len={ep.episode_len} totR={ep.total_reward:+.2f} grade={ep.final_grade:.3f}",
                  flush=True)

        # GRPO update — per-step grad accumulation + single optimizer.step()
        loss_val, kl, mean_r, std_r = grpo_step(
            episodes, model, device, optimizer,
            ref_model=ref_model, kl_coef=cfg.kl_coef,
        )
        mean_grade = float(np.mean([ep.final_grade for ep in episodes]))

        elapsed = time.time() - t0
        print(f"[iter {it:3d}] meanR={mean_r:+.3f} stdR={std_r:.3f} "
              f"meanGrade={mean_grade:.3f} loss={loss_val:+.4f} kl={kl:.3f} "
              f"({time.time()-it_start:.1f}s, total {elapsed:.0f}s)",
              flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                it, mean_r, std_r, mean_grade, loss_val, kl, elapsed,
            ])
        if wandb_run is not None:
            wandb_run.log({
                "iter": it, "mean_reward": mean_r, "std_reward": std_r,
                "mean_grade": mean_grade, "loss": loss_val, "kl": kl,
                "elapsed_s": elapsed,
            })

        # Save adapter
        if it % cfg.save_every == 0 or it == cfg.num_iterations:
            ckpt_dir = cfg.output_dir / f"iter_{it:04d}"
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            print(f"  [ckpt] saved adapter to {ckpt_dir}", flush=True)

    # Final eval
    print("\n[final eval] greedy rollout ...", flush=True)
    eval_ep = rollout_episode(cfg, env, model, tokenizer, device, sample=False)
    print(f"  greedy: len={eval_ep.episode_len} totR={eval_ep.total_reward:+.2f} grade={eval_ep.final_grade:.3f}",
          flush=True)

    if wandb_run is not None:
        wandb_run.finish()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO training for FarmSimulation")
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--env-url", type=str, default=None)
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--num-iterations", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--kl-coef", type=float, default=None)
    ap.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tiny", action="store_true",
                    help="Sanity-check mode for tiny GPUs (forces small K, max_steps, lora_r)")
    args = ap.parse_args()

    cfg = TrainConfig.from_env_and_args(args)
    print("=" * 60, flush=True)
    print("  FarmSimulation GRPO Training", flush=True)
    print("=" * 60, flush=True)
    for k, v in vars(cfg).items():
        print(f"  {k:18s} = {v}", flush=True)
    print("=" * 60, flush=True)

    train(cfg)


if __name__ == "__main__":
    main()
