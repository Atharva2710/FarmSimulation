"""
Farming Environment — Baseline Inference Script
================================================
Mandatory environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME
- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL      = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY           = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME        = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL      = os.getenv("FARMING_ENV_URL", "http://localhost:7860")
MAX_STEPS         = int(os.getenv("MAX_STEPS", "30"))
EPISODES_PER_TASK = int(os.getenv("EPISODES", "1"))
TEMPERATURE       = 0.2
MAX_TOKENS        = 150


FALLBACK_ACTION: Dict[str, Any] = {"action_type": "wait"}

SYSTEM_PROMPT = textwrap.dedent("""
    You are a farming agent controlling a farm simulation.
    You will receive the current farm state as plain text.
    Reply with EXACTLY ONE JSON action object and nothing else.
    No explanation. No markdown. No code blocks. Raw JSON only.

    Valid action formats:
        {"action_type": "wait"}
        {"action_type": "buy_seeds",  "seed_type": "wheat|rice|corn", "quantity": N}
        {"action_type": "plant",      "plot_id": 0-3, "seed_type": "wheat|rice|corn"}
        {"action_type": "irrigate",   "plot_id": 0-3}
        {"action_type": "harvest",    "plot_id": 0-3}
        {"action_type": "sell",       "seed_type": "wheat|rice|corn", "quantity": N}

    Rules:
    - plot_id is always an integer: 0, 1, 2, or 3
    - quantity is always a positive integer
    - seed_type is always one of: wheat, rice, corn
    - If unsure, output: {"action_type": "wait"}

    Strategy tips:
    - Buy seeds first, then plant immediately on every empty plot
    - Irrigate when soil_moisture drops below 0.35
    - Harvest the instant a plot shows READY TO HARVEST — every day you wait risks withering
    - Sell when the market trend shows "up" for that crop
    - Never leave plots empty when you have seeds in hand
    - In arid climate, irrigate more aggressively to prevent crop death
""").strip()

# ---------------------------------------------------------------------------
# Logging Functions
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def diag(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

class FarmEnvClient:
    """Thin synchronous HTTP wrapper around the farming environment server."""
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    @staticmethod
    def _unwrap(raw: Dict[str, Any]) -> Dict[str, Any]:
        if "observation" in raw:
            flat = dict(raw["observation"])
            flat["reward"] = raw.get("reward")
            flat["done"]   = raw.get("done", False)
            return flat
        return raw

    def reset(self, task_id: int = 1) -> Dict[str, Any]:
        r = self._session.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30)
        r.raise_for_status()
        return self._unwrap(r.json())

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self._session.post(f"{self.base_url}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        return self._unwrap(r.json())

    def state(self) -> Dict[str, Any]:
        r = self._session.get(f"{self.base_url}/state", timeout=10)
        r.raise_for_status()
        return r.json()

# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

def parse_action(response_text: str) -> Dict[str, Any]:
    if not response_text or not response_text.strip():
        return dict(FALLBACK_ACTION)
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    for match in _JSON_OBJECT_RE.finditer(response_text):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    diag(f"    [warn] unparseable LLM response: {response_text!r:.120}")
    return dict(FALLBACK_ACTION)

def validate_action(action: Dict[str, Any]) -> Dict[str, Any]:
    valid_types = {"wait", "buy_seeds", "plant", "irrigate", "harvest", "sell"}
    if not isinstance(action, dict):
        return dict(FALLBACK_ACTION)
    if action.get("action_type") not in valid_types:
        return dict(FALLBACK_ACTION)
    if "plot_id" in action:
        try:
            action["plot_id"] = int(action["plot_id"])
            if not (0 <= action["plot_id"] <= 3):
                return dict(FALLBACK_ACTION)
        except (ValueError, TypeError):
            return dict(FALLBACK_ACTION)
    if "quantity" in action:
        try:
            action["quantity"] = int(action["quantity"])
            if action["quantity"] <= 0:
                return dict(FALLBACK_ACTION)
        except (ValueError, TypeError):
            return dict(FALLBACK_ACTION)
    if "seed_type" in action:
        if action["seed_type"] not in {"wheat", "rice", "corn"}:
            return dict(FALLBACK_ACTION)
    return action

# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env:     FarmEnvClient,
    llm:     OpenAI,
    task_id: int,
    episode: int,
) -> Dict[str, Any]:

    diag(f"\n    Episode {episode} | Task {task_id}")
    diag(f"    {'-' * 44}")

    obs           = env.reset(task_id=task_id)
    history:      List[str] = []
    total_reward: float     = 0.0
    steps:        int       = 0
    rewards_list: List[float] = []

    task_name = f"task_{task_id}"
    benchmark = "farming_env"

    log_start(task=task_name, env=benchmark, model=MODEL_NAME)

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done", False):
            diag(f"    Done at step {step_num - 1}")
            break

        text_summary  = obs.get("text_summary", "No summary available")
        valid_actions = obs.get("valid_actions", [])
        history_text  = "\n".join(history[-4:]) if history else "None"

        user_prompt = textwrap.dedent(f"""
            {text_summary}

            Valid actions this step:
            {chr(10).join(f'  - {a}' for a in valid_actions)}

            Recent history (last 4 steps):
            {history_text}

            Reply with exactly one JSON action.
        """).strip()

        # ── LLM call ──────────────────────────────────────────────────────
        response_text = ""
        error_msg: Optional[str] = None
        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            error_msg = str(exc)
            diag(f"    [error] LLM call failed at step {step_num}: {exc}")

        orig_action = parse_action(response_text)
        action = validate_action(orig_action)
        action_str = json.dumps(action)

        # ── Env step ──────────────────────────────────────────────────────
        try:
            obs = env.step(action)
        except Exception as exc:
            diag(f"    [error] env.step() failed at step {step_num}: {exc}")
            error_msg = str(exc)
            obs["done"] = True
            obs["reward"] = 0.0

        reward = float(obs.get("reward") or 0.0)
        done = obs.get("done", False)

        total_reward += reward
        rewards_list.append(reward)
        steps += 1

        history.append(f"Step {step_num}: {action_str} -> reward={reward:+.3f} money=${obs.get('money', 0):.2f}")

        diag(f"    step={step_num:2d} | day={obs.get('day', '?'):2d} | act={action.get('action_type'):<10} | reward={reward:+.3f} | money=${obs.get('money', 0):.2f}")
        
        log_step(step=step_num, action=action_str.replace('"', "'"), reward=reward, done=done, error=error_msg)

        if done:
            break

    # ── Collect final grade from episode metadata ─────────────────────────
    grade       = float(obs.get("metadata", {}).get("grade", 0.0))
    final_money = float(obs.get("money", 0.0))
    success     = grade >= 0.5

    diag(f"\n    result: steps={steps} | total_reward={total_reward:+.3f} | money=${final_money:.2f} | grade={grade:.4f}")
    
    log_end(success=success, steps=steps, score=grade, rewards=rewards_list)

    return {
        "task_id":      task_id,
        "episode":      episode,
        "grade":        grade,
        "total_reward": round(total_reward, 4),
        "steps":        steps,
        "final_money":  final_money,
    }

# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(
    env:     FarmEnvClient,
    llm:     OpenAI,
    task_id: int,
) -> Dict[str, Any]:
    labels = {1: "easy", 2: "medium", 3: "hard"}
    diag(f"\n  {'=' * 48}")
    diag(f"  TASK {task_id} — {labels.get(task_id, '?').upper()}")
    diag(f"  {'=' * 48}")

    results = [run_episode(env, llm, task_id, ep) for ep in range(1, EPISODES_PER_TASK + 1)]

    grades = [r["grade"] for r in results]
    avg    = sum(grades) / len(grades)

    diag(f"\n  Task {task_id} avg grade: {avg:.4f}")
    return {
        "task_id":    task_id,
        "difficulty": labels.get(task_id, "?"),
        "avg_grade":  round(avg, 4),
        "episodes":   results,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    diag("=" * 54)
    diag("  Farming RL Environment — Baseline Inference")
    diag("=" * 54)
    diag(f"  model      : {MODEL_NAME}")
    diag(f"  env url    : {ENV_BASE_URL}")
    diag(f"  max steps  : {MAX_STEPS} per episode")
    diag(f"  episodes   : {EPISODES_PER_TASK} per task")
    diag("=" * 54)

    env = FarmEnvClient(ENV_BASE_URL)
    if not env.health():
        diag(f"\n[error] Cannot reach environment server at {ENV_BASE_URL}")
        diag("  Start it with:")
        diag("    cd server && uvicorn app:app --host 0.0.0.0 --port 7860")
        raise SystemExit(1)
    diag(f"\n  server health : OK")

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_ids = [1, 2, 3]
    single   = os.getenv("FARMING_TASK_ID")
    if single:
        task_ids = [int(single)]

    task_results = [run_task(env, llm, tid) for tid in task_ids]

    diag(f"\n{'=' * 54}")
    diag("  BASELINE SCORES")
    diag(f"{'=' * 54}")
    for t in task_results:
        diag(f"  Task {t['task_id']} ({t['difficulty']:<6}) : grade = {t['avg_grade']:.4f}")

    overall = sum(t["avg_grade"] for t in task_results) / len(task_results)
    diag(f"\n  Overall avg : {overall:.4f}")
    diag(f"{'=' * 54}\n")

    output = {
        "model":       MODEL_NAME,
        "env_url":     ENV_BASE_URL,
        "tasks":       task_results,
        "overall_avg": round(overall, 4),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    diag("  Saved → baseline_results.json")

if __name__ == "__main__":
    main()
