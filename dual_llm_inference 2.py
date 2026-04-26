"""
Dual-LLM Inference Script — Farming RL Environment
====================================================
Two Ollama models work together each step:

  Model 1 — ACTOR   : Reads the farm state, picks the best action.
  Model 2 — CRITIC  : Evaluates the proposed action's risk/reward
                      and either approves it or overrides with a safer one.

Run locally with Ollama:
    # Terminal 1 — start the farm server
    cd server && uvicorn app:app --host 0.0.0.0 --port 7860

    # Terminal 2 — run this script
    python dual_llm_inference.py

Environment variables (all optional — defaults shown):
    ACTOR_MODEL    ollama model for the Actor   (default: llama3.2)
    CRITIC_MODEL   ollama model for the Critic  (default: mistral)
    OLLAMA_URL     Ollama API base URL           (default: http://localhost:11434/v1)
    FARMING_ENV_URL  Farm server URL             (default: http://localhost:7860)
    MAX_STEPS      Max steps per episode         (default: 30)
    FARMING_TASK_ID  Run a single task (1/2/3)   (default: all three)
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ─── Config ──────────────────────────────────────────────────────────────────

OLLAMA_URL   = os.getenv("OLLAMA_URL",       "http://localhost:11434/v1")
ACTOR_MODEL  = os.getenv("ACTOR_MODEL",      "llama3.2")
CRITIC_MODEL = os.getenv("CRITIC_MODEL",     "mistral")
ENV_URL      = os.getenv("FARMING_ENV_URL",  "http://localhost:7860")
MAX_STEPS    = int(os.getenv("MAX_STEPS",    "30"))
TEMPERATURE  = 0.2

VALID_ACTIONS = [
    "wait", "end_day", "buy_seeds", "plant", "irrigate",
    "harvest", "sell", "clear", "apply_fertilizer",
    "spray_pesticide", "pull_weeds", "pump_water",
]

FALLBACK_ACTION: Dict[str, Any] = {"action_type": "wait"}

# ─── Prompts ──────────────────────────────────────────────────────────────────

ACTOR_SYSTEM = textwrap.dedent("""
    You are the FARM ACTOR — a strategic decision-maker for a farming simulation.
    Your job: read the current farm state and choose the SINGLE best action to
    maximize long-term profit.

    AVAILABLE ACTIONS:
    - wait                   : do nothing (end sub-step)
    - end_day                : advance to next day
    - buy_seeds              : {seed_type: wheat|rice|corn, quantity: N}
    - plant                  : {plot_id: 0-3, seed_type: wheat|rice|corn}
    - irrigate               : {plot_id: 0-3}
    - harvest                : {plot_id: 0-3}
    - sell                   : {seed_type: wheat|rice|corn, quantity: N}
    - clear                  : {plot_id: 0-3}  (remove dead/withered crop)
    - apply_fertilizer       : {plot_id: 0-3}
    - spray_pesticide        : {plot_id: 0-3}
    - pull_weeds             : {plot_id: 0-3}
    - pump_water             : {}  (refill water tank from aquifer)

    PRIORITIES (in order):
    1. If any plot is WITHERED → clear it immediately
    2. If water tank < 20% AND aquifer > 0 → pump_water
    3. If any plot moisture < 30% → irrigate that plot
    4. If pests > 0.3 on any plot → spray_pesticide
    5. If weeds > 0.3 on any plot → pull_weeds
    6. If ready-to-harvest plot exists → harvest
    7. If money > $80 and empty plot exists → buy_seeds + plant corn
    8. If storage > 0 and market price is high → sell
    9. Otherwise → end_day

    RESPOND WITH STRICT JSON ONLY — no markdown, no explanation outside JSON:
    {
      "thought": "brief reasoning (1-2 sentences)",
      "action": {
        "action_type": "...",
        "plot_id": 0,
        "seed_type": "corn",
        "quantity": 1
      }
    }
    Only include plot_id / seed_type / quantity if the action needs them.
""").strip()

CRITIC_SYSTEM = textwrap.dedent("""
    You are the FARM CRITIC — a risk/reward evaluator for a farming simulation.
    You review an action proposed by the Actor and decide:
      1. APPROVE it if the risk is acceptable
      2. OVERRIDE it with a safer action if it's too risky

    RISK FACTORS to check:
    - 💀 CRITICAL: Spending money when money < $20 (bankruptcy risk)
    - 💀 CRITICAL: Planting when water tank < 15% (crop will die of thirst)
    - ⚠️  HIGH: Selling when market price is below 7-day average (bad timing)
    - ⚠️  HIGH: Irrigating when soil moisture already > 80% (waste of water)
    - ⚠️  HIGH: Skipping pest/weed control when severity > 0.5 (yield damage)
    - ℹ️  LOW: Waiting when there is clear action available (opportunity cost)

    REWARD FACTORS to check:
    - ✅ Harvesting a mature crop = high immediate reward
    - ✅ Selling when price > 7-day average = market timing bonus
    - ✅ Planting with good moisture = good growth setup
    - ✅ Pumping water when tank is low = survival insurance

    RESPOND WITH STRICT JSON ONLY — no markdown, no explanation outside JSON:
    {
      "risk_level": "low|medium|high|critical",
      "risk_reason": "brief explanation of the main risk",
      "reward_estimate": "low|medium|high",
      "reward_reason": "brief explanation of the expected reward",
      "verdict": "approve|override",
      "final_action": {
        "action_type": "...",
        "plot_id": 0,
        "seed_type": "corn",
        "quantity": 1
      },
      "override_reason": "why you overrode (leave empty string if approved)"
    }
    If verdict is "approve", copy the proposed action into final_action unchanged.
    Only include plot_id / seed_type / quantity in final_action if needed.
""").strip()

# ─── Logging ─────────────────────────────────────────────────────────────────

def diag(msg: str) -> None:
    print(msg, flush=True)

def log_start(task: int, actor: str, critic: str) -> None:
    print(f"[START] task={task} actor={actor} critic={critic}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             verdict: str, risk: str) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} verdict={verdict} risk={risk}",
        flush=True
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True
    )

# ─── JSON extraction helper ───────────────────────────────────────────────────

def extract_json(text: str) -> Optional[Dict]:
    """Extract first valid JSON object from a string (handles markdown fences)."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None

def clean_action(raw: Dict) -> Dict[str, Any]:
    """Sanitize an action dict — remove None/missing fields, validate action_type."""
    action: Dict[str, Any] = {}
    atype = raw.get("action_type", "wait")
    if atype not in VALID_ACTIONS:
        atype = "wait"
    action["action_type"] = atype

    for field in ("plot_id", "seed_type", "quantity"):
        val = raw.get(field)
        if val is not None:
            action[field] = val

    return action

# ─── Farm Environment HTTP Client ────────────────────────────────────────────

class FarmClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._s = requests.Session()

    def health(self) -> bool:
        try:
            r = self._s.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def reset(self, task_id: int) -> Dict:
        r = self._s.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict) -> Dict:
        r = self._s.post(
            f"{self.base_url}/step",
            json={"action": action},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()

# ─── Actor ───────────────────────────────────────────────────────────────────

def actor_choose_action(
    llm: OpenAI,
    obs_text: str,
    history: List[str],
) -> Tuple[Dict[str, Any], str]:
    """
    Model 1 — Actor.
    Returns (proposed_action_dict, actor_thought).
    """
    # Compact recent history (last 4 steps)
    history_block = "\n".join(history[-4:]) if history else "No actions taken yet."

    user_msg = textwrap.dedent(f"""
        === CURRENT FARM STATE ===
        {obs_text}

        === RECENT HISTORY (last 4 steps) ===
        {history_block}

        Choose the best action now. Respond with JSON only.
    """).strip()

    try:
        resp = llm.chat.completions.create(
            model=ACTOR_MODEL,
            messages=[
                {"role": "system", "content": ACTOR_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=300,
        )
        raw_text = resp.choices[0].message.content or ""
        parsed = extract_json(raw_text)
        if parsed:
            action  = clean_action(parsed.get("action", {}))
            thought = parsed.get("thought", "—")
            return action, thought
    except Exception as e:
        diag(f"    [ACTOR ERROR] {e}")

    return FALLBACK_ACTION.copy(), "Fallback: Actor failed."

# ─── Critic ──────────────────────────────────────────────────────────────────

def critic_evaluate(
    llm: OpenAI,
    obs_text: str,
    proposed_action: Dict[str, Any],
    actor_thought: str,
) -> Tuple[Dict[str, Any], str, str, str]:
    """
    Model 2 — Critic.
    Returns (final_action, risk_level, reward_estimate, verdict).
    """
    user_msg = textwrap.dedent(f"""
        === CURRENT FARM STATE ===
        {obs_text}

        === ACTOR'S PROPOSED ACTION ===
        {json.dumps(proposed_action, indent=2)}

        === ACTOR'S REASONING ===
        {actor_thought}

        Evaluate the risk and reward of this action. Respond with JSON only.
    """).strip()

    try:
        resp = llm.chat.completions.create(
            model=CRITIC_MODEL,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=400,
        )
        raw_text = resp.choices[0].message.content or ""
        parsed = extract_json(raw_text)
        if parsed:
            verdict         = parsed.get("verdict", "approve")
            risk_level      = parsed.get("risk_level", "unknown")
            reward_estimate = parsed.get("reward_estimate", "unknown")
            risk_reason     = parsed.get("risk_reason", "—")
            reward_reason   = parsed.get("reward_reason", "—")
            override_reason = parsed.get("override_reason", "")

            if verdict == "override":
                final_action = clean_action(parsed.get("final_action", {}))
                diag(f"    [CRITIC] ⚠️  OVERRIDE → {final_action['action_type']} | {override_reason}")
            else:
                final_action = proposed_action
                verdict = "approve"

            return final_action, risk_level, reward_estimate, verdict
    except Exception as e:
        diag(f"    [CRITIC ERROR] {e}")

    # Fallback: approve the actor's action
    return proposed_action, "unknown", "unknown", "approve"

# ─── Episode runner ───────────────────────────────────────────────────────────

def run_episode(
    env:    FarmClient,
    actor_llm:  OpenAI,
    critic_llm: OpenAI,
    task_id: int,
    episode: int,
) -> Dict[str, Any]:
    diag(f"\n  ── Episode {episode} | Task {task_id} ──────────────────────────")

    obs = env.reset(task_id)
    log_start(task_id, ACTOR_MODEL, CRITIC_MODEL)

    history:      List[str]   = []
    rewards_list: List[float] = []
    total_reward  = 0.0
    steps         = 0
    done          = obs.get("done", False)

    # Stats tracking
    actor_approvals  = 0
    critic_overrides = 0

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        obs_text = obs.get("text_summary", str(obs))
        money    = obs.get("money", 0)
        day      = obs.get("day", step)

        diag(f"\n  Step {step:02d} | Day {day} | Money ${money:.2f}")

        # ── Model 1: Actor picks action ──────────────────────────────────
        proposed_action, actor_thought = actor_choose_action(
            actor_llm, obs_text, history
        )
        diag(f"    [ACTOR]  → {proposed_action['action_type']} | {actor_thought[:80]}")

        # ── Model 2: Critic evaluates risk/reward ────────────────────────
        final_action, risk_level, reward_est, verdict = critic_evaluate(
            critic_llm, obs_text, proposed_action, actor_thought
        )

        if verdict == "approve":
            actor_approvals += 1
            diag(f"    [CRITIC] ✅ APPROVED | risk={risk_level} | reward_est={reward_est}")
        else:
            critic_overrides += 1

        # ── Execute the final action in the environment ──────────────────
        try:
            result    = env.step(final_action)
            reward    = float(result.get("reward", 0.0))
            done      = result.get("done", False)
            error_msg = result.get("error")
            obs       = result
        except Exception as e:
            diag(f"    [ENV ERROR] {e}")
            reward    = 0.0
            error_msg = str(e)

        total_reward += reward
        rewards_list.append(reward)
        steps += 1

        # Update history for Actor's context
        history.append(
            f"Step {step}: {final_action['action_type']} → reward={reward:+.2f} "
            f"[risk={risk_level}, critic={verdict}]"
        )

        log_step(step, final_action["action_type"], reward, done, verdict, risk_level)

        if error_msg:
            diag(f"    [ENV]    error: {error_msg}")

    # ── Final grade ──────────────────────────────────────────────────────
    grade       = float(obs.get("metadata", {}).get("grade", 0.0))
    final_money = float(obs.get("money", 0.0))
    success     = grade >= 0.5

    diag(f"\n  ── Result ──────────────────────────────────────────────────────")
    diag(f"  steps={steps} | total_reward={total_reward:+.3f} | money=${final_money:.2f} | grade={grade:.4f}")
    diag(f"  actor_approvals={actor_approvals} | critic_overrides={critic_overrides}")
    diag(f"  override_rate={critic_overrides/(steps or 1)*100:.1f}%")

    log_end(success=success, steps=steps, score=grade, rewards=rewards_list)

    return {
        "task_id":          task_id,
        "episode":          episode,
        "grade":            grade,
        "total_reward":     round(total_reward, 4),
        "steps":            steps,
        "final_money":      final_money,
        "actor_approvals":  actor_approvals,
        "critic_overrides": critic_overrides,
        "override_rate":    round(critic_overrides / (steps or 1), 4),
    }

# ─── Task runner ─────────────────────────────────────────────────────────────

def run_task(
    env:        FarmClient,
    actor_llm:  OpenAI,
    critic_llm: OpenAI,
    task_id:    int,
) -> Dict[str, Any]:
    labels = {1: "easy", 2: "medium", 3: "hard"}
    diag(f"\n{'=' * 56}")
    diag(f"  TASK {task_id} — {labels.get(task_id, '?').upper()}")
    diag(f"{'=' * 56}")

    result = run_episode(env, actor_llm, critic_llm, task_id, episode=1)

    diag(f"\n  Task {task_id} grade: {result['grade']:.4f}")
    return {
        "task_id":    task_id,
        "difficulty": labels.get(task_id, "?"),
        "grade":      result["grade"],
        "episode":    result,
    }

# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    diag("=" * 56)
    diag("  Farming RL — Dual-LLM Inference (Actor + Critic)")
    diag("=" * 56)
    diag(f"  actor model  : {ACTOR_MODEL}")
    diag(f"  critic model : {CRITIC_MODEL}")
    diag(f"  ollama url   : {OLLAMA_URL}")
    diag(f"  env url      : {ENV_URL}")
    diag(f"  max steps    : {MAX_STEPS}")
    diag("=" * 56)

    # ── Connect to farm server ───────────────────────────────────────────
    env = FarmClient(ENV_URL)
    if not env.health():
        diag(f"\n[ERROR] Cannot reach farm server at {ENV_URL}")
        diag("  Start it with:")
        diag("    cd server && uvicorn app:app --host 0.0.0.0 --port 7860")
        sys.exit(1)
    diag(f"\n  farm server  : OK")

    # ── Connect to Ollama (both models share the same base URL) ──────────
    # Ollama's OpenAI-compatible endpoint — no API key needed
    actor_llm  = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    critic_llm = OpenAI(base_url=OLLAMA_URL, api_key="ollama")

    # ── Verify models are available ──────────────────────────────────────
    diag(f"\n  Checking Ollama models...")
    try:
        models_resp = requests.get(
            OLLAMA_URL.replace("/v1", "") + "/api/tags", timeout=5
        )
        available = [m["name"] for m in models_resp.json().get("models", [])]
        diag(f"  Available models: {', '.join(available) or 'none found'}")

        for model_name in [ACTOR_MODEL, CRITIC_MODEL]:
            # Ollama model names can be "llama3.2" or "llama3.2:latest"
            found = any(model_name in m for m in available)
            status = "✅" if found else "⚠️  (not found — will try anyway)"
            diag(f"  {model_name}: {status}")
    except Exception as e:
        diag(f"  ⚠️  Could not check models: {e} — will try anyway")

    # ── Determine which tasks to run ─────────────────────────────────────
    single_task = os.getenv("FARMING_TASK_ID")
    task_ids    = [int(single_task)] if single_task else [1, 2, 3]

    # ── Run tasks ────────────────────────────────────────────────────────
    task_results = [
        run_task(env, actor_llm, critic_llm, tid)
        for tid in task_ids
    ]

    # ── Summary ──────────────────────────────────────────────────────────
    diag(f"\n{'=' * 56}")
    diag("  DUAL-LLM RESULTS SUMMARY")
    diag(f"{'=' * 56}")
    diag(f"  {'Task':<8} {'Difficulty':<10} {'Grade':<8} {'Overrides'}")
    diag(f"  {'-'*50}")
    for t in task_results:
        ep = t["episode"]
        override_pct = ep["override_rate"] * 100
        diag(
            f"  Task {t['task_id']:<4} {t['difficulty']:<10} "
            f"{t['grade']:<8.4f} {override_pct:.1f}%"
        )

    overall = sum(t["grade"] for t in task_results) / len(task_results)
    diag(f"\n  Overall avg grade : {overall:.4f}")
    diag(f"{'=' * 56}\n")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "actor_model":  ACTOR_MODEL,
        "critic_model": CRITIC_MODEL,
        "ollama_url":   OLLAMA_URL,
        "tasks":        task_results,
        "overall_avg":  round(overall, 4),
    }
    out_file = "dual_llm_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    diag(f"  Saved → {out_file}")

if __name__ == "__main__":
    main()
