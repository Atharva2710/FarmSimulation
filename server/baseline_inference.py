import os
import sys
import time
import uuid
from typing import Dict, Any, Optional

# Ensure project root and server directory are in path
# and models.py (at root) can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI
from server.farming_environment import FarmingEnvironment
from server.agents.hybrid import HybridAgent

def run_baseline_eval():
    """
    Baseline inference script for Farming Simulation.
    Evaluates the HybridAgent across all 3 tasks (Easy, Medium, Hard).
    Uses OPENAI_API_KEY for inference.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please run: export OPENAI_API_KEY='your-key-here'")
        return

    # Initialize the agent
    # We use HybridAgent since it contains the baseline prompt and strategy logic
    agent = HybridAgent()
    
    tasks = [1, 2, 3]
    results = {}

    print("\n" + "="*50)
    print("🌾 FARMING SIMULATION: BASELINE EVALUATION")
    print("="*50)
    print(f"📅 Run Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Agent: {agent.name}")
    print("-" * 50)

    for task_id in tasks:
        difficulty = ["", "EASY", "MEDIUM", "HARD"][task_id]
        print(f"▶️  Testing Task {task_id} ({difficulty})...")
        
        # Instantiate environment for the specific task
        env = FarmingEnvironment(task_id=task_id)
        
        # Reset with fixed seed for reproducibility
        obs = env.reset(seed=42)
        
        done = False
        step_count = 0
        total_reward = 0.0
        
        # Episode Loop
        while not done:
            # HybridAgent.act uses LLM for strategy when api_token is provided
            # It falls back to Heuristic if API fails
            act_result = agent.act(obs, api_token=api_key)
            action = act_result["action"]
            thought = act_result.get("thought", "...")
            
            # --- ACTION SANITATION ---
            # LLMs sometimes return quantity: 0 or include it in actions that don't need it.
            # We clean it here to satisfy Pydantic validation in models.py.
            if isinstance(action, dict):
                if "quantity" in action:
                    val = action.get("quantity")
                    if val is None or (isinstance(val, (int, float)) and val <= 0):
                        if action.get("action_type") in ["buy_seeds", "sell"]:
                            action["quantity"] = 1
                        else:
                            del action["quantity"]
            
            # Step the environment
            try:
                obs = env.step(action)
            except Exception as e:
                print(f"\n❌ EXECUTION CRASH on Day {obs.day}!")
                print(f"   Action: {action}")
                print(f"   Error: {str(e)}")
                raise e
            
            # --- ACTION HISTORY PRINTING ---
            act_type = action.get("action_type") if isinstance(action, dict) else str(action)
            reward_str = f"{obs.reward:+.2f}" if obs.reward != 0 else " 0.00"
            print(f"   [Step {step_count+1:3d} | Day {obs.day:2d}] {act_type:<10} | Reward: {reward_str} | Money: ${obs.money:7.2f}")
            print(f"     💭 Thought: {thought}")

            done = obs.done
            total_reward += (obs.reward or 0.0)
            step_count += 1
            
            # Script-side safety check for environment bugs (Auto-shift overrun)
            if obs.day > env._max_days + 1:
                print(f"   🛑 FORCED END: Environment failed to terminate at Day {env._max_days}")
                done = True

            if step_count > 1000: # Safety break
                print("   ⚠️ Safety timeout triggered.")
                break

        # Collect final metrics from environment metadata
        metadata = env.get_metadata()
        score = metadata.get("last_grade", 0.0)
        results[task_id] = {
            "score": score,
            "reward": total_reward,
            "days": obs.day,
            "actions": step_count,
            "money": obs.money
        }
        
        print(f"✅ COMPLETED | Score: {score:.4f} | Final Money: ${obs.money:.2f}")
        print("-" * 50)

    # Final Summary Report
    print("\n📊 EVALUATION SUMMARY")
    print("=" * 50)
    for tid in tasks:
        diff = ["", "Easy", "Medium", "Hard"][tid]
        res = results[tid]
        print(f"Task {tid} ({diff}): {res['score']:.4f}  (Actions: {res['actions']}, Days: {res['days']})")
    
    avg_score = sum(r["score"] for r in results.values()) / len(tasks)
    print("-" * 50)
    print(f"OVERALL BASELINE SCORE: {avg_score:.4f}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_baseline_eval()
