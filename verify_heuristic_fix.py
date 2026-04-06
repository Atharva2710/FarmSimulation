
import sys
import os
import json
import time

# Ensure we can import from server
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'server')))

from server.farming_environment import FarmingEnvironment
from server.agents.heuristic import HeuristicAgent

def run_test_episode(task_id=3, max_days=30):
    env = FarmingEnvironment()
    env.reset(task_id=task_id)
    agent = HeuristicAgent()
    
    print(f"--- STARTING HEURISTIC FIX VERIFICATION (TASK {task_id}) ---")
    withered_at_start = 0
    
    for day in range(1, max_days + 1):
        obs = env.get_observation()
        if obs.done:
            break
            
        action, thought = agent.act(obs)
        
        # Track withering
        current_withered = sum(1 for p in obs.plots if p.stage == "withered")
        
        print(f"Day {obs.day} | Money: ${obs.money:.2f} | Tank: {obs.water_tank*100:.0f}% | Action: {action['action_type']} | Reasoning: {thought}")
        
        env.step(action)
        
    final_obs = env.get_observation()
    metadata = env.get_metadata()
    
    print("\n--- TEST RESULTS ---")
    print(f"Total Days: {metadata['day']}")
    print(f"Final Money: ${obs.money:.2f}")
    print(f"Withered Crops: {metadata['withered_count']}")
    
    if metadata['withered_count'] == 0:
        print("✅ SUCCESS: No crops withered during the drought simulation!")
    elif metadata['withered_count'] < 3:
        print("⚠️ PARTIAL SUCCESS: Managed to keep most crops alive, but some were lost.")
    else:
        print("❌ FAILURE: Multiple crops withered despite the fix.")

if __name__ == "__main__":
    run_test_episode(task_id=3)
