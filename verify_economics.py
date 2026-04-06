
import sys
import os
import json
import time

# Ensure we can import from server
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'server')))

from server.farming_environment import FarmingEnvironment
from server.agents.heuristic import HeuristicAgent

def run_economic_test(task_id=2, max_days=45):
    env = FarmingEnvironment()
    env.reset(task_id=task_id)
    agent = HeuristicAgent() # We use Heuristic as a baseline, but the logic is now smarter
    
    print(f"--- STARTING ECONOMIC AUDIT (TASK {task_id}) ---")
    
    total_revenue = 0.0
    premium_revenue = 0.0
    crop_counts = {"wheat": 0, "rice": 0, "corn": 0}
    
    for day in range(1, max_days + 1):
        obs = env.get_observation()
        if obs.done:
            break
            
        action, thought = agent.act(obs)
        
        # Track planting choices
        if action["action_type"] == "plant":
            crop_counts[action["seed_type"]] += 1
            
        # Track sales
        if action["action_type"] == "sell":
            crop = action["seed_type"]
            qty = action["quantity"]
            price = obs.market_prices[crop].sell_price
            from models import SEED_CONFIG
            base_price = SEED_CONFIG[crop]["base_sell"]
            
            rev = price * qty
            total_revenue += rev
            if price > base_price:
                premium_revenue += (price - base_price) * qty
        
        env.step(action)
        
    metadata = env.get_metadata()
    
    print("\n--- ECONOMIC RESULTS ---")
    print(f"Total Revenue: ${total_revenue:.2f}")
    print(f"Premium Scored: ${premium_revenue:.2f}")
    print(f"Crop Distribution: {crop_counts}")
    print(f"Final Grade: {metadata['last_grade']}")
    
    # In Task 2, we want to see Corn/Rice priority and Premium Revenue > 0
    if premium_revenue > 0:
        print("✅ SUCCESS: Agent captured market premiums!")
    else:
        print("⚠️ WARNING: Agent sold at or below base prices. ROI low.")
        
    if crop_counts["corn"] > crop_counts["wheat"]:
        print("✅ SUCCESS: Agent prioritized High-ROI Corn.")
    else:
        print("⚠️ WARNING: Agent defaulted to low-value Wheat.")

if __name__ == "__main__":
    run_economic_test()
