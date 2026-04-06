
import sys
import os
import json
from typing import Dict, Any

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from server.agents.hybrid import HybridAgent
from server.farming_environment import FarmingEnvironment
from models import FarmObservation

def test_phase1_audit():
    print("--- Phase 1 Audit Verification ---")
    
    # 1. Test Agent Perception Prompting
    agent = HybridAgent()
    env = FarmingEnvironment(task_id=1)
    obs = env.reset()
    
    print("\n[Step 1] Testing HybridAgent handle response with state_summary...")
    # Mocking a valid LLM response structure (simulating what LLM should return)
    mock_response = {
        "state_summary": {
            "money": 200.0,
            "water_tank": 0.8,
            "critical_plot_id": 0,
            "critical_plot_moisture": 0.5
        },
        "action": {"action_type": "plant", "plot_id": 0, "seed_type": "wheat"},
        "thought": "Planting wheat to start cycle."
    }
    
    # We can't easily mock the OpenAI call inside act() without patching,
    # but we can test if the environment.step() handles this dict correctly.
    
    print("\n[Step 2] Testing FarmingEnvironment.step() with audit metadata...")
    # env.step now accepts thought and state_summary
    obs_after = env.step(
        action=mock_response["action"],
        thought=mock_response["thought"],
        state_summary=mock_response["state_summary"]
    )
    
    # Check if history recorded the fidelity
    history = env.get_metadata()["action_history"]
    last_event = history[-1]
    
    print(f"Action: {last_event['action']['type']}")
    print(f"Thought: {last_event['action']['thought']}")
    
    obs = env.get_observation()
    min_moisture = 1.1
    actual_crit_id = -1
    for p in obs.plots:
        if p.stage not in ["empty", "withered"] and p.soil_moisture < min_moisture:
            min_moisture = p.soil_moisture
            actual_crit_id = p.plot_id
            
    print(f"Actual: Money=${obs.money}, Tank={obs.water_tank}, CritID={actual_crit_id}")
    print(f"Perceived: Money=${mock_response['state_summary']['money']}, Tank={mock_response['state_summary']['water_tank']}, CritID={mock_response['state_summary']['critical_plot_id']}")

    fidelity = last_event['action'].get('fidelity')
    if fidelity:
        print(f"Fidelity Report: {json.dumps(fidelity, indent=2)}")
        if fidelity.get("overall_fidelity", 0) > 0.9:
            print("✅ SUCCESS: State fidelity calculated correctly!")
        else:
            print("❌ FAILURE: Fidelity score low or missing.")
    else:
        print("❌ FAILURE: No fidelity report found in history.")

    # Check if money error is calculated (perceived 200, actual should be 200 - seed_cost is not yet deducted since we just planted)
    # Actually, in _handle_plant, money isn't deducted (seeds are from inventory).
    # Soil moisture error: perceived 0.5, actual 0.5.
    
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    test_phase1_audit()
