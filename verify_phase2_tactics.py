
import sys
import os
import json

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from server.farming_environment import FarmingEnvironment
from models import FarmAction, SEED_CONFIG

def test_phase2_tactics():
    print("--- Phase 2 Tactics Verification ---")
    
    env = FarmingEnvironment(task_id=1)
    env.reset()
    
    # 1. Plant a crop
    print("\n[Step 1] Planting wheat on Plot 0...")
    env._seed_inventory["wheat"] = 1
    env.step(FarmAction(action_type="plant", plot_id=0, seed_type="wheat"))
    
    plot = env._plots[0]
    print(f"Plot 0 Stage: {plot.stage}, Days Planted: {plot.days_planted}")
    
    # 2. Advance days to maturity
    grow_days = SEED_CONFIG["wheat"]["grow_days"]
    print(f"\n[Step 2] Advancing {grow_days + 1} days to reach maturity...")
    for _ in range(grow_days + 1):
        env.step(FarmAction(action_type="wait"))
    
    plot = env._plots[0]
    print(f"Plot 0 Stage: {plot.stage}, Days Planted: {plot.days_planted}, Days Mature: {plot.days_mature}")
    
    if plot.days_mature > 0:
        print("✅ SUCCESS: maturity_latency (days_mature) is tracking!")
    else:
        print("❌ FAILURE: days_mature did not increment.")

    # 3. Test Critical State Tracking
    print("\n[Step 3] Manually lowering health to test critical state...")
    plot.health = 0.1
    env.step(FarmAction(action_type="wait"))
    print(f"Plot 0 Health: {plot.health}, Days Critical: {plot.days_critical}")
    
    if plot.days_critical > 0:
        print("✅ SUCCESS: critical_state exposure (days_critical) is tracking!")
    else:
        print("❌ FAILURE: days_critical did not increment.")

    # 4. Test Audit Report Integration
    history = env.get_metadata()["action_history"]
    last_event = history[-1]
    tactical = last_event["action"].get("tactical")
    
    if tactical:
        print(f"\n[Step 4] Tactical Audit Report: {json.dumps(tactical, indent=2)}")
        if tactical["tactical_score"] < 1.0:
            print("✅ SUCCESS: Tactical Score reflected the neglect/danger!")
    else:
        print("❌ FAILURE: No tactical audit found in history.")

    # 5. Test Reset on Harvest
    print("\n[Step 5] Harvesting to verify reset...")
    env.step(FarmAction(action_type="harvest", plot_id=0))
    plot = env._plots[0]
    print(f"Post-Harvest Plot 0: Stage={plot.stage}, Days Mature={plot.days_mature}, Days Critical={plot.days_critical}")
    
    if plot.days_mature == 0 and plot.days_critical == 0:
        print("✅ SUCCESS: Counters reset correctly!")
    else:
        print("❌ FAILURE: Counters did NOT reset.")

    print("\n--- Phase 2 Verification Complete ---")

if __name__ == "__main__":
    test_phase2_tactics()
