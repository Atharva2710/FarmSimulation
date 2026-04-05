
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'server')))
from server.farming_environment import FarmingEnvironment, FarmAction
from models import PESTICIDE_COST, FERTILIZER_COST, IRRIGATION_COST, PUMP_COST, SEED_CONFIG

def test_difficulty_actions(task_id):
    print(f"\n--- Testing Task {task_id} ---")
    env = FarmingEnvironment()
    env.reset(task_id=task_id)
    cfg = env._task_config()
    mult = cfg["input_mult"]
    
    # 1. Spray Pesticide
    print(f"Testing Spray Pesticide (Mult: {mult})...")
    plot_id = 0
    env._plots[plot_id].has_pests = True
    env._plots[plot_id].pest_severity = 0.5
    initial_money = env._money
    
    action = FarmAction(action_type="spray_pesticide", plot_id=plot_id)
    obs = env.step(action)
    reward = obs.reward
    done = obs.done
    
    expected_cost = PESTICIDE_COST * mult
    actual_cost = initial_money - env._money
    
    success = not env._plots[plot_id].has_pests and env._plots[plot_id].pest_severity == 0.0
    print(f"  Cost: Expected {expected_cost}, Actual {actual_cost}")
    print(f"  Pests Removed: {success}")
    if not success or actual_cost != expected_cost:
        print(f"  FAILED: Spray Pesticide in Task {task_id}")

    # 2. Apply Fertilizer
    print(f"Testing Apply Fertilizer...")
    env._plots[plot_id].nitrogen = 0.1
    initial_money = env._money
    action = FarmAction(action_type="apply_fertilizer", plot_id=plot_id)
    env.step(action)
    expected_cost = FERTILIZER_COST * mult
    actual_cost = initial_money - env._money
    print(f"  Cost: Expected {expected_cost}, Actual {actual_cost}")
    print(f"  Nitrogen: {env._plots[plot_id].nitrogen}")

    # 3. Pull Weeds
    print(f"Testing Pull Weeds...")
    env._plots[plot_id].has_weeds = True
    action = FarmAction(action_type="pull_weeds", plot_id=plot_id)
    env.step(action)
    print(f"  Weeds Removed: {not env._plots[plot_id].has_weeds}")

    # 4. Pump Water
    print(f"Testing Pump Water...")
    env._aquifer = 100
    env._water_tank = 10
    initial_money = env._money
    action = FarmAction(action_type="pump_water")
    env.step(action)
    expected_cost = PUMP_COST * mult
    actual_cost = initial_money - env._money
    print(f"  Cost: Expected {expected_cost}, Actual {actual_cost}")
    print(f"  Water Tank: {env._water_tank}")

    # 5. Buy Seeds
    print(f"Testing Buy Seeds...")
    initial_money = env._money
    action = FarmAction(action_type="buy_seeds", seed_type="wheat", quantity=2)
    env.step(action)
    # Buy price is dynamic but mult applies
    print(f"  Seeds in Inv: {env._seed_inventory['wheat']}")
    print(f"  Money spent: {initial_money - env._money}")

    # 6. Plant
    print(f"Testing Plant...")
    action = FarmAction(action_type="plant", plot_id=1, seed_type="wheat")
    env.step(action)
    print(f"  Plot 1 Stage: {env._plots[1].stage}")

    # 7. Irrigate
    print(f"Testing Irrigate...")
    env._plots[1].soil_moisture = 0.2
    action = FarmAction(action_type="irrigate", plot_id=1)
    env.step(action)
    print(f"  Soil Moisture: {env._plots[1].soil_moisture}")

if __name__ == "__main__":
    for i in [1, 2, 3]:
        test_difficulty_actions(i)
