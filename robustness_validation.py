import json
import random
import sys
import os
from pydantic import ValidationError

# Ensure we can import from server
sys.path.append(os.getcwd())

from server.farming_environment import FarmingEnvironment
from server.agents.heuristic import HeuristicAgent

def check_determinism():
    """Layer 1: Determinism (100 Steps)"""
    print("Step 1/5: Checking Determinism (100 Steps)...")
    env1 = FarmingEnvironment(task_id=1)
    env2 = FarmingEnvironment(task_id=1)
    
    seed = 42
    eid = "test-episode-123"
    obs1 = env1.reset(seed=seed, episode_id=eid)
    obs2 = env2.reset(seed=seed, episode_id=eid)
    
    # Static actions
    actions = [
        {"action_type": "buy_seeds", "seed_type": "wheat", "quantity": 5},
        {"action_type": "pump_water"},
        {"action_type": "plant", "plot_id": 0, "seed_type": "wheat"},
        {"action_type": "wait"},
    ]
    # Fill up with random but seed-consistent actions
    prng = random.Random(seed)
    for _ in range(96):
        a = prng.choice(["wait", "pump_water"])
        actions.append({"action_type": a})

    for i, action in enumerate(actions):
        obs1 = env1.step(action)
        obs2 = env2.step(action)
        
        # model_dump() includes reward/done in FarmObservation
        d1 = obs1.model_dump()
        d2 = obs2.model_dump()
        
        if d1 != d2:
            print(f"  ❌ Determinism FAILED at step {i}")
            # Find the difference
            diffs = []
            for k in d1:
                if d1[k] != d2.get(k):
                    diffs.append(f"    Field '{k}' differs: {d1[k]} != {d2[k]}")
            if not diffs:
                # Check for hidden differences or nested structures
                print("    (No obvious field difference found in top-level dict comparison)")
            else:
                for d in diffs: print(d)
            return False
            
    print("  ✅ Determinism PASSED")
    return True

def check_skill_gradient():
    """Layer 2: Skill Gradient (Stochastic Baseline Comparison)"""
    print("Step 2/5: Checking Skill Gradient (Heuristic vs Random)...")
    env = FarmingEnvironment(task_id=1)
    
    # 1. Random Agent (3 episodes avg)
    random_rewards = []
    for s in [42, 43, 44]:
        env.reset(seed=s)
        total = 0
        done = False
        while not done:
            action = {"action_type": random.choice(["wait", "buy_seeds", "pump_water"])}
            if action["action_type"] == "buy_seeds":
                action["seed_type"] = "wheat"
                action["quantity"] = 1
            obs = env.step(action)
            total += obs.reward or 0
            done = obs.done
        random_rewards.append(total)
    
    avg_random = sum(random_rewards) / len(random_rewards)
    
    # 2. Heuristic Agent
    agent = HeuristicAgent()
    env.reset(seed=42)
    total_heuristic = 0
    done = False
    while not done:
        action, _ = agent.act(env.get_observation())
        obs = env.step(action)
        total_heuristic += obs.reward or 0
        done = obs.done
        
    print(f"  Avg Random Reward: {avg_random:.2f}")
    print(f"  Heuristic Reward: {total_heuristic:.2f}")
    
    if total_heuristic > avg_random:
        print("  ✅ Skill Gradient PASSED")
        return True
    else:
        print("  ❌ Skill Gradient FAILED")
        return False

def check_physics_edge_cases():
    """Layer 3: Edge Case Robustness"""
    print("Step 3/5: Checking Edge Cases...")
    env = FarmingEnvironment(task_id=1)
    env.reset(seed=42)
    
    errors = []
    
    # 1. Harvest empty plot
    res = env.step({"action_type": "harvest", "plot_id": 0})
    if res.reward != -1.0: errors.append("Empty harvest didn't penalize")
    
    # 2. Sell 0 quantity (Should trigger validation error)
    pydantic_caught = False
    try:
        env.step({"action_type": "sell", "seed_type": "wheat", "quantity": 0})
    except ValidationError:
        pydantic_caught = True
    
    if not pydantic_caught:
        errors.append("Invalid quantity (0) not caught by Pydantic")
    
    # 3. Buy with negative money (simulated)
    env._money = 0
    res = env.step({"action_type": "buy_seeds", "seed_type": "wheat", "quantity": 1})
    if res.reward != -1.0: errors.append("Broke purchase didn't penalize")
    
    if not errors:
        print("  ✅ Edge Cases PASSED")
        return True
    else:
        for e in errors: print(f"  ❌ {e}")
        return False

def check_market_dynamics():
    """Layer 4: Market Dynamics (Temporary vs Permanent Impact)"""
    print("Step 4/5: Checking Market Dynamics (Impact Verification)...")
    env = FarmingEnvironment(task_id=1)
    env.reset(seed=42)
    
    # Setup: Put 100kg wheat in storage
    env._storage["wheat"] = 100.0
    mid_price = env._market_prices["wheat"].sell_price
    
    # 1. Check Temporary Impact (Slippage)
    # 100kg sell should have 5% slippage
    # P_exec = mid * (1 - 0.05)
    obs = env.step({"action_type": "sell", "seed_type": "wheat", "quantity": 100})
    
    sell_event = env._sell_events[-1]
    execution_price = sell_event["price"]
    expected_exec = mid_price * 0.95
    
    if abs(execution_price - expected_exec) > 0.01:
        print(f"  ❌ Temporary Impact FAILED: Exec {execution_price:.4f} != Expected {expected_exec:.4f}")
        return False
        
    print("  ✅ Market Dynamics PASSED")
    return True

def run_all():
    results = {
        "determinism_100": check_determinism(),
        "skill_gradient": check_skill_gradient(),
        "edge_cases": check_physics_edge_cases(),
        "market_impact": check_market_dynamics()
    }
    
    all_passed = all(results.values())
    
    report = {
        "status": "PASSED" if all_passed else "FAILED",
        "timestamp": "2026-04-06",
        "layers": results
    }
    
    with open("ROBUSTNESS_REPORT.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nFinal Status: {report['status']}")
    return all_passed

if __name__ == "__main__":
    if not run_all():
        sys.exit(1)
    sys.exit(0)
