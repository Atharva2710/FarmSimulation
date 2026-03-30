import random
from models import FarmAction
from server.farming_environment import FarmingEnvironment

def run_random_agent(task_id: int):
    print(f"\n--- Running Task {task_id} Simulation ---")
    env = FarmingEnvironment(task_id=task_id)
    obs = env.reset(seed=42)
    done = False
    step = 0
    total_reward = 0.0

    while not done:
        # Create a valid pool of random actions
        actions = [
            FarmAction(action_type="wait"),
        ]
        
        # We can try buying if we have money
        if obs.money > 10:
            actions.append(FarmAction(action_type="buy_seeds", seed_type="wheat", quantity=1))
        
        # We can try planting if we have seeds
        if obs.seed_inventory.get("wheat", 0) > 0:
            # find first empty plot
            for p in obs.plots:
                if p.stage == "empty":
                    actions.append(FarmAction(action_type="plant", plot_id=p.plot_id, seed_type="wheat"))
                    break
        
        # We can try irrigating or harvesting
        for p in obs.plots:
            if p.stage not in ("empty", "withered"):
                actions.append(FarmAction(action_type="irrigate", plot_id=p.plot_id))
            if p.stage == "mature":
                actions.append(FarmAction(action_type="harvest", plot_id=p.plot_id))
        
        # We can try selling
        if sum(obs.storage.values()) > 0:
            for k, v in obs.storage.items():
                if int(v) > 0:
                    actions.append(FarmAction(action_type="sell", seed_type=k, quantity=int(v)))
                    break

        action = random.choice(actions)
        obs = env.step(action)
        
        if obs.reward is not None:
            total_reward += obs.reward
        
        step += 1
        done = obs.done
        
        # Failsafe infinite loop break
        if step > 100:
            print("ERROR: Simulation exceeded maximum expected steps.")
            break

    print(f"Task {task_id} finished after {step} days (Target: {env._max_days}).")
    print(f"Final Money: ${obs.money:.2f}")
    print(f"Total Accumulated Reward: {total_reward:.2f}")
    assert obs.done, f"Task {task_id} did not finish properly."

if __name__ == "__main__":
    import subprocess
    import sys

    try:
        print("Running test_phase2.py...")
        subprocess.run([sys.executable, "test_phase2.py"], check=True)
        
        print("\nRunning test_phase3.py...")
        subprocess.run([sys.executable, "test_phase3.py"], check=True)
        
        print("\nRunning end-to-end full simulations with random agent...")
        run_random_agent(1)
        run_random_agent(2)
        run_random_agent(3)
        
        print("\n✅ Verification complete: The Farming Simulation environment is fully stable!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Verification failed during script execution: {e}")
    except AssertionError as e:
        print(f"\n❌ Verification failed during simulation: {e}")
