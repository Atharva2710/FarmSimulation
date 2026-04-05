
import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'server')))
from server.farming_environment import FarmingEnvironment, FarmAction

def run_random_episode(task_id):
    env = FarmingEnvironment()
    env.reset(task_id=task_id)
    print(f"Running Random Episode for Task {task_id}...")
    
    for day in range(60):
        obs = env.get_observation()
        if obs.done:
            break
            
        # Pick a random valid action
        valid = obs.valid_actions
        if not valid:
            act_type = "wait"
        else:
            act_type = random.choice(valid)
            
        action = {"action_type": act_type}
        if act_type in ["plant", "irrigate", "harvest", "clear", "apply_fertilizer", "spray_pesticide", "pull_weeds"]:
            action["plot_id"] = random.randint(0, 3)
        if act_type == "buy_seeds":
            action["seed_type"] = random.choice(["wheat", "rice", "corn"])
            action["quantity"] = random.randint(1, 5)
        if act_type == "sell":
            # Just sell some wheat if we have it
            action["seed_type"] = "wheat"
            action["quantity"] = 1
        if act_type == "plant":
            action["seed_type"] = random.choice(["wheat", "rice", "corn"])
            
        try:
            env.step(action)
        except Exception as e:
            print(f"CRASH in Task {task_id} on day {day} with action {act_type}: {e}")
            return False
            
    print(f"Task {task_id} completed successfully. Final Money: {env._money:.2f}")
    return True

if __name__ == "__main__":
    for i in [1, 2, 3]:
        run_random_episode(i)
