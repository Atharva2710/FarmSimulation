from server.farming_environment import FarmingEnvironment
from models import FarmAction

def make_env(task_id=1):
    env = FarmingEnvironment(task_id=task_id)
    env.reset(seed=42)
    return env

def test_invalid_actions():
    env = make_env()
    # unknown action type
    obs = env.step(FarmAction(action_type="unknown_action"))
    assert obs.reward is not None

    # plant on occupied plot
    env2 = make_env()
    env2._seed_inventory["wheat"] = 5
    env2.step(FarmAction(action_type="plant", plot_id=0, seed_type="wheat"))
    obs = env2.step(FarmAction(action_type="plant", plot_id=0, seed_type="wheat"))
    assert obs.reward == -0.9, f"expected -0.9 got {obs.reward}"

    # buy with no money
    env3 = make_env()
    env3._money = 0.0
    obs = env3.step(FarmAction(action_type="buy_seeds", seed_type="corn", quantity=100))
    assert obs.reward == -1.0
    print("invalid_actions() OK")

def test_full_crop_cycle():
    env = make_env()
    rewards = []

    # buy wheat seeds
    obs = env.step(FarmAction(action_type="buy_seeds", seed_type="wheat", quantity=2))
    rewards.append(obs.reward)
    assert env._seed_inventory["wheat"] == 2

    # plant on plot 0
    obs = env.step(FarmAction(action_type="plant", plot_id=0, seed_type="wheat"))
    rewards.append(obs.reward)
    assert env._plots[0].stage == "seedling"  # after planting, before day advance

    # wait for wheat to grow (7 days grow time, already 2 steps in)
    for _ in range(7):
        obs = env.step(FarmAction(action_type="wait"))
        rewards.append(obs.reward)

    # should be mature now
    assert env._plots[0].stage == "mature", f"expected mature got {env._plots[0].stage}"

    # harvest
    obs = env.step(FarmAction(action_type="harvest", plot_id=0))
    rewards.append(obs.reward)
    assert obs.reward > 0, "harvest should give positive reward"
    assert env._storage["wheat"] > 0
    assert env._plots[0].stage == "empty"

    # sell
    stored = env._storage["wheat"]
    obs = env.step(FarmAction(action_type="sell", seed_type="wheat", quantity=int(stored)))
    rewards.append(obs.reward)
    assert obs.reward > 0
    assert env._money > 200.0, f"should have profited, got {env._money}"

    print(f"full_crop_cycle() OK — final money: ${env._money:.2f}, rewards: {[round(r,2) for r in rewards]}")

def test_irrigation_wasteful():
    env = make_env()
    env._seed_inventory["rice"] = 1
    env.step(FarmAction(action_type="plant", plot_id=1, seed_type="rice"))

    # over-irrigate
    env._plots[1].soil_moisture = 0.95
    obs = env.step(FarmAction(action_type="irrigate", plot_id=1))
    assert obs.reward < 0, f"wasteful irrigation should penalise, got {obs.reward}"
    print("irrigation_wasteful() OK")

def test_withering_penalty():
    env = make_env()
    env._seed_inventory["wheat"] = 1
    env.step(FarmAction(action_type="plant", plot_id=0, seed_type="wheat"))

    # advance past harvest window without harvesting (7 grow + 3 window + 1 = 11 days)
    for _ in range(13):
        env.step(FarmAction(action_type="wait"))

    assert env._plots[0].stage == "withered", f"expected withered got {env._plots[0].stage}"
    print("withering_penalty() OK")

def test_rewards_are_varied():
    env = make_env()
    import random
    random.seed(99)
    rewards = []
    for _ in range(20):
        action = random.choice([
            FarmAction(action_type="wait"),
            FarmAction(action_type="buy_seeds", seed_type="wheat", quantity=1),
            FarmAction(action_type="plant", plot_id=0, seed_type="wheat"),
        ])
        obs = env.step(action)
        rewards.append(obs.reward)
    unique = set(rewards)
    assert len(unique) > 1, "all rewards identical — reward function is broken"
    print(f"rewards_are_varied() OK — unique reward values: {len(unique)}")

def test_episode_ends():
    env = FarmingEnvironment(task_id=1)
    env.reset(seed=0)
    obs = None
    for _ in range(35):
        obs = env.step(FarmAction(action_type="wait"))
        if obs.done:
            break
    assert obs.done == True, "episode should have ended by day 30"
    print(f"episode_ends() OK — ended on day {env._day}")

if __name__ == "__main__":
    test_invalid_actions()
    test_full_crop_cycle()
    test_irrigation_wasteful()
    test_withering_penalty()
    test_rewards_are_varied()
    test_episode_ends()
    print("\nPhase 3 complete — all action and reward tests passed")
