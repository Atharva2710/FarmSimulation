from server.farming_environment import FarmingEnvironment

def test_environment_logic():
    env = FarmingEnvironment(task_id=3)
    env.reset()
    
    print("\n[V] Starting logic verification...")

    # 1. Setup Plot 0 correctly
    p = env._plots[0]
    p.crop_type = "wheat"
    p.stage = "growing"
    p.health = 1.0
    p.soil_moisture = 0.95 # Trigger root rot soon
    p.nitrogen = 1.0
    p.phosphorus = 1.0
    p.potassium = 1.0

    # 2. Test Wasteful Irrigation Reward
    obs = env.step({"action_type": "irrigate", "plot_id": 0})
    print(f" -> Wasteful Irrigation Reward: {obs.reward} (Expected -0.5 approx)")

    # 3. Test Rescue Irrigation
    p.soil_moisture = 0.1
    obs = env.step({"action_type": "irrigate", "plot_id": 0})
    print(f" -> Rescue Irrigation Reward: {obs.reward} (Expected +0.5 approx)")

    # 4. Test Biological Damage (Root Rot)
    p.soil_moisture = 0.95
    p.health = 1.0
    env.step({"action_type": "wait"}) # advances 1 day
    print(f" -> Root Rot Health: {p.health:.2f} (Expected < 1.0)")

    # 5. Test Market Elasticity
    start_price = env._market_prices["wheat"].sell_price
    env._storage["wheat"] = 100.0
    env.step({"action_type": "sell", "seed_type": "wheat", "quantity": 50})
    end_price = env._market_prices["wheat"].sell_price
    print(f" -> Price before sale: {start_price:.2f} | after: {end_price:.2f} (Should be lower)")

    # 6. Test Storage Overflow Penalty
    env._storage = {"wheat": 195.0} # Capacity 200
    p.stage = "mature"
    p.health = 1.0 # 10kg yield
    obs = env.step({"action_type": "harvest", "plot_id": 0})
    print(f" -> Storage Overflow Reward: {obs.reward} (Expected negative/low via penalty)")

if __name__ == "__main__":
    test_environment_logic()
