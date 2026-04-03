from server.farming_environment import FarmingEnvironment
env = FarmingEnvironment()
# 1. plant wheat on plot 0
env.step({"action_type": "plant", "plot_id": 0, "seed_type": "wheat"})
print("Planted. weeds:", env._plots[0].has_weeds, "NPK:", env._plots[0].nitrogen)
# 2. advance day so it drains
for _ in range(5):
    env.step({"action_type": "wait"})
print("After waiting. weeds:", env._plots[0].has_weeds, "NPK:", env._plots[0].nitrogen)
# 3. Pull weeds and fertilize
env._plots[0].has_weeds = True # force weeds
env.step({"action_type": "pull_weeds", "plot_id": 0})
print("After pulling weeds:", env._plots[0].has_weeds)
env.step({"action_type": "apply_fertilizer", "plot_id": 0})
print("After fertilize:", env._plots[0].nitrogen)
