import sys
import os
sys.path.append(os.getcwd())

from server.farming_environment import FarmingEnvironment
from models import FarmAction

def test_manual_expansion():
    # Initial state: 4 plots, $1000 money
    env = FarmingEnvironment()
    print(f"Initial plots: {len(env._plots)}")
    print(f"Initial money: ${env._money}")

    # Buy a plot
    action = FarmAction(action_type="buy_plot")
    env.step(action)
    
    print(f"Plots after buy: {len(env._plots)}")
    print(f"Money after buy: ${env._money}")
    
    assert len(env._plots) == 5, "Should have 5 plots"
    assert env._money < 1000, "Money should have decreased"
    print("✅ Manual expansion test passed!")

def test_heuristic_expansion():
    # Setup environment where all plots are planted and money is high
    env = FarmingEnvironment()
    env._money = 200 # Above $150 threshold
    
    # Plant all 4 plots
    for i in range(4):
        env._plots[i].stage = "growing"
        env._plots[i].crop_type = "wheat"
    
    from server.agents.heuristic import HeuristicAgent
    agent = HeuristicAgent()
    
    # Observe and act
    obs = env.get_observation()
    action_dict, thought = agent.act(obs, mode="physics")
    
    print(f"Agent thought: {thought}")
    print(f"Agent action: {action_dict}")
    
    assert action_dict["action_type"] == "buy_plot", "Heuristic agent should buy plot when capital is high and plots are full"
    
    env.step(action_dict)
    print(f"Plots after heuristic buy: {len(env._plots)}")
    assert len(env._plots) == 5, "Should have 5 plots after heuristic action"
    print("✅ Heuristic expansion test passed!")

if __name__ == "__main__":
    test_manual_expansion()
    test_heuristic_expansion()
