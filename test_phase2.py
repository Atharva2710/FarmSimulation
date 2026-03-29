from server.farming_environment import FarmingEnvironment

def test_reset():
    env = FarmingEnvironment(task_id=1)
    obs = env.reset()

    assert obs.day == 0
    assert obs.money == 200.0
    assert abs(obs.water_tank - 0.8) < 0.01
    assert len(obs.plots) == 4
    assert all(p.stage == "empty" for p in obs.plots)
    assert obs.done == False
    assert obs.reward is None
    assert obs.climate is not None
    assert obs.climate.climate_type == "temperate"
    assert set(obs.seed_inventory.keys()) == {"wheat", "rice", "corn"}
    assert set(obs.storage.keys()) == {"wheat", "rice", "corn"}
    assert set(obs.market_prices.keys()) == {"wheat", "rice", "corn"}
    assert len(obs.text_summary) > 0
    assert len(obs.valid_actions) > 0
    print(f"reset() OK — day={obs.day}, money={obs.money}")
    print(obs.text_summary)

def test_day_advances():
    env = FarmingEnvironment(task_id=1)
    env.reset()
    for i in range(1, 6):
        obs = env.step(None)   # stub step, action ignored
        assert obs.day == i, f"expected day {i}, got {obs.day}"
    print(f"day_advances() OK — stepped to day {obs.day}")

def test_climate_rotation():
    env = FarmingEnvironment(task_id=1)
    env.reset()
    climates_seen = set()
    for _ in range(35):
        obs = env.step(None)
        climates_seen.add(obs.climate.climate_type)
    assert "temperate" in climates_seen
    assert "arid"      in climates_seen
    assert "tropical"  in climates_seen
    print(f"climate_rotation() OK — saw: {climates_seen}")

def test_market_prices_move():
    env = FarmingEnvironment(task_id=1)
    env.reset()
    initial_price = env._market_prices["wheat"].sell_price
    prices = []
    for _ in range(25):
        env.step(None)
        prices.append(env._market_prices["wheat"].sell_price)
    assert len(set(prices)) > 1, "prices never changed"
    print(f"market_prices_move() OK — wheat range: {min(prices):.2f}–{max(prices):.2f}")

def test_state_property():
    env = FarmingEnvironment(task_id=2)
    env.reset()
    state = env.state
    assert state.task_id == 2
    assert state.step_count == 0
    assert state.episode_id is not None
    env.step(None)
    state2 = env.state
    assert state2.step_count == 1
    print(f"state_property() OK — episode_id={state.episode_id}")

def test_task3_drought():
    env = FarmingEnvironment(task_id=3)
    env.reset()
    assert env._drought_active == True
    obs = env.reset()
    assert obs.money == 100.0
    print("task3_drought() OK")

if __name__ == "__main__":
    test_reset()
    test_day_advances()
    test_climate_rotation()
    test_market_prices_move()
    test_state_property()
    test_task3_drought()
    print("\nPhase 2 complete — all environment tests passed")
