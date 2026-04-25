
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from server.farming_environment import FarmingEnvironment, ACTION_LABOR_COSTS
from server.audit_utils import calculate_state_fidelity
from models import SEED_CONFIG, FarmObservation

MCP_RESERVED = {"reset", "step", "state", "close"}


def _build_perfect_summary(obs: FarmObservation) -> dict:
    """Construct a state_summary that perfectly matches the current obs."""
    best_roi_crop = "wheat"
    max_roi = 0.0
    for crop, cfg in SEED_CONFIG.items():
        market = obs.market_prices.get(crop)
        if market:
            roi = (market.sell_price * cfg["yield_kg"]) / cfg["grow_days"]
            if roi > max_roi:
                max_roi = roi
                best_roi_crop = crop

    best_trend_crop = "wheat"
    max_trend = -2.0
    for crop in SEED_CONFIG.keys():
        market = obs.market_prices.get(crop)
        if market and market.trend > max_trend:
            max_trend = market.trend
            best_trend_crop = crop

    actual_crit_id = -1
    min_moisture = 1.1
    for p in obs.plots:
        if p.stage not in ["empty", "withered"] and p.soil_moisture < min_moisture:
            min_moisture = p.soil_moisture
            actual_crit_id = p.plot_id

    return {
        "highest_roi_crop": best_roi_crop,
        "market_trend_best": best_trend_crop,
        "plot_needing_water": actual_crit_id,
    }


def test_phase1_audit():
    print("--- Phase 1 Audit Verification ---")

    env = FarmingEnvironment(task_id=1)
    obs = env.reset()

    print("\n[Step 1] Building perfect state_summary from actual obs...")
    perfect_summary = _build_perfect_summary(obs)
    print(f"  Summary: {perfect_summary}")

    print("\n[Step 2] Calling env.step() with perfect audit metadata...")
    # Use write_journal (0.1h labor) to avoid triggering an auto day-advance
    # which would update market prices and invalidate the summary.
    action = {"action_type": "write_journal"}
    env.step(action=action, thought="Observing fields.", state_summary=perfect_summary)

    history = env.get_metadata()["action_history"]
    last_event = history[-1]
    fidelity = last_event["action"].get("fidelity")

    if fidelity:
        print(f"Fidelity Report: {json.dumps(fidelity, indent=2)}")
        if fidelity.get("overall_fidelity", 0) >= 1.0:
            print("✅ SUCCESS: Perfect state fidelity confirmed!")
        else:
            print("❌ FAILURE: Fidelity score below 1.0.")
            sys.exit(1)
    else:
        print("❌ FAILURE: No fidelity report found in history.")
        sys.exit(1)

    print("\n[Step 3] MCP reserved-name collision check...")
    action_types = set(ACTION_LABOR_COSTS.keys())
    collisions = action_types & MCP_RESERVED
    if collisions:
        print(f"❌ FAILURE: Action types collide with MCP reserved names: {collisions}")
        sys.exit(1)
    else:
        print(f"  Action types: {sorted(action_types)}")
        print(f"✅ No MCP reserved-name collisions.")

    print("\n--- Verification Complete ---")


if __name__ == "__main__":
    test_phase1_audit()
