
import sys
import os
from typing import Dict, Any

# Mocking parts of the system for standalone testing
class MockPlot:
    def __init__(self, pid, moisture, stage="growing"):
        self.plot_id = pid
        self.soil_moisture = moisture
        self.stage = stage
        self.days_mature = 0
        self.days_critical = 0

class MockMarket:
    def __init__(self, price, trend):
        self.sell_price = price
        self.trend = trend

class MockObs:
    def __init__(self):
        self.money = 100.0
        self.water_tank = 0.5
        self.plots = [MockPlot(0, 0.2), MockPlot(1, 0.8)]
        self.market_prices = {
            "wheat": MockMarket(8.0, 0.1),
            "corn": MockMarket(20.0, -0.5)
        }

# Import the actual utility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'server')))
from server.audit_utils import calculate_state_fidelity

def test_audit_fidelity():
    obs = MockObs()
    
    # 1. Perfect Understanding
    perfect_summary = {
        "highest_roi_crop": "corn", # (20*35)/18 = 38.8 vs (8*10)/7 = 11.4
        "plot_needing_water": 0,    # 0.2 < 0.8
        "market_trend_best": "wheat" # 0.1 > -0.5
    }
    
    report = calculate_state_fidelity(perfect_summary, obs)
    print(f"Perfect Report: {report}")
    assert report["overall_fidelity"] == 1.0
    assert not report["hallucination_detected"]
    
    # 2. Hallucination (Wrong ROI, Wrong Plot)
    bad_summary = {
        "highest_roi_crop": "wheat",
        "plot_needing_water": 1,
        "market_trend_best": "corn"
    }
    report = calculate_state_fidelity(bad_summary, obs)
    print(f"Hallucination Report: {report}")
    assert report["overall_fidelity"] == 0.0
    assert report["hallucination_detected"]
    
    print("\n✅ AUDIT SYSTEM VERIFIED!")

if __name__ == "__main__":
    test_audit_fidelity()
