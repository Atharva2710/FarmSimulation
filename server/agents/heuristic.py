
from typing import Dict, Any, Tuple
import random
import sys
import os

# We need access to models and environment constants
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models import FarmObservation, SEED_CONFIG, PUMP_COST, FERTILIZER_COST, PESTICIDE_COST

class HeuristicAgent:
    """
    A rule-based agent that follows a priority 'Triage' system:
    Safety > Resources > Growth > Profit
    """
    
    def __init__(self):
        self.name = "Heuristic-Alpha"

    def act(self, obs: FarmObservation) -> Tuple[Dict[str, Any], str]:
        """
        High-Intelligence Heuristic: ROI-Optimized Triage
        """
        plots = obs.plots
        money = obs.money
        tank = obs.water_tank
        inv = obs.seed_inventory
        storage = obs.storage
        climate_type = getattr(obs.climate, "climate_type", "Temperate").lower()
        
        # ── 1. EMERGENCY MAINTENANCE (Level 10) ──────────────────────────────
        
        # Priority 1: Clear Withered (Space is critical)
        for p in plots:
            if p.stage == "withered":
                return {"action_type": "clear", "plot_id": p.plot_id}, f"Clearing Plot {p.plot_id}: Removing failed crop to expand."

        # Priority 2: Spray Pests (Immediate threat)
        for p in plots:
            if p.has_pests and money >= PESTICIDE_COST:
                return {"action_type": "spray_pesticide", "plot_id": p.plot_id}, f"Spraying Plot {p.plot_id}: Pest detected. Activating residual protection."

        # Priority 3: Weeding (Nutrient competition)
        for p in plots:
            if p.has_weeds:
                return {"action_type": "pull_weeds", "plot_id": p.plot_id}, f"Weeding Plot {p.plot_id}: Removing competition for nutrients."

        # Priority 4: Immediate Harvest (Profit Realization)
        for p in plots:
            if p.stage == "mature":
                return {"action_type": "harvest", "plot_id": p.plot_id}, f"Harvesting Plot {p.plot_id}: Maturity reached. Maximizing storage."

        # ── 2. RESOURCE MAINTENANCE (Level 8) ─────────────────────────────
        
        # Climate-Aware Irrigation Thresholds
        moisture_threshold = 0.55
        if climate_type == "arid": moisture_threshold = 0.65
        elif climate_type == "tropical": moisture_threshold = 0.60
        
        # Critical Irrigation (Below threshold)
        for p in plots:
            if p.stage not in ["empty", "withered"] and p.soil_moisture < moisture_threshold:
                if tank >= 15.0:
                    return {"action_type": "irrigate", "plot_id": p.plot_id}, f"Irrigating Plot {p.plot_id}: Climate ({climate_type}) requires {moisture_threshold*100:.0f}% moisture."
                elif money >= PUMP_COST and getattr(obs, 'aquifer', 1.0) > 0:
                    return {"action_type": "pump_water"}, "Emergency Pump: Tank low during critical irrigation cycle."

        # Fertilize (Ensure health stays near 1.0)
        for p in plots:
            if p.stage in ["seedling", "growing"] and money >= FERTILIZER_COST:
                if p.health < 0.9 or p.nitrogen < 0.4:
                    return {"action_type": "apply_fertilizer", "plot_id": p.plot_id}, f"Fertilizing Plot {p.plot_id}: Boosting crop health ({p.health*100:.0f}%)."

        # ── 3. STRATEGIC EXPANSION & PUMPING (Level 5) ────────────────────────
        
        # Pre-emptive Pumping (If nothing urgent and tank < 80%)
        if tank < 0.8 and money >= PUMP_COST and getattr(obs, 'aquifer', 1.0) > 0:
            # Only pump if we have at least $40 left for emergency pesticides/fertilizer
            if money >= PUMP_COST + 40:
                return {"action_type": "pump_water"}, "Refilling Tank: Environment stable, building water reserve."

        # Expansion
        empty_plots = [p for p in plots if p.stage == "empty"]
        if empty_plots:
            # ROI Decision: CORN > RICE > WHEAT
            target_seed = "wheat"
            if money > 60.0: target_seed = "corn"
            elif money > 40.0: target_seed = "rice"
            
            # Use seed from inventory if available
            inv_count = inv.get(target_seed, 0)
            if inv_count > 0:
                p = empty_plots[0]
                return {"action_type": "plant", "plot_id": p.plot_id, "seed_type": target_seed}, f"Planting {target_seed.upper()}: Expansion into Plot {p.plot_id}."
            
            # Otherwise buy the target seed
            # Only buy bulk if we have a healthy surplus
            if money >= SEED_CONFIG[target_seed]["base_buy"] * 4 + 50: 
                return {"action_type": "buy_seeds", "seed_type": target_seed, "quantity": 4}, f"Buying Seeds: Bulk {target_seed.upper()} purchase (Funds: ${money:.0f})."
            elif money >= SEED_CONFIG[target_seed]["base_buy"] + 20:
                return {"action_type": "buy_seeds", "seed_type": target_seed, "quantity": 1}, f"Buying Seeds: Single {target_seed.upper()} purchase to maintain reserves."
            elif money >= SEED_CONFIG["wheat"]["base_buy"]:
                return {"action_type": "buy_seeds", "seed_type": "wheat", "quantity": 1}, "Buying Seeds: Low funds. Standardizing on Wheat."

        # ── 4. MARKET LIQUIDATION (Level 3) ──────────────────────────────────
        
        for crop, qty in storage.items():
            # Sell threshold: 10kg for Wheat, 15kg for others
            sell_threshold = 15.0
            if crop == "wheat": sell_threshold = 10.0
            
            if qty >= sell_threshold:
                price_info = obs.market_prices.get(crop)
                # Sell if market is safe or we need cash desperately
                if price_info and (price_info.trend >= -0.1 or money < 30):
                    return {"action_type": "sell", "seed_type": crop, "quantity": int(qty)}, f"Selling {qty:.0f}kg {crop.upper()}: Liquidating for revenue."

        # ── 5. FINAL FALLBACK ────────────────────────────────────────────────
        
        return {"action_type": "wait"}, "Waiting: Environment stable. Observing growth cycles."
