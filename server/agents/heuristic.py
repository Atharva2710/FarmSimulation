
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

    def act(self, obs: FarmObservation, mode: str = "physics") -> Tuple[Dict[str, Any], str]:
        """
        Dispatches to the selected heuristic mode.
        """
        if mode == "legacy":
            return self._act_legacy(obs)
        return self._act_physics(obs)

    def _act_physics(self, obs: FarmObservation) -> Tuple[Dict[str, Any], str]:
        """
        FAO-56 Physics-Informed Triage: ROI-Optimized
        """
        plots = obs.plots
        money = obs.money
        tank = obs.water_tank
        inv = obs.seed_inventory
        storage = obs.storage
        
        # ── 0. PHYSICS CALCULATIONS (FAO-56) ──────────────────────────────────
        climate = obs.climate
        temp = getattr(climate, "temperature", 22.0)
        hum = getattr(climate, "humidity", 0.6)
        # simplified ETo: higher temp and lower humidity = higher transpiration
        eto = (temp / 100.0) * (1.1 - hum) 
        
        # ── 1. IMMEDIATE SURVIVAL (Level 10) ──────────────────────────────────
        
        # Priority 1: Critical Dehydration (Depletion > RAW * 1.5)
        critical_plots = []
        for p in plots:
            if p.stage in ["empty", "withered", "mature"]: continue
            crop_cfg = SEED_CONFIG.get(p.crop_type, {})
            p_const = crop_cfg.get("p", 0.5)
            # RAW trigger: moisture < (1 - p)
            # Critical trigger: moisture < (1 - p) * 0.7
            if p.soil_moisture < (1.0 - p_const) * 0.7:
                critical_plots.append(p)
                
        if critical_plots:
            driest = min(critical_plots, key=lambda p: p.soil_moisture)
            if tank >= 15.0:
                return {"action_type": "irrigate", "plot_id": driest.plot_id}, f"CRITICAL PHYSICS ALERT (Plot {driest.plot_id}): Moisture {driest.soil_moisture*100:.0f}% is below stress limit for {driest.crop_type.upper()}. RAW depletion exceeded."
            elif money >= PUMP_COST and getattr(obs, 'aquifer', 1.0) > 0:
                return {"action_type": "pump_water"}, "Emergency Physics-Fix: Pumping water to prevent stress-induced yield loss."

        # Priority 2: Clear Withered
        for p in plots:
            if p.stage == "withered":
                return {"action_type": "clear", "plot_id": p.plot_id}, f"Physics Cleanup: Removing Plot {p.plot_id} remnants."

        # ── 2. HAZARD MAINTENANCE (Level 9) ──────────────────────────────────
        
        # Priority 3: Spray Pests
        for p in plots:
            if p.has_pests and money >= PESTICIDE_COST:
                return {"action_type": "spray_pesticide", "plot_id": p.plot_id}, f"Hazard Def: Neutralizing pests on Plot {p.plot_id}."

        # Priority 4: Weeding
        for p in plots:
            if p.has_weeds:
                return {"action_type": "pull_weeds", "plot_id": p.plot_id}, f"Hazard Def: Removing nutrient competition."

        # ── 3. GROWTH & RESOURCE BALANCING (Level 8) ─────────────────────────
        
        # Priority 5: Harvest
        for p in plots:
            if p.stage == "mature":
                return {"action_type": "harvest", "plot_id": p.plot_id}, f"Yield Capture: Plot {p.plot_id} is biologically mature."

        # Priority 6: Maintenance Irrigation (FAO-56 Trigger)
        thirsty_plots = []
        for p in plots:
            if p.stage in ["empty", "withered", "mature"]: continue
            crop_cfg = SEED_CONFIG.get(p.crop_type, {})
            kc = crop_cfg.get("Kc", 1.0)
            etc = eto * kc
            
            # Maintenance Trigger: If current moisture < (1 - p + buffer)
            p_const = SEED_CONFIG[p.crop_type].get("p", 0.5)
            raw_limit = (1.0 - p_const)
            if p.soil_moisture < raw_limit + 0.02:
                thirsty_plots.append((p, etc, raw_limit))
                
        if thirsty_plots:
            target, etc_val, _ = min(thirsty_plots, key=lambda x: x[0].soil_moisture)
            if tank >= 15.0:
                return {"action_type": "irrigate", "plot_id": target.plot_id}, f"Physics-Informed Irrigation (Plot {target.plot_id}): Demand ETc={etc_val:.3f}. Current depletion approaching limit p={SEED_CONFIG[target.crop_type]['p']}."
            elif money >= PUMP_COST and getattr(obs, 'aquifer', 1.0) > 0:
                return {"action_type": "pump_water"}, "Resource Optimization: Recharging tank for physics-driven demand."

        # Priority 7: Fertilizer
        for p in plots:
            if p.stage in ["seedling", "growing"] and money >= FERTILIZER_COST:
                if p.health < 0.9 or p.nitrogen < 0.4:
                    return {"action_type": "apply_fertilizer", "plot_id": p.plot_id}, "NPK Supplement: Optimizing nutrient uptake."

        # ── 4. STRATEGIC EXPANSION & PUMPING (Level 5) ────────────────────────
        
        tank_floor = 0.5 if (eto > 0.2) else 0.3 # Pumping floor varies with ET demand
        if tank < tank_floor and money >= PUMP_COST and getattr(obs, 'aquifer', 1.0) > 0:
            if money >= PUMP_COST + 30:
                return {"action_type": "pump_water"}, f"Safety Recharge: High ET demand ({eto:.2f}) detected."

        empty_plots = [p for p in plots if p.stage == "empty"]
        if empty_plots:
            target_seed = "wheat"
            if money > 80.0: target_seed = "corn"
            elif money > 50.0: target_seed = "rice"
            
            if inv.get(target_seed, 0) > 0:
                return {"action_type": "plant", "plot_id": empty_plots[0].plot_id, "seed_type": target_seed}, f"Expanding: Biological diversity increase."
            
            if money >= SEED_CONFIG[target_seed]["base_buy"] * 4 + 60: 
                return {"action_type": "buy_seeds", "seed_type": target_seed, "quantity": 4}, "Procurement: Bulk seed purchase."
            elif money >= SEED_CONFIG[target_seed]["base_buy"] + 30:
                return {"action_type": "buy_seeds", "seed_type": target_seed, "quantity": 1}, "Procurement: Single seed purchase."

        # ── 5. MARKET LIQUIDATION (Level 3) ──────────────────────────────────
        max_days = getattr(obs, "max_days", 30)
        is_closing_bell = (obs.day >= max_days - 5)
        for crop, qty in storage.items():
            if qty <= 0.1: continue
            sell_threshold = 1.0 if is_closing_bell else 15.0
            if qty >= sell_threshold:
                price_info = obs.market_prices.get(crop)
                is_peak = (price_info and price_info.sell_price > price_info.avg_7d)
                if is_closing_bell or is_peak or money < 20:
                    return {"action_type": "sell", "seed_type": crop, "quantity": int(qty)}, f"Liquidation: {'Closing Bell' if is_closing_bell else 'Market Peak (>' + str(price_info.avg_7d) + ')'}."

        return {"action_type": "wait"}, "Steady State: No physics-based triggers active."
    def _act_legacy(self, obs: FarmObservation) -> Tuple[Dict[str, Any], str]:
        """
        Legacy Threshold-Based Triage (Survival First)
        """
        plots = obs.plots
        money = obs.money
        tank = obs.water_tank
        inv = obs.seed_inventory
        storage = obs.storage

        # Priority 1: Survive (Moisture < 30%)
        thirsty = [p for p in plots if p.stage not in ["empty", "withered"] and p.soil_moisture < 0.3]
        if thirsty:
            target = min(thirsty, key=lambda p: p.soil_moisture)
            if tank >= 15.0:
                return {"action_type": "irrigate", "plot_id": target.plot_id}, f"Legacy Priority: Plot {target.plot_id} below 30% moisture."
            elif money >= PUMP_COST:
                return {"action_type": "pump_water"}, "Legacy Pumping: Low tank for thirsty crops."

        # Priority 2: Harvest
        for p in plots:
            if p.stage == "mature":
                return {"action_type": "harvest", "plot_id": p.plot_id}, "Legacy Priority: Mature crop harvested."

        # Priority 3: Clear
        for p in plots:
            if p.stage == "withered":
                return {"action_type": "clear", "plot_id": p.plot_id}, "Legacy Priority: Clearing withered remnants."

        # Priority 4: Hazards
        for p in plots:
            if p.has_pests and money >= PESTICIDE_COST:
                return {"action_type": "spray_pesticide", "plot_id": p.plot_id}, "Legacy Hazard: Spraying pests."
            if p.has_weeds:
                return {"action_type": "pull_weeds", "plot_id": p.plot_id}, "Legacy Hazard: Pulling weeds."

        # Priority 5: Expand
        empty_plots = [p for p in plots if p.stage == "empty"]
        if empty_plots:
            seed = "wheat"
            if money > 150: seed = "corn"
            elif money > 80: seed = "rice"
            
            if inv.get(seed, 0) > 0:
                return {"action_type": "plant", "plot_id": empty_plots[0].plot_id, "seed_type": seed}, f"Legacy Expansion: Planting {seed}."
            if money >= SEED_CONFIG[seed]["base_buy"] + 20:
                return {"action_type": "buy_seeds", "seed_type": seed, "quantity": 1}, f"Legacy Buy: Purchasing {seed} seeds."

        # Priority 6: Sell
        for crop, qty in storage.items():
            if qty >= 10.0:
                return {"action_type": "sell", "seed_type": crop, "quantity": int(qty)}, f"Legacy Profit: Selling {qty}kg of {crop}."

        return {"action_type": "wait"}, "Legacy Idle: No threshold triggers active."
