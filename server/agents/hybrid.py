
import json
import os
import re
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
import textwrap

from .heuristic import HeuristicAgent
from models import FarmObservation, SEED_CONFIG

# Constants for LLM
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

class HybridAgent:
    """
    An AI agent that uses a Large Language Model (LLM) 
    augmented by a Heuristic (Rule-based) 'Advisory' system.
    """
    
    def __init__(self):
        self.heuristic = HeuristicAgent()
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self.name = f"Hybrid (LLM+{MODEL_NAME})"
        
        self.system_prompt = textwrap.dedent("""
            You are the CHIEF ECONOMIST & GROWTH STRATEGIST.
            Goal: Maximize NET WORTH through "Balanced Exponential Growth."
            
            STRATEGIC HIERARCHY:
            1. SURVIVAL (Priority 1): Maintain > 3 days of Water Runway and Moisture > RAW Depletion Limit (e.g., Rice: 80%, Corn: 50%, Wheat: 45%).
            2. PUMPING: If Tank < 15L and Runway < 5 days, PUMP water.
            3. EXPONENTIAL GROWTH (Priority 2):
               - If Money > $150: Plant CORN (Highest yield).
               - If Money > $80: Plant RICE.
               - Otherwise: Plant WHEAT.
            4. MARKET TIMING (Peak Detection):
               - HODL storage if Current Price < 7-day Average.
               - SELL ALL if Current Price > 7-day Average AND Premium > 10%.
            
            REASONING STRUCTURE:
            - RISK AUDIT: Moisture Stress? Tank Critical?
            - MARKET AUDIT: Is current price at a 7-day peak?
            - ACTION: Prioritize EXPANSION if survival is secure.
            
            OUTPUT FORMAT (Strict JSON):
            {
              "fidelity_audit": {"runway_days": float, "net_worth": float, "is_peak": bool},
              "action": {"action_type": "...", "plot_id": 0-3, "seed_type": "...", "quantity": 1},
              "thought": "Deep audit of Runway, 7-day Price Delta, and Growth ROI."
            }
        """).strip()

    def _build_state_text(self, obs: FarmObservation) -> str:
        """Converts the observation into a clean text summary for the LLM."""
        # Telemetry Calculations
        temp = getattr(obs.climate, "temperature", 22.0)
        hum = getattr(obs.climate, "humidity", 0.6)
        eto = (temp / 100.0) * (1.1 - hum)
        
        # Calculate Economics
        storage_val = sum(qty * obs.market_prices[c].sell_price for c, qty in obs.storage.items() if c in obs.market_prices)
        inv_val = sum(qty * SEED_CONFIG[c]['base_buy'] for c, qty in obs.seed_inventory.items() if c in SEED_CONFIG)
        net_worth = obs.money + storage_val + inv_val
        
        # Calculate Water Runway (Days)
        total_etc = sum((eto * SEED_CONFIG[p.crop_type if p.crop_type else 'wheat'].get('Kc', 1.0)) for p in obs.plots if p.stage not in ['empty', 'withered'])
        runway = (obs.water_tank / total_etc) if total_etc > 0 else 99

        lines = [
            f"--- ECONOMIC DASHBOARD (Day {obs.day}) ---",
            f"NET WORTH: ${net_worth:.2f} | MONEY: ${obs.money:.2f}",
            f"WATER SECURITY: {obs.water_tank:.1f}L | RUNWAY: {runway:.1f} DAYS",
            f"CLIMATE: {obs.climate.climate_type} (ETo: {eto:.3f})",
            f"STORAGE: {obs.storage}",
            "\nMARKET INTELLIGENCE (Current vs 7-Day Avg):",
        ]
        
        for seed, cfg in SEED_CONFIG.items():
            market = obs.market_prices.get(seed)
            if market:
                premium = ((market.sell_price / cfg['base_sell']) - 1) * 100
                lines.append(f"- {seed.upper()}: ${market.sell_price:.2f} ({premium:+.1f}% vs Base), Trend: {market.trend:+.2f}")

        lines.append("\nPLOT STATUS & PROJECTIONS:")
        for p in obs.plots:
            if p.stage == "empty": continue
            
            crop_cfg = SEED_CONFIG.get(p.crop_type, {})
            p_const = crop_cfg.get("p", "N/A")
            kc = crop_cfg.get("Kc", 1.0)
            yield_kg = crop_cfg.get("yield_kg", 0)
            etc = eto * kc
            proj_val = yield_kg * (obs.market_prices[p.crop_type].sell_price if p.crop_type in obs.market_prices else 0)
            
            status = f"- Plot {p.plot_id}: {p.stage} {p.crop_type if p.crop_type else ''}"
            vitals = f"  Moisture: {p.soil_moisture*100:.1f}% | RAW: {p_const} | ETc: {etc:.3f} | Proj $: {proj_val:.2f}"
            lines.append(status + "\n" + vitals)
            
        return "\n".join(lines)

    def act(self, obs: FarmObservation, api_token: Optional[str] = None) -> Dict[str, Any]:
        """
        1. Run Heuristic to get safe advice.
        2. Call LLM for strategic optimization.
        """
        if api_token and api_token.strip():
            if not hasattr(self, "_last_token") or self._last_token != api_token:
                self.client = OpenAI(base_url=API_BASE_URL, api_key=api_token)
                self._last_token = api_token
        
        # 1. Get Heuristic Suggestion (Safety Fallback)
        h_action, h_thought = self.heuristic.act(obs)
        
        # 2. Build Message
        state_text = self._build_state_text(obs)
        user_message = textwrap.dedent(f"""
            --- SENSOR DATA (Day {obs.day}) ---
            {state_text}
            
            --- HEURISTIC ADVISORY (SAFETY FALLBACK) ---
            Suggestion: {json.dumps(h_action)}
            Logic: {h_thought}
            
            --- TRIAGE AUDIT TEMPLATE ---
            1. RISK AUDIT: Moisture Stress? Runway Panic?
            2. MARKET AUDIT: Is current sell premium > 10%?
            3. ACTION: Choose the action that maximizes long-term ROI.
            
            Reply with exactly one JSON object.
        """).strip()
        
        # Build Fidelity Audit Data
        storage_val = sum(qty * obs.market_prices[c].sell_price for c, qty in obs.storage.items() if c in obs.market_prices)
        inv_val = sum(qty * SEED_CONFIG[c]['base_buy'] for c, qty in obs.seed_inventory.items() if c in SEED_CONFIG)
        net_worth = obs.money + storage_val + inv_val
        
        temp = getattr(obs.climate, "temperature", 22.0)
        hum = getattr(obs.climate, "humidity", 0.6)
        eto = (temp / 100.0) * (1.1 - hum)
        total_etc = sum((eto * SEED_CONFIG[p.crop_type if p.crop_type else 'wheat'].get('Kc', 1.0)) for p in obs.plots if p.stage not in ['empty', 'withered'])
        runway = (obs.water_tank / total_etc) if total_etc > 0 else 99
        
        fidelity = {
            "runway_days": round(runway, 2),
            "net_worth": round(net_worth, 2),
            "hallucination_detected": False # Placeholder
        }

        try:
            if not api_token or not api_token.strip():
                raise Exception("401: Unauthorized")

            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            raw_text = response.choices[0].message.content
            match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                action = data.get("action", h_action)
                thought = data.get("thought", "Strategic execution via market analysis.")
                audit = data.get("fidelity_audit", {})
                
                # Merge for logging
                if isinstance(action, dict):
                    action["thought"] = thought
                    action["state_summary"] = audit # Use audit for state summary field
                
                return {
                    "action": action,
                    "thought": f"🧠 {thought}",
                    "fidelity_audit": fidelity
                }
            
            return {
                "action": h_action,
                "thought": "Error: Unparseable response. Falling back to Safety Mode.",
                "fidelity_audit": fidelity
            }
            
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "unauthorized" in err_str.lower():
                return {
                    "action": h_action,
                    "thought": f"❌ AUTH ERROR (401): Please check your HF Token in the control panel.",
                    "fidelity_audit": fidelity
                }
            return {
                "action": h_action,
                "thought": f"❌ LLM ERROR: {err_str}. Using Safety Mode.",
                "fidelity_audit": fidelity
            }
