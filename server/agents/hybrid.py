
import json
import os
import re
from typing import Dict, Any, Tuple, Optional
from openai import OpenAI
import textwrap

from .heuristic import HeuristicAgent
from models import FarmObservation

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
            You are the STRATEGIC COMMANDER of an automated farm. 
            You must process high-resolution sensor data and make optimal decisions.
            
            CONNECTION RULES (Observation -> Action):
            1. IF Soil Moisture < 0.5 AND Water Tank > 15L -> ACTION: irrigate
            2. IF Pests Present AND Money >= 1.5 -> ACTION: spray_pesticide (CRITICAL)
            3. IF Plot Empty AND Money > 50 -> ACTION: plant corn (HIGH ROI)
            4. IF Plot Empty AND Money < 20 -> ACTION: plant wheat (CASH FLOW)
            
            Your decision-making process MUST follow this hierarchy:
            - SURVIVAL: Stop Pests, Weeds, and Dehydration.
            - EXPANSION: Fill every empty plot as fast as possible.
            - ROI: Plant Corn ($12) for high profit; Wheat ($5) for quick cash.
            - LIQUIDATION: Sell crops only if prices are stable or rising.
            
            OUTPUT FORMAT (Strict JSON):
            {
              "analysis": "Brief analysis of current threats and opportunities",
              "action": {"action_type": "...", "plot_id": 0-3, ...},
              "thought": "Why this action is the most intelligent choice based on the data"
            }
        """).strip()

    def _build_state_text(self, obs: FarmObservation) -> str:
        """Converts the observation into a clean text summary for the LLM."""
        lines = [
            f"Day: {obs.day}",
            f"Money: ${obs.money:.2f}",
            f"Water Tank: {obs.water_tank:.1f}L / 100L",
            f"Aquifer: {obs.aquifer:.1f}L",
            f"Inventory: {obs.seed_inventory}",
            f"Storage: {obs.storage}",
            f"Climate: {obs.climate.climate_type} ({obs.climate.temperature}°C, {obs.climate.humidity*100:.0f}% Humidity)",
        ]
        lines.append("\nPlots:")
        for p in obs.plots:
            lines.append(f"- Plot {p.plot_id}: {p.stage} {p.crop_type if p.crop_type else ''} (Health: {p.health*100:.0f}%, Water: {p.soil_moisture*100:.0f}%)")
            if p.has_pests: lines.append(f"  [!] PESTS DETECTED (Severity: {p.pest_severity*100:.0f}%)")
            if p.has_weeds: lines.append(f"  [!] WEEDS DETECTED")
            if p.pesticide_protection > 0: lines.append(f"  [Shield: {p.pesticide_protection} days]")
            
        return "\n".join(lines)

    def act(self, obs: FarmObservation) -> Tuple[Dict[str, Any], str]:
        """
        1. Run Heuristic to get advice.
        2. Combine advice with state in prompt.
        3. Call LLM for final decision.
        """
        # 1. Get Heuristic Suggestion
        h_action, h_thought = self.heuristic.act(obs)
        
        # 2. Build Message
        state_text = self._build_state_text(obs)
        user_message = f"""
        --- SENSOR DATA (Day {obs.day}) ---
        {state_text}
        
        --- MARKET INTEL ---
        {json.dumps(obs.model_dump().get("market_prices", {}), indent=2)}
        
        --- HEURISTIC ADVISORY (Optimal Rules) ---
        The system logic suggests: {json.dumps(h_action)}
        Rule Justification: {h_thought}
        
        Analyze the sensor data and market intel. Compare it to the heuristic advisory. 
        If the heuristic is moving too slow or picking low-yield seeds, override it for higher ROI.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=250,
                temperature=0.2
            )
            
            raw_text = response.choices[0].message.content
            # Attempt to parse JSON from response
            match = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                action = data.get("action", h_action)
                analysis = data.get("analysis", "State analyzed.")
                thought = data.get("thought", "Determined via strategic correlation.")
                return action, f"🧠 {analysis} | {thought}"
            
            return h_action, f"Error: LLM returned unparseable response. Falling back to Heuristic. ({raw_text[:50]}...)"
            
        except Exception as e:
            return h_action, f"Error: LLM call failed ({str(e)}). Falling back to Heuristic."
