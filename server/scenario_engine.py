
import random
import json
import time
import traceback
from typing import Dict, Any, List, Optional
from server.scenario_definitions import SCENARIOS, Scenario
from agents import HeuristicAgent, HybridAgent
from models import FarmObservation

class ScenarioEngine:
    def __init__(self, agent: Any):
        self.agent = agent
        self.results = []
        self.summary = {
            "passed": 0,
            "failed": 0,
            "score": 0.0,
            "difficulty_scores": {1: 0.0, 2: 0.0, 3: 0.0}
        }

    def run_tests(
        self, 
        difficulty: Optional[int] = None, 
        seed: int = 42, 
        hf_token: Optional[str] = None, 
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Runs the test suite with configuration injection."""
        random.seed(seed)
        self.results = []
        
        # Inject metadata if HybridAgent
        if isinstance(self.agent, HybridAgent):
            if model_name: 
                import os
                # This affects the global MODEL_NAME if imported in hybrid.py, 
                # but better to update the instance property if it exists.
                # Since HybridAgent uses MODEL_NAME from os.environ, we update OS env.
                os.environ["MODEL_NAME"] = model_name
        
        test_queue = [s for s in SCENARIOS if difficulty is None or s.difficulty == difficulty]
        
        passed_count = 0
        difficulty_counts = {1: 0, 2: 0, 3: 0}
        difficulty_passed = {1: 0, 2: 0, 3: 0}

        for i, scenario in enumerate(test_queue):
            print(f"[ENGINE] Running Scenario {i+1}/{len(test_queue)}: {scenario.name}...")
            obs = scenario.get_observation(seed=seed)
            
            try:
                # Execution with timeout/loop protection
                start_time = time.time()
                
                # Call agent
                if isinstance(self.agent, HybridAgent):
                    res = self.agent.act(obs, api_token=hf_token)
                    # Handle dict return
                    action = res["action"]
                    thought = res["thought"]
                else:
                    action, thought = self.agent.act(obs)
                
                duration = time.time() - start_time
                
                # Validation
                act_type = action.get("action_type") if isinstance(action, dict) else action
                is_passed = act_type in scenario.expected_actions
                
                status = "✅ PASS" if is_passed else "❌ FAIL"
                if is_passed:
                    passed_count += 1
                    difficulty_passed[scenario.difficulty] += 1
                
                difficulty_counts[scenario.difficulty] += 1
                
                self.results.append({
                    "id": i + 1,
                    "name": scenario.name,
                    "difficulty": scenario.difficulty,
                    "expected": scenario.expected_actions,
                    "actual": act_type,
                    "thought": thought,
                    "status": status,
                    "duration": round(duration, 3)
                })
                
            except Exception as e:
                print(f"[ENGINE] Error in Scenario {scenario.name}: {e}")
                self.results.append({
                    "id": i + 1,
                    "name": scenario.name,
                    "difficulty": scenario.difficulty,
                    "expected": scenario.expected_actions,
                    "actual": "ERROR",
                    "thought": f"Crash: {str(e)}",
                    "status": "💥 CRASH",
                    "duration": 0.0
                })
                difficulty_counts[scenario.difficulty] += 1

        # Summary calculations
        total = len(self.results)
        self.summary["passed"] = passed_count
        self.summary["failed"] = total - passed_count
        self.summary["score"] = (passed_count / total * 100) if total > 0 else 0
        
        for d in [1, 2, 3]:
            count = difficulty_counts[d]
            if count > 0:
                self.summary["difficulty_scores"][d] = round((difficulty_passed[d] / count * 100), 1)

        return {
            "summary": self.summary,
            "results": self.results
        }
