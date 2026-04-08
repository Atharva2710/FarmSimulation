
import random
import json
import time
import traceback
from typing import Dict, Any, List, Optional
from server.scenario_definitions import SCENARIOS, Scenario
from server.agents import HeuristicAgent, HybridAgent
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

    async def run_tests_stream(
        self, 
        difficulty: Optional[int] = None, 
        seed: int = 42, 
        hf_token: Optional[str] = None, 
        model_name: Optional[str] = None,
        use_heuristic: bool = False,
        h_mode: str = "physics"
    ) -> Dict[str, Any]:
        """Async generator for test results."""
        import asyncio
        random.seed(seed)
        self.results = []
        
        test_queue = [s for s in SCENARIOS if difficulty is None or s.difficulty == difficulty]
        
        passed_count = 0
        difficulty_counts = {1: 0, 2: 0, 3: 0}
        difficulty_passed = {1: 0, 2: 0, 3: 0}

        for i, scenario in enumerate(test_queue):
            obs = scenario.get_observation(seed=seed)
            try:
                start_time = time.time()
                
                # We wrap the sync act() in a thread to keep UI responsive
                if isinstance(self.agent, HybridAgent):
                    # Use asyncio.to_thread for the blocking LLM call
                    res = await asyncio.to_thread(self.agent.act, obs, api_token=hf_token, use_heuristic=use_heuristic, mode=h_mode)
                    action = res["action"]
                    thought = res["thought"]
                else:
                    action, thought = self.agent.act(obs, mode=h_mode)
                
                duration = time.time() - start_time
                act_type = action.get("action_type") if isinstance(action, dict) else action
                
                # Check for LLM errors in thought
                is_err = thought.startswith("❌")
                
                # Logic: Pass only if action matches AND it wasn't a defaulted wait due to error (unless fallback was on)
                is_passed = act_type in scenario.expected_actions
                if is_err and not use_heuristic:
                    is_passed = False
                    
                status = "✅ PASS" if is_passed else "❌ FAIL"
                
                if is_passed:
                    passed_count += 1
                    difficulty_passed[scenario.difficulty] += 1
                difficulty_counts[scenario.difficulty] += 1
                
                result = {
                    "id": i + 1,
                    "name": scenario.name,
                    "difficulty": scenario.difficulty,
                    "expected": scenario.expected_actions,
                    "actual": act_type,
                    "thought": thought,
                    "status": status,
                    "duration": round(duration, 3)
                }
                self.results.append(result)
                yield result
                
            except Exception as e:
                difficulty_counts[scenario.difficulty] += 1
                err_result = {
                    "id": i + 1,
                    "name": scenario.name,
                    "difficulty": scenario.difficulty,
                    "expected": scenario.expected_actions,
                    "actual": "ERROR",
                    "thought": f"Crash: {str(e)}",
                    "status": "💥 CRASH",
                    "duration": 0.0
                }
                self.results.append(err_result)
                yield err_result

        # Final Summary
        total = len(self.results)
        self.summary["passed"] = passed_count
        self.summary["failed"] = total - passed_count
        self.summary["score"] = (passed_count / total * 100) if total > 0 else 0
        self.summary["total"] = total
        
        yield {"summary": self.summary}
