"""
Phase 6 tests for inference.py
===============================
Unit tests run without any server.
Integration tests start the server in a subprocess.

Run: python3 test_inference_mock.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import unittest
from unittest.mock import MagicMock

# Make sure project root is on the path so we can import inference
import os
sys.path.insert(0, os.path.dirname(__file__))

from inference import (
    FALLBACK_ACTION,
    FarmEnvClient,
    parse_action,
    validate_action,
    run_episode,
    run_task,
)

BASE = "http://localhost:7860"


# ---------------------------------------------------------------------------
# Unit tests — no server, no LLM
# ---------------------------------------------------------------------------

class TestParseAction(unittest.TestCase):

    def test_clean_json(self):
        result = parse_action('{"action_type": "wait"}')
        self.assertEqual(result["action_type"], "wait")

    def test_json_buried_in_prose(self):
        result = parse_action('Sure thing! {"action_type": "harvest", "plot_id": 2}')
        self.assertEqual(result["action_type"], "harvest")
        self.assertEqual(result["plot_id"], 2)

    def test_json_in_markdown_fences(self):
        result = parse_action(
            '```json\n{"action_type": "plant", "plot_id": 0, "seed_type": "wheat"}\n```'
        )
        self.assertEqual(result["action_type"], "plant")

    def test_empty_string_returns_fallback(self):
        self.assertEqual(parse_action(""), FALLBACK_ACTION)

    def test_whitespace_only_returns_fallback(self):
        self.assertEqual(parse_action("   \n\t  "), FALLBACK_ACTION)

    def test_pure_garbage_returns_fallback(self):
        self.assertEqual(parse_action("I have no idea what to do"), FALLBACK_ACTION)

    def test_multiple_json_objects_takes_first(self):
        result = parse_action(
            '{"action_type": "irrigate", "plot_id": 1} or {"action_type": "wait"}'
        )
        self.assertEqual(result["action_type"], "irrigate")


class TestValidateAction(unittest.TestCase):

    def test_valid_wait(self):
        result = validate_action({"action_type": "wait"})
        self.assertEqual(result["action_type"], "wait")

    def test_unknown_action_type(self):
        self.assertEqual(validate_action({"action_type": "nuke"}), FALLBACK_ACTION)

    def test_plot_id_out_of_range_high(self):
        self.assertEqual(
            validate_action({"action_type": "irrigate", "plot_id": 9}),
            FALLBACK_ACTION,
        )

    def test_plot_id_out_of_range_negative(self):
        self.assertEqual(
            validate_action({"action_type": "irrigate", "plot_id": -1}),
            FALLBACK_ACTION,
        )

    def test_plot_id_string_coerced_to_int(self):
        result = validate_action({"action_type": "harvest", "plot_id": "2"})
        self.assertEqual(result["plot_id"], 2)

    def test_quantity_zero_rejected(self):
        self.assertEqual(
            validate_action({"action_type": "buy_seeds", "seed_type": "wheat", "quantity": 0}),
            FALLBACK_ACTION,
        )

    def test_quantity_negative_rejected(self):
        self.assertEqual(
            validate_action({"action_type": "sell", "seed_type": "corn", "quantity": -5}),
            FALLBACK_ACTION,
        )

    def test_quantity_string_coerced_to_int(self):
        result = validate_action({"action_type": "buy_seeds", "seed_type": "wheat", "quantity": "3"})
        self.assertEqual(result["quantity"], 3)

    def test_invalid_seed_type(self):
        self.assertEqual(
            validate_action({"action_type": "plant", "plot_id": 0, "seed_type": "potato"}),
            FALLBACK_ACTION,
        )

    def test_non_dict_input(self):
        self.assertEqual(validate_action("wait"), FALLBACK_ACTION)
        self.assertEqual(validate_action(None), FALLBACK_ACTION)
        self.assertEqual(validate_action(42), FALLBACK_ACTION)

    def test_valid_full_buy_action(self):
        result = validate_action({"action_type": "buy_seeds", "seed_type": "corn", "quantity": 3})
        self.assertEqual(result["action_type"], "buy_seeds")
        self.assertEqual(result["quantity"], 3)

    def test_valid_plant_action(self):
        result = validate_action({"action_type": "plant", "plot_id": 3, "seed_type": "rice"})
        self.assertEqual(result["plot_id"], 3)
        self.assertEqual(result["seed_type"], "rice")

    def test_all_valid_action_types_pass(self):
        for atype in ["wait", "buy_seeds", "plant", "irrigate", "harvest", "sell"]:
            result = validate_action({"action_type": atype})
            self.assertEqual(result["action_type"], atype)


# ---------------------------------------------------------------------------
# Integration tests — server must be running
# ---------------------------------------------------------------------------

def _start_server() -> subprocess.Popen:
    proc = subprocess.Popen(
        ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"],
        cwd=os.path.join(os.path.dirname(__file__), "server"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)
    return proc


def _make_mock_llm(actions: list) -> MagicMock:
    """
    Returns a mock OpenAI client that cycles through the given action list.
    When the list is exhausted, returns wait.
    """
    call_count = [0]

    def mock_create(**kwargs):
        idx    = min(call_count[0], len(actions) - 1)
        action = actions[idx]
        call_count[0] += 1
        resp                             = MagicMock()
        resp.choices[0].message.content  = json.dumps(action)
        return resp

    mock                         = MagicMock()
    mock.chat.completions.create = mock_create
    return mock


def test_health(proc: subprocess.Popen) -> None:
    client = FarmEnvClient(BASE)
    assert client.health(), "server not healthy"
    print("  health OK")


def test_reset(proc: subprocess.Popen) -> None:
    client = FarmEnvClient(BASE)
    obs    = client.reset(task_id=1)
    assert "day"          in obs, f"missing 'day', got keys: {list(obs.keys())}"
    assert "money"        in obs
    assert "plots"        in obs
    assert "text_summary" in obs
    assert obs["done"]   == False
    assert obs["reward"] is None
    assert len(obs["plots"]) == 4
    print(f"  reset OK — day={obs['day']} money={obs['money']}")


def test_step_wait(proc: subprocess.Popen) -> None:
    client = FarmEnvClient(BASE)
    client.reset(task_id=1)
    obs = client.step({"action_type": "wait"})
    assert obs["day"]    == 1
    assert obs["reward"] is not None
    assert obs["done"]   == False
    print(f"  step_wait OK — reward={obs['reward']}")


def test_step_invalid_returns_penalty(proc: subprocess.Popen) -> None:
    client = FarmEnvClient(BASE)
    client.reset(task_id=1)
    obs = client.step({"action_type": "harvest", "plot_id": 0})
    assert obs["reward"] == -1.0, f"expected -1.0 got {obs['reward']}"
    print(f"  step_invalid OK — reward={obs['reward']}")


def test_state_endpoint(proc: subprocess.Popen) -> None:
    client = FarmEnvClient(BASE)
    client.reset(task_id=1)
    state = client.state()
    # openenv-core /state uses response_model=State (base class) so only
    # episode_id and step_count are returned over HTTP.
    assert "episode_id" in state, f"missing episode_id, got: {list(state.keys())}"
    assert "step_count" in state, f"missing step_count, got: {list(state.keys())}"
    assert state["step_count"] >= 0
    print(f"  state OK — episode_id={state['episode_id']} step_count={state['step_count']}")


def test_run_episode_smart_agent(proc: subprocess.Popen) -> None:
    """
    Full episode with a mock LLM following: buy → plant all → wait → harvest → sell.
    Verifies that run_episode() returns keys and a valid grade.
    """
    env = FarmEnvClient(BASE)
    llm = _make_mock_llm([
        {"action_type": "buy_seeds", "seed_type": "wheat", "quantity": 4},
        {"action_type": "plant", "plot_id": 0, "seed_type": "wheat"},
        {"action_type": "plant", "plot_id": 1, "seed_type": "wheat"},
        {"action_type": "plant", "plot_id": 2, "seed_type": "wheat"},
        {"action_type": "plant", "plot_id": 3, "seed_type": "wheat"},
        *[{"action_type": "wait"}] * 30,
    ])

    result = run_episode(env, llm, task_id=1, episode=1)

    assert "grade"        in result
    assert "total_reward" in result
    assert "steps"        in result
    assert "final_money"  in result
    assert 0.0 <= result["grade"] <= 1.0
    assert result["steps"] > 0

    print(
        f"  run_episode OK — grade={result['grade']:.4f} "
        f"steps={result['steps']} money=${result['final_money']:.2f}"
    )


def test_run_task_returns_avg_grade(proc: subprocess.Popen) -> None:
    env = FarmEnvClient(BASE)
    llm = _make_mock_llm([{"action_type": "wait"}])

    result = run_task(env, llm, task_id=1)

    assert result["task_id"]    == 1
    assert result["difficulty"] == "easy"
    assert 0.0 <= result["avg_grade"] <= 1.0
    assert len(result["episodes"]) > 0
    print(f"  run_task OK — avg_grade={result['avg_grade']:.4f}")


def test_episode_done_flag_respected(proc: subprocess.Popen) -> None:
    """Episode must stop when done=True even if MAX_STEPS not reached."""
    env = FarmEnvClient(BASE)
    llm = _make_mock_llm([{"action_type": "wait"}])

    # Run with a very large MAX_STEPS — the env should terminate on its own
    import inference as inf
    original = inf.MAX_STEPS
    inf.MAX_STEPS = 999
    try:
        result = run_episode(env, llm, task_id=1, episode=1)
        # Task 1 max_days=30, so episode must end before step 999
        assert result["steps"] <= 35, f"episode ran too long: {result['steps']} steps"
    finally:
        inf.MAX_STEPS = original
    print(f"  done_flag OK — episode ended at step {result['steps']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 54)
    print("  Phase 6 — inference.py test suite")
    print("=" * 54)

    # Run unit tests first (no server needed)
    print("\nUnit tests (no server required)...")
    loader      = unittest.TestLoader()
    suite       = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestParseAction))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateAction))
    runner      = unittest.TextTestRunner(verbosity=0)
    unit_result = runner.run(suite)

    if not unit_result.wasSuccessful():
        print("\nUnit tests FAILED — fix before running integration tests")
        sys.exit(1)
    print("Unit tests passed ✓")

    # Integration tests — start server
    print("\nStarting server for integration tests...")
    proc = _start_server()

    try:
        test_health(proc)
        test_reset(proc)
        test_step_wait(proc)
        test_step_invalid_returns_penalty(proc)
        test_state_endpoint(proc)
        test_run_episode_smart_agent(proc)
        test_run_task_returns_avg_grade(proc)
        test_episode_done_flag_respected(proc)
    finally:
        proc.terminate()

    print(f"\n{'=' * 54}")
    print("  Phase 6 complete — all inference tests passed ✓")
    print(f"{'=' * 54}")
