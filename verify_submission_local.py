"""
verify_submission_local.py — §15 checklist items that can be verified without a live HF Space.
Run at any point during development; run again at H 29 before final submit.
Items requiring a live Space URL are noted but skipped here.
"""
import sys
import os
import json
import subprocess

sys.path.insert(0, os.path.abspath("."))

PASS = 0
FAIL = 0

def ok(msg):
    global PASS
    PASS += 1
    print(f"  ✅ {msg}")

def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  ❌ {msg}")

def skip(msg):
    print(f"  ⏭  SKIP (needs live Space): {msg}")


print("\n=== §15 Local Submission Checklist ===\n")

# ── Engineering table-stakes ──────────────────────────────────────────────────
print("[ Engineering table-stakes ]")

try:
    from openenv.core import Environment
    from server.farming_environment import FarmingEnvironment
    assert issubclass(FarmingEnvironment, Environment)
    ok("Uses OpenEnv Environment base class")
except Exception as e:
    fail(f"OpenEnv base class check: {e}")

try:
    from server.farming_environment import ACTION_LABOR_COSTS
    reserved = {"reset", "step", "state", "close"}
    collisions = set(ACTION_LABOR_COSTS.keys()) & reserved
    if collisions:
        fail(f"MCP reserved name collision: {collisions}")
    else:
        ok("No MCP reserved-name collisions")
except Exception as e:
    fail(f"MCP name check: {e}")

try:
    result = subprocess.run(["openenv", "validate"], capture_output=True, text=True, cwd=".")
    if result.returncode == 0:
        ok("openenv validate passes")
    else:
        fail(f"openenv validate: {result.stdout} {result.stderr}")
except Exception as e:
    fail(f"openenv validate error: {e}")

try:
    import yaml
    with open("openenv.yaml") as f:
        cfg = yaml.safe_load(f)
    assert "tasks" in cfg and len(cfg["tasks"]) == 3
    for task in cfg["tasks"]:
        assert "grader" in task, f"task {task['id']} missing grader block"
    ok("openenv.yaml has 3 tasks with composite grader blocks")
except ImportError:
    # yaml not available, check with json-like approach
    with open("openenv.yaml") as f:
        content = f.read()
    if "grader:" in content and "dimensions:" in content:
        ok("openenv.yaml has grader/dimensions blocks (yaml module not available for deep check)")
    else:
        fail("openenv.yaml missing grader/dimensions blocks")
except Exception as e:
    fail(f"openenv.yaml check: {e}")

print()

# ── Core verification suite ───────────────────────────────────────────────────
print("[ Core verification suite ]")

try:
    result = subprocess.run([sys.executable, "verify_all.py"], capture_output=True, text=True)
    if "Verification complete" in result.stdout and result.returncode == 0:
        ok("verify_all.py passes (phase2 + phase3 + phase4 + e2e)")
    else:
        fail(f"verify_all.py: {result.stdout[-300:]} {result.stderr[-200:]}")
except Exception as e:
    fail(f"verify_all.py error: {e}")

try:
    result = subprocess.run([sys.executable, "robustness_validation.py"], capture_output=True, text=True)
    if "Final Status: PASSED" in result.stdout:
        ok("robustness_validation.py PASSED")
    else:
        fail(f"robustness_validation.py: {result.stdout[-200:]}")
except Exception as e:
    fail(f"robustness_validation.py error: {e}")

try:
    result = subprocess.run([sys.executable, "verify_phase1_audit.py"], capture_output=True, text=True)
    if "Verification Complete" in result.stdout and result.returncode == 0:
        ok("verify_phase1_audit.py PASSED")
    else:
        fail(f"verify_phase1_audit.py: {result.stdout[-200:]}")
except Exception as e:
    fail(f"verify_phase1_audit.py error: {e}")

print()

# ── Hygiene checks ────────────────────────────────────────────────────────────
print("[ Hygiene ]")

env_files = [f for f in os.listdir(".") if f.startswith(".env") or f == "secrets.json"]
if env_files:
    fail(f"Possible secrets files in root: {env_files}")
else:
    ok("No .env / secrets files in root")

if os.path.exists("baseline_results_pivoted.json"):
    with open("baseline_results_pivoted.json") as f:
        b = json.load(f)
    ok(f"baseline_results_pivoted.json present (random/heuristic recorded)")
else:
    fail("baseline_results_pivoted.json missing")

if os.path.exists("Planning/archive"):
    archived = os.listdir("Planning/archive")
    ok(f"Planning/archive exists with {len(archived)} docs")
else:
    fail("Planning/archive missing")

root_files = os.listdir(".")
distracting = [f for f in ["META_HACKATHON_ANALYSIS.md", "STRATEGIC_SUMMARY.md",
               "FIXES_APPLIED.md", "honey_app.py", "main.py"] if f in root_files]
if distracting:
    fail(f"Stale meta-files still in root: {distracting}")
else:
    ok("Root clean — meta-files archived")

notebook_path = "ws4-notebook/notebooks/train_grpo_unsloth.ipynb"
if os.path.exists(notebook_path):
    ok("GRPO training notebook present")
else:
    alt = "notebooks/train_grpo_unsloth.ipynb"
    if os.path.exists(alt):
        ok(f"GRPO training notebook present at {alt}")
    else:
        fail(f"GRPO training notebook not found at {notebook_path} or {alt}")

print()

# ── Requires live HF Space (skip locally) ─────────────────────────────────────
print("[ Requires live HF Space — run validate-submission.sh <url> ]")
skip("validate-submission.sh <hf-space-url> passes (Space ping + Docker build)")
skip("HF Space /reset and /step respond with 200")
skip("HF Hub checkpoint downloads with from_pretrained()")
skip("Colab notebook runs end-to-end on fresh runtime")
skip("Loss/reward PNGs committed + embedded in README")
skip("README links to Space, Hub, Colab, video")
skip("robustness_validation.py shows: random < heuristic < 72B < 0.5B-trained")

print()
print(f"=== Result: {PASS} passed, {FAIL} failed ===")
if FAIL > 0:
    sys.exit(1)
