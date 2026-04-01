"""
Phase 7 checkpoint.
Requires: docker CLI installed, port 7860 free.
Run from project root: python test_phase7.py
"""

import subprocess
import time
import sys
import requests

IMAGE_NAME = "farming-env-test"
CONTAINER_NAME = "farming-env-phase7"
PORT = 7860
BASE = f"http://localhost:{PORT}"


def run(cmd, **kwargs):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kwargs)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    return result


def stop_container():
    run(f"docker rm -f {CONTAINER_NAME} 2>/dev/null || true")


def test_docker_build():
    print("\n[1] Docker build...")
    # Build context is project root, Dockerfile is in server/
    result = run(
        f"docker build -f server/Dockerfile -t {IMAGE_NAME} ."
    )
    assert result.returncode == 0, (
        f"docker build failed:\n{result.stderr}"
    )
    print("    build OK")


def test_container_starts():
    print("\n[2] Container start...")
    stop_container()
    result = run(
        f"docker run -d --name {CONTAINER_NAME} "
        f"-p {PORT}:{PORT} "
        f"-e FARMING_TASK_ID=1 "
        f"{IMAGE_NAME}"
    )
    assert result.returncode == 0, f"docker run failed:\n{result.stderr}"
    time.sleep(4)  # wait for uvicorn to start
    print("    container started")


def test_health_endpoint():
    print("\n[3] GET /health...")
    r = requests.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200, f"health returned {r.status_code}: {r.text}"
    print(f"    OK — {r.json()}")


def test_reset_endpoint():
    print("\n[4] POST /reset...")
    r = requests.post(f"{BASE}/reset", json={}, timeout=15)
    assert r.status_code == 200, f"reset returned {r.status_code}: {r.text}"
    body = r.json()
    obs = body["observation"]
    assert obs["day"] == 0
    assert obs["money"] == 200.0
    assert body["done"] == False
    assert len(obs["plots"]) == 4
    assert len(obs["text_summary"]) > 0
    print(f"    OK — day={obs['day']}, money={obs['money']}")


def test_step_endpoint():
    print("\n[5] POST /step...")
    requests.post(f"{BASE}/reset", json={}, timeout=15)
    r = requests.post(
        f"{BASE}/step",
        json={"action": {"action_type": "wait"}},
        timeout=15,
    )
    assert r.status_code == 200, f"step returned {r.status_code}: {r.text}"
    body = r.json()
    obs = body["observation"]
    assert obs["day"] == 1
    assert body["reward"] is not None
    print(f"    OK — day={obs['day']}, reward={body['reward']}")


def test_state_endpoint():
    print("\n[6] GET /state...")
    r = requests.get(f"{BASE}/state", timeout=10)
    assert r.status_code == 200, f"state returned {r.status_code}: {r.text}"
    body = r.json()
    assert "episode_id" in body
    assert "step_count" in body
    print(f"    OK — episode_id={body['episode_id']}")


def test_openenv_validate():
    print("\n[7] openenv validate...")
    result = run("openenv validate")
    assert result.returncode == 0, (
        f"openenv validate failed:\n{result.stderr}\n{result.stdout}"
    )
    print("    validate OK")


def test_readme_has_hf_header():
    print("\n[8] README.md HF Space header...")
    with open("README.md") as f:
        content = f.read()
    assert content.startswith("---"), "README.md must start with --- YAML header"
    assert "sdk: docker" in content, "README.md must contain 'sdk: docker'"
    assert "openenv" in content, "README.md must have openenv tag"
    print("    OK")


def test_baseline_results_exist():
    print("\n[9] baseline_results.json exists...")
    import json, os
    assert os.path.exists("baseline_results.json"), (
        "baseline_results.json not found — run inference.py first"
    )
    with open("baseline_results.json") as f:
        data = json.load(f)
    assert "tasks" in data
    assert "overall_avg" in data
    for task in data["tasks"]:
        assert 0.0 <= task["avg_grade"] <= 1.0, (
            f"grade out of range for task {task['task_id']}: {task['avg_grade']}"
        )
    print(f"    OK — overall_avg={data['overall_avg']}")


if __name__ == "__main__":
    print("=" * 52)
    print("  Phase 7 — Docker + HF Space deployment checks")
    print("=" * 52)

    try:
        test_docker_build()
        test_container_starts()
        test_health_endpoint()
        test_reset_endpoint()
        test_step_endpoint()
        test_state_endpoint()
    finally:
        # stop_container()
        run(f"docker rmi {IMAGE_NAME} 2>/dev/null || true")

    # these don't need docker
    test_openenv_validate()
    test_readme_has_hf_header()
    test_baseline_results_exist()

    print(f"\n{'=' * 52}")
    print("  Phase 7 complete — ready to push to HF Hub")
    print(f"{'=' * 52}")
