import subprocess
import time
import requests

BASE = "http://localhost:7860"


def start_server():
    proc = subprocess.Popen(
        ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"],
        cwd="server",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(3)   # give uvicorn time to start
    return proc


def test_health(proc):
    r = requests.get(f"{BASE}/health")
    assert r.status_code == 200, f"health failed: {r.status_code}"
    print("health OK")


def test_reset(proc):
    r = requests.post(f"{BASE}/reset", json={})
    assert r.status_code == 200, f"reset failed: {r.status_code} {r.text}"
    body = r.json()
    obs = body.get("observation", body)
    assert "day"          in obs
    assert "money"        in obs
    assert "plots"        in obs
    assert "text_summary" in obs
    assert body["done"]   == False
    assert body["reward"] is None
    print(f"reset OK — day={obs['day']} money={obs['money']}")


def test_step_wait(proc):
    requests.post(f"{BASE}/reset", json={})
    r = requests.post(f"{BASE}/step", json={"action": {"action_type": "wait"}})
    assert r.status_code == 200, f"step failed: {r.status_code} {r.text}"
    body = r.json()
    obs = body.get("observation", body)
    assert obs["day"]     == 1
    assert body["reward"] is not None
    assert body["done"]   == False
    print(f"step_wait OK — reward={body['reward']}")


def test_step_invalid(proc):
    requests.post(f"{BASE}/reset", json={})
    r = requests.post(f"{BASE}/step", json={"action": {"action_type": "harvest", "plot_id": 0}})
    assert r.status_code == 200
    body = r.json()
    # reward includes daily passive + post-advance, but action reward is -1.0
    # check that reward is negative (harvest on empty plot is invalid)
    assert body["reward"] < 0, f"expected negative reward, got {body['reward']}"
    print(f"step_invalid OK — reward={body['reward']}")


def test_state(proc):
    requests.post(f"{BASE}/reset", json={})
    r = requests.get(f"{BASE}/state")
    assert r.status_code == 200
    body = r.json()
    assert "step_count" in body
    assert "episode_id" in body
    print(f"state OK — episode_id={body.get('episode_id')}")


def test_full_sequence(proc):
    requests.post(f"{BASE}/reset", json={})

    # buy wheat seeds
    r = requests.post(f"{BASE}/step", json={
        "action": {"action_type": "buy_seeds", "seed_type": "wheat", "quantity": 2}
    })
    obs = r.json().get("observation", r.json())
    assert obs["seed_inventory"]["wheat"] == 2

    # plant on plot 0
    r = requests.post(f"{BASE}/step", json={
        "action": {"action_type": "plant", "plot_id": 0, "seed_type": "wheat"}
    })
    assert r.status_code == 200
    obs = r.json().get("observation", r.json())
    print(f"  After plant: day={obs['day']} plot0={obs['plots'][0]}")

    # grow for days, irrigating periodically to keep crop healthy
    for i in range(8):
        if i % 3 == 0:
            r = requests.post(f"{BASE}/step", json={
                "action": {"action_type": "irrigate", "plot_id": 0}
            })
        else:
            r = requests.post(f"{BASE}/step", json={"action": {"action_type": "wait"}})
        obs = r.json().get("observation", r.json())
        print(f"  Day {obs['day']}: stage={obs['plots'][0]['stage']} health={obs['plots'][0]['health']} moisture={obs['plots'][0]['soil_moisture']} days_planted={obs['plots'][0]['days_planted']}")

    # check plot state before harvest
    obs_before = r.json().get("observation", r.json())
    print(f"  PRE-HARVEST plot0: {obs_before['plots'][0]}")

    # harvest
    r = requests.post(f"{BASE}/step", json={
        "action": {"action_type": "harvest", "plot_id": 0}
    })
    body = r.json()
    print(f"  Harvest response reward={body['reward']}")
    obs = body.get("observation", body)
    print(f"  POST-HARVEST plot0: {obs['plots'][0]}")
    assert body["reward"] > 0, f"harvest should reward, got {body['reward']}"
    print(f"full_sequence OK — post-harvest reward={body['reward']}")


if __name__ == "__main__":
    proc = start_server()
    try:
        test_health(proc)
        test_reset(proc)
        test_step_wait(proc)
        test_step_invalid(proc)
        test_state(proc)
        test_full_sequence(proc)
        print("\nPhase 5 complete — all server tests passed")
    finally:
        proc.terminate()
