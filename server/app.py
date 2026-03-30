from __future__ import annotations

import os
import sys

import uvicorn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openenv.core import create_app
from farming_environment import FarmingEnvironment
from models import FarmAction, FarmObservation


# Create a singleton environment for HTTP statefulness during development/testing
GLOBAL_ENV = None

def make_env() -> FarmingEnvironment:
    """
    Factory function — create_app calls this every time it needs
    a fresh environment instance. 
    By returning a singleton, we preserve state across stateless HTTP calls.
    task_id can be overridden via FARMING_TASK_ID environment variable.
    """
    global GLOBAL_ENV
    if GLOBAL_ENV is None:
        task_id = int(os.getenv("FARMING_TASK_ID", "1"))
        GLOBAL_ENV = FarmingEnvironment(task_id=task_id)
    return GLOBAL_ENV


app = create_app(
    env=make_env,
    action_cls=FarmAction,
    observation_cls=FarmObservation,
    env_name="farming-env",
)


def main():
    """Entry point for running the server directly."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
