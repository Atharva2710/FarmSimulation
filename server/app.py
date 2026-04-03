from __future__ import annotations

import os

import uvicorn

from openenv.core import create_app
from farming_environment import FarmingEnvironment
from models import FarmAction, FarmObservation  

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


import gradio as gr
from gradio_app import create_gradio_ui

app = create_app(
    env=make_env,
    action_cls=FarmAction,
    observation_cls=FarmObservation,
    env_name="farming-env",
)

# Mount the interactive Gradio dashboard at the root path.
# This makes the Space look and feel like a modern application
# while keeping the /reset, /step, etc. endpoints functional for agents.
ui = create_gradio_ui(make_env)
app = gr.mount_gradio_app(app, ui, path="/")


def main():
    """Entry point for running the server directly.
    
    """
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
