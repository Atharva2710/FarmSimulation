import sys
import asyncio
sys.path.insert(0, './server')
from gradio_app import create_gradio_ui

def make_env():
    from farming_environment import FarmingEnvironment
    return FarmingEnvironment(task_id=1)

import gradio as gr
from gradio.blocks import Block
ui = create_gradio_ui(make_env)

# Try to find the do_wait function directly
# We can just extract it from the local scope? No, we can't easily.
# But wait! What if I just call make_env().step({"action_type": "wait"})? I did that and it worked.
