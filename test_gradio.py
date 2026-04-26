import sys
import asyncio
sys.path.insert(0, './server')
from gradio_app import create_gradio_ui

def make_env():
    from farming_environment import FarmingEnvironment
    return FarmingEnvironment(task_id=1)

ui = create_gradio_ui(make_env)
