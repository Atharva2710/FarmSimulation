import os
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from farming_environment import FarmingEnvironment
import traceback

try:
    print("Testing FarmingEnvironment initialization...")
    env = FarmingEnvironment(task_id=1)
    print("Testing get_observation()...")
    obs = env.get_observation()
    print("Testing _build_text_summary()...")
    summary = env._build_text_summary()
    print("Success!")
    print(f"Summary length: {len(summary)}")
except Exception as e:
    print("CRASH DETECTED!")
    traceback.print_exc()
    sys.exit(1)
