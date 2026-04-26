from gradio_client import Client
import time
import subprocess

# Start server
proc = subprocess.Popen(["python3", "server/app.py"])
time.sleep(3)

try:
    client = Client("http://127.0.0.1:7860/")
    
    # Check endpoints
    print(client.view_api(return_format="dict"))
    
    # Try hitting the wait endpoint (assuming it's /do_wait)
    # the endpoints are numbered
finally:
    proc.terminate()
