import os
import sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd



currentpath = os.getcwd()


def run_sim(i):
    # Construct the command to run sim.py with the current index
    command = ["python3", " sim.py", str(i)]
    os.system(" ".join(command))

# Use ThreadPoolExecutor to run tasks in parallel
with ThreadPoolExecutor() as executor:
    executor.map(run_sim, range(4))
