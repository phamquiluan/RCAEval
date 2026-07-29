"""Tests."""
import subprocess
import sys


def test_eventadl():
    command = [sys.executable, "main.py", "--method", "eventadl", "--dataset", "eventadl-falcon", "--test"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"
