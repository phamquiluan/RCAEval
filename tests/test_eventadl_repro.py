"""Tests."""
import subprocess


def test_eventadl():
    command = ["python", "main.py", "--method", "eventadl", "--dataset", "eventadl-falcon", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"
