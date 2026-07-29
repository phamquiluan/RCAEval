"""Tests."""
import subprocess
import sys
from pathlib import Path

import pytest

DATASET_DIR = Path(__file__).resolve().parent.parent / "data" / "eventadl-falcon"


@pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason="eventadl-falcon dataset not available (see docs/EVENTADL.md)",
)
def test_eventadl():
    command = [sys.executable, "main.py", "--method", "eventadl", "--dataset", "eventadl-falcon", "--test"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"
