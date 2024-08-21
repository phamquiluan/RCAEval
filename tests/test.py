"""Tests."""
import os
import shutil
from os import path
import subprocess
import pytest

from typing import Callable
import numpy as np
import pandas as pd
import pytest
import tempfile

from RCAEval.utility import (
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
    download_syn_rcd_dataset,
    download_syn_circa_dataset,
    download_syn_causil_dataset,
    download_rca_rcd_dataset,
    download_rca_circa_dataset,
)


@pytest.mark.parametrize("func", [
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
    download_syn_rcd_dataset,
    download_syn_circa_dataset,
    download_syn_causil_dataset,
    download_rca_rcd_dataset,
    download_rca_circa_dataset,
])
def test_download_dataset(func: Callable):
    """Test download dataset."""
    local_path = tempfile.NamedTemporaryFile().name
    func(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)    
    

def test_rq2_script():
    command = ["python", "rq2.py", "--method", "baro", "--dataset", "online-boutique"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"