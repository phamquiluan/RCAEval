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


def test_rq2_script():
    command = ["python", "main.py", "--method", "baro", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"


def test_multi_source():
    import pandas as pd
    from RCAEval.e2e import mmbaro, mmcirca
    from RCAEval.utility import download_multi_source_sample

    download_multi_source_sample()

    data_dir = path.join("data", "multi-source-data")
    data = {
        "metric": pd.read_csv(path.join(data_dir, "metrics.csv")),
        "logts": pd.read_csv(path.join(data_dir, "logts.csv")),
        "tracets_err": pd.read_csv(path.join(data_dir, "tracets_err.csv")),
        "tracets_lat": pd.read_csv(path.join(data_dir, "tracets_lat.csv")),
    }
    with open(path.join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.read().strip())

    for func in (mmbaro, mmcirca):
        ranks = func(data, inject_time, dataset="re2-ob")["ranks"]
        assert len(ranks) > 0, f"{func.__name__} returned no ranks"
        print(f"{func.__name__} top 5 root causes:", ranks[:5])


def test_basic():
    # You can put the code here to a file named test.py
    from RCAEval.e2e.baro import baro
    from RCAEval.utility import download_data, read_data

    # download a sample data to data.csv
    download_data()

    # read data from data.csv
    data = read_data("data.csv")
    anomaly_detected_timestamp = 1692569339

    # perform root cause analysis
    root_causes = baro(data, anomaly_detected_timestamp)["ranks"]

    # print the top 5 root causes
    print("Top 5 root causes:", root_causes[:5])