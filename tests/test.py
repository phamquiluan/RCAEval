"""Tests."""
import os
import shutil
from os import path

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
    # local_path = tempfile.NamedTemporaryFile().name
    # download_online_boutique_dataset(local_path=local_path)
    # assert path.exists(local_path), local_path
    # shutil.rmtree(local_path)
    
    local_path = tempfile.NamedTemporaryFile().name
    func(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)    
    
