from datetime import datetime
import json
import pathlib
import shutil
import sys
import os
from os.path import join
import requests
import zipfile
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ENCODING = "utf-8"


def is_py312():
    return sys.version_info.major == 3 and sys.version_info.minor == 12


def is_py310():
    return sys.version_info.major == 3 and sys.version_info.minor == 10


def is_py38():
    return sys.version_info.major == 3 and sys.version_info.minor == 8


def dump_json(filename: str, data):
    """
    Dump data into a json file
    """
    with open(filename, "w", encoding=ENCODING) as obj:
        json.dump(data, obj, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(filename: str):
    """
    Load data from a json file
    """
    with open(filename, encoding=ENCODING) as obj:
        return json.load(obj)


def convert_adjacency_matrix(adj, node_names):
    """
    convert metrics adj to service adj
    """
    services = list(set([name.split("_")[0] for name in node_names]))
    # print(services)
    num_services = len(services)

    service_adj = np.zeros((num_services, num_services))

    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            if adj[i][j] == 1:
                service_adj[services.index(node_names[i].split("_")[0])][
                    services.index(node_names[j].split("_")[0])
                ] = 1

    # remove cycles
    for i in range(num_services):
        service_adj[i][i] = 0

    return service_adj, services  # services is node_names but for services


def download_data(remote_url=None, local_path=None):
    """Download data from a remote URL."""
    if remote_url is None:
        remote_url = "https://github.com/phamquiluan/baro/releases/download/0.0.4/simple_data.csv"
    if local_path is None:
        local_path = "data.csv"

    response = requests.get(remote_url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024 # 1 Kibibyte

    progress_bar = tqdm(
        desc=f"Downloading {local_path}..",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(local_path, "wb") as ref:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            ref.write(data)

    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def download_metric_sample(remote_url=None, local_path=None):
    """Download a sample metric case"""
    if remote_url is None:
        remote_url = "https://github.com/phamquiluan/baro/releases/download/0.0.4/simple_data.csv"
    if local_path is None:
        local_path = "data.csv"

    download_data(remote_url, local_path)
    

def download_multi_source_sample(local_path=None):
    """Download a sample multi-source telemetry data case"""
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "multi-source-data")):
        return
    
    download_data("https://github.com/phamquiluan/RCAEval/releases/download/0.2.0/multi-source-data.zip", "multi-source-data.zip")
    with zipfile.ZipFile("multi-source-data.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("multi-source-data.zip")


def download_online_boutique_dataset(local_path=None):
    """Download the Online Boutique dataset from Zenodo."""
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "online-boutique")):
        return
    download_data("https://zenodo.org/records/13305663/files/online-boutique.zip?download=1", "online-boutique.zip")
    with zipfile.ZipFile("online-boutique.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("online-boutique.zip")

###################################################
# LLM Ref Stack
def prepare_llm_ref_stack_dataset(local_root_path, dataset_name):
    def extract_anomaly_type(folder_name):
        if 'cpu-stress' in folder_name:
            return 'cpu-stress', 'stress-chaos-cpu', "cpu"
        elif 'memory-stress' in folder_name:
            return 'memory-stress', 'stress-chaos-memory', "mem"
        elif 'network-stress' in folder_name:
            return 'network-stress', 'network-chaos-delay', "delay"
        return None, None, None

    """Prepare the LLM Ref Stack dataset from the local root path."""
    local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, dataset_name)):
        return
    else:
        os.makedirs(join(local_path, dataset_name))
    
    for item in os.listdir(os.path.join(local_root_path, dataset_name)):
        item_path = os.path.join(local_root_path, dataset_name, item)
        if not os.path.isdir(item_path) or item.startswith('.'):
            continue

        anomaly_type, fault_name, fault_name_short = extract_anomaly_type(item)
        if not anomaly_type:
            continue

        target = item.split(f'-{anomaly_type}-')[-1]

        for cand in os.listdir(os.path.join(item_path, 'results')):
            cand_path = os.path.join(item_path, 'results', cand)
            if not os.path.isdir(cand_path) or cand.startswith('.'):
                continue

            iteration = cand.split('-iteration-')[-1]

            new_iter_path = pathlib.Path(f"{local_path}/{dataset_name}/{target}_{fault_name_short}/{iteration}")
            new_iter_path.mkdir(parents=True, exist_ok=True)
            ### prepare metric data
            exp_dir_path = pathlib.Path(f"{cand_path}/rca-collector-results")
            data_df = pd.read_csv(exp_dir_path.joinpath('data.csv'))
            latency_df = pd.read_csv(exp_dir_path.joinpath('latency_aggregated_90.csv'))
            latency_df = latency_df.rename(columns={col: f"{col}_latency-90" for col in latency_df.columns if col != "time"})
            new_data_df = pd.merge(data_df, latency_df, on="time")
            new_data_df["time"] = pd.to_datetime(new_data_df["time"]).astype("int64") // 10**9
            new_data_df = new_data_df.loc[:, ~new_data_df.columns.str.contains('^Unnamed')]
            new_data_df.to_csv(new_iter_path.joinpath('data.csv'), index=False)
            ### Copy mpg data
            shutil.copy(exp_dir_path.joinpath('mpg.csv'), new_iter_path.joinpath('mpg.csv'))
            ### Copy detailed latency data
            shutil.copy(exp_dir_path.joinpath('latency_merged_90.csv'), new_iter_path.joinpath('latency_merged_90.csv'))
            ### prepare injection time
            # Load JSON data from file
            with open(exp_dir_path.parent.joinpath('apply_result_chaos_anomaly_injection.json'), 'r') as f:
                data = json.load(f)
            # Extract creationTimestamp
            timestamp_str = data["result"]["metadata"]["creationTimestamp"]
            # Convert to epoch time (UTC)
            dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
            # Write to file
            with open(new_iter_path.joinpath('inject_time.txt'), "w") as f:
                f.write(str(int(dt.timestamp())) + "\n")

###################################################

def download_sock_shop_1_dataset(local_path=None):
    """Download the Sock Shop 1 dataset from Zenodo."""
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "sock-shop-1")):
        return
    download_data("https://zenodo.org/records/13305663/files/sock-shop-1.zip?download=1", "sock-shop-1.zip")
    with zipfile.ZipFile("sock-shop-1.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("sock-shop-1.zip")

    
def download_sock_shop_2_dataset(local_path=None):
    """Download the Sock Shop 2 dataset from Zenodo."""
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "sock-shop-2")):
        return
    download_data("https://zenodo.org/records/13305663/files/sock-shop-2.zip?download=1", "sock-shop-2.zip")
    with zipfile.ZipFile("sock-shop-2.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("sock-shop-2.zip")
    

def download_train_ticket_dataset(local_path=None):
    """Download the Train Ticket dataset from Zenodo."""
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "train-ticket")):
        return
    download_data("https://zenodo.org/records/13305663/files/train-ticket.zip?download=1", "train-ticket.zip")
    with zipfile.ZipFile("train-ticket.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("train-ticket.zip")
    

def download_re1ob_dataset(local_path=None):
    """Download the RE1 dataset, Online Boutique system from Zenodo."""
    if local_path == None:
        local_path = join("data", "RE1")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "RE1-OB")):
        return
    download_data("https://zenodo.org/records/14590730/files/RE1-OB.zip?download=1", "RE1-OB.zip")
    with zipfile.ZipFile("RE1-OB.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("RE1-OB.zip")


def download_re1ss_dataset(local_path=None):
    """Download the RE1 dataset, Sock Shop system from Zenodo."""
    if local_path == None:
        local_path = join("data", "RE1")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "RE1-SS")):
        return
    download_data("https://zenodo.org/records/14590730/files/RE1-SS.zip?download=1", "RE1-SS.zip")
    with zipfile.ZipFile("RE1-SS.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("RE1-SS.zip")
    

def download_re1tt_dataset(local_path=None):
    """Download the RE1 dataset, Train Ticket system from Zenodo."""
    if local_path == None:
        local_path = join("data", "RE1")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "RE1-TT")):
        return
    download_data("https://zenodo.org/records/14590730/files/RE1-TT.zip?download=1", "RE1-TT.zip")
    with zipfile.ZipFile("RE1-TT.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("RE1-TT.zip")
    

def download_re1_dataset(local_path=None):
    """Download the RE1 dataset from Zenodo."""
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    RE1_local_path = join(local_path, "RE1")
    if os.path.exists(RE1_local_path):
        return

    download_re1ob_dataset(local_path=RE1_local_path)
    download_re1ss_dataset(local_path=RE1_local_path)
    download_re1tt_dataset(local_path=RE1_local_path)


def download_re2ob_dataset(local_path=None):
    """Download the RE2 dataset, Online Boutique system from Zenodo."""
    if local_path == None:
        local_path = join("data", "RE2")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "RE2-OB")):
        return
    download_data("https://zenodo.org/records/14590730/files/RE2-OB.zip?download=1", "RE2-OB.zip")
    with zipfile.ZipFile("RE2-OB.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("RE2-OB.zip")    


def download_re2ss_dataset(local_path=None):
    """Download the RE2 dataset, Sock Shop system from Zenodo."""
    if local_path == None:
        local_path = join("data", "RE2")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "RE2-SS")):
        return
    download_data("https://zenodo.org/records/14590730/files/RE2-SS.zip?download=1", "RE2-SS.zip")
    with zipfile.ZipFile("RE2-SS.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("RE2-SS.zip")    
    

def download_re2tt_dataset(local_path=None):
    """Download the RE2 dataset, Train Ticket system from Zenodo."""
    if local_path == None:
        local_path = join("data", "RE2")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "RE2-TT")):
        return
    download_data("https://zenodo.org/records/14590730/files/RE2-TT.zip?download=1", "RE2-TT.zip")
    with zipfile.ZipFile("RE2-TT.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("RE2-TT.zip")    
    

def download_re2_dataset(local_path=None):
    """Download the RE2 dataset from Zenodo."""
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    
    RE2_local_path = join(local_path, "RE2")
    if os.path.exists(RE2_local_path):
        return
    
    download_re2ob_dataset(local_path=RE2_local_path)
    download_re2ss_dataset(local_path=RE2_local_path)
    download_re2tt_dataset(local_path=RE2_local_path)
        

def download_re3ob_dataset(local_path=None):
    """Download the RE3 dataset, Online Boutique system from Zenodo."""
    if local_path == None:
        local_path = join("data", "RE3")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "RE3-OB")):
        return
    download_data("https://zenodo.org/records/14590730/files/RE3-OB.zip?download=1", "RE3-OB.zip")
    with zipfile.ZipFile("RE3-OB.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("RE3-OB.zip")
    

def download_re3ss_dataset(local_path=None):
    """Download the RE3 dataset, Sock Shop system from Zenodo."""
    if local_path == None:
        local_path = join("data", "RE3")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "RE3-SS")):
        return
    download_data("https://zenodo.org/records/14590730/files/RE3-SS.zip?download=1", "RE3-SS.zip")
    with zipfile.ZipFile("RE3-SS.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("RE3-SS.zip")
    

def download_re3tt_dataset(local_path=None):
    """Download the RE3 dataset, Train Ticket system from Zenodo."""
    if local_path == None:
        local_path = join("data", "RE3")
    if not os.path.exists(local_path):
        os.makedirs(local_path)
    if os.path.exists(join(local_path, "RE3-TT")):
        return
    download_data("https://zenodo.org/records/14590730/files/RE3-TT.zip?download=1", "RE3-TT.zip")
    with zipfile.ZipFile("RE3-TT.zip", 'r') as file:
        file.extractall(local_path)
    os.remove("RE3-TT.zip")
    

def download_re3_dataset(local_path=None):
    """Download the RE3 dataset from Zenodo."""
    if local_path == None:
        local_path = "data"
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    RE3_local_path = join(local_path, "RE3")
    if os.path.exists(RE3_local_path):
        return

    download_re3ob_dataset(local_path=RE3_local_path)
    download_re3ss_dataset(local_path=RE3_local_path)
    download_re3tt_dataset(local_path=RE3_local_path)
    
def read_data(data_path, strip=True):
    """Read CSV data for root cause analysis."""
    data = pd.read_csv(data_path)
    data_dir = os.path.dirname(data_path)

    ############# PREPROCESSING ###############
    if "time.1" in data:
        data = data.drop(columns=["time.1"])
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.ffill()
    data = data.fillna(0)

    # remove latency-50 columns
    data = data.loc[:, ~data.columns.str.endswith("latency-50")]
    # rename latency-90 columns to latency
    data = data.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in data.columns
            if c.endswith("_latency-90")
        }
    )

    return data


