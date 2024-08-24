import argparse
import glob
import json
import os
import shutil
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from os.path import abspath, basename, dirname, exists, join

# turn off all warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.classes.graph import Node

from RCAEval.io.time_series import drop_constant, drop_time, preprocess
from RCAEval.utility import (
    dump_json,
    is_py310,
    load_json,
    download_rca_circa_dataset,
    download_rca_rcd_dataset,
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
)


if is_py310():
    from RCAEval.e2e import ( 
        circa,
        cloudranger,
        cmlp_pagerank,
        dummy,
        e_diagnosis,
        easyrca,
        fci_pagerank,
        fci_randomwalk,
        ges_pagerank,
        granger_pagerank,
        granger_randomwalk,
        lingam_pagerank,
        lingam_randomwalk,
        micro_diag,
        microcause,
        nsigma,
        ntlr_pagerank,
        ntlr_randomwalk,
        pc_pagerank,
        pc_randomwalk,
        robust_scaler,
        run,
    )

    baro = robust_scaler
else:
    from RCAEval.e2e import dummy, e_diagnosis, ht, rcd

try:
    from RCAEval.e2e.ges_pagerank import fges_pagerank, fges_randomwalk
except ImportError:
    print("fges_pagerank not available")

try:
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from RCAEval.e2e.causalrca import causalrca
except ImportError:
    pass
    # print("causalrca not available")



AVAILABLE_METHODS = sorted(
    [
        "causalrca",
        "circa",
        "cloudranger",
        "dummy",
        "fci_pagerank" "ges_pagerank",
        "granger_pagerank",
        "lingam_pagerank",
        "micro_diag",
        "microcause",
        "nsigma",
        "pc_pagerank",
        "pc_randomwalk",
        "rcd",
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    parser.add_argument("--method", type=str, help="Choose a method.")
    parser.add_argument("--dataset", type=str, help="Choose a dataset.", choices=[
        "rcd10", "rcd50", "circa10", "circa50", "online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket"
    ])
    parser.add_argument("--length", type=int, default=None, help="Time series length (RQ4)")
    parser.add_argument("--tdelta", type=int, default=0, help="Specify $t_delta$ to simulate delay in anomaly detection")
    parser.add_argument("--test", action="store_true", help="Perform smoke test on certain methods without fully run on all data")
    args = parser.parse_args()

    if args.method not in globals():
        raise ValueError(f"{args.method=} not defined. Available: {AVAILABLE_METHODS}")

    return args


args = parse_args()
is_synthetic = False

# download dataset
if "circa" in args.dataset:
    is_synthetic = True
    download_rca_circa_dataset()
elif "rcd" in args.dataset:
    is_synthetic = True
    download_rca_rcd_dataset()
elif "online-boutique" in args.dataset:
    download_online_boutique_dataset()
elif "sock-shop-1" in args.dataset:
    download_sock_shop_1_dataset()
elif "sock-shop-2" in args.dataset:
    download_sock_shop_2_dataset()
elif "train-ticket" in args.dataset:
    download_train_ticket_dataset()

DATASET_MAP = {
    "circa10": "data/rca_circa/10",
    "circa50": "data/rca_circa/50",
    "rcd10": "data/rca_rcd/10",
    "rcd50": "data/rca_rcd/50",
    "online-boutique": "data/online-boutique",
    "sock-shop-1": "data/sock-shop-1",
    "sock-shop-2": "data/sock-shop-2",
    "train-ticket": "data/train-ticket",
}
dataset = DATASET_MAP[args.dataset]



# prepare input paths
data_paths = list(glob.glob(os.path.join(dataset, "**/data.csv"), recursive=True))
new_data_paths = []
for p in data_paths: 
    if os.path.exists(p.replace("data.csv", "simple_data.csv")):
        new_data_paths.append(p.replace("data.csv", "simple_data.csv"))
    else:
        new_data_paths.append(p)
data_paths = new_data_paths
if args.test is True:
    data_paths = data_paths[:2]


# prepare output paths
from tempfile import TemporaryDirectory
# output_path = TemporaryDirectory().name
output_path = "output"
report_path = join(output_path, f"report.xlsx")
result_path = join(output_path, "results")
os.makedirs(result_path, exist_ok=True)




def process(data_path):
    run_args = argparse.Namespace()
    run_args.root_path = os.getcwd()
    run_args.data_path = data_path
    
    # convert length from minutes to seconds
    if args.length is None:
        args.length = 10 if not is_synthetic else 2000
    data_length = args.length * 60 // 2

    data_dir = dirname(data_path)
    # for synthetic dataset
    if "rca_" in data_path:
        # dataset = basename(dirname(dirname(dirname(dirname(data_path)))))
        case = basename(dirname(data_path))

        service = "SIM"
        metric = None
        with open(join(data_dir, "root_cause.txt")) as f:
            metric = f.read().splitlines()[0]

        # read inject_time
        with open(join(data_dir, "inject_time.txt")) as f:
            inject_time = int(f.readlines()[0].strip()) + args.tdelta

        with open(join(data_dir, "fe_service.txt")) as f:
            sli = f.readlines()[0].strip()

        sli = "SIM_" + sli
    # for real world dataset
    else:
        # dataset = basename(dirname(dirname(dirname(data_path))))
        service, metric = basename(dirname(dirname(data_path))).split("_")
        case = basename(dirname(data_path))

    rp = join(result_path, f"{service}_{metric}_{case}.json")

    # == Load and Preprocess data ==
    data = pd.read_csv(data_path)
    if "time.1" in data:
        data = data.drop(columns=["time.1"])

    if "rca_" in data_path:
        data.columns = ["SIM_" + c for c in data.columns]

    if "time" not in data:
        data["time"] = data.index

    if "sock-shop" in data_path:
        data = data.loc[:, ~data.columns.str.endswith("_lat_50")]
        data = data.loc[:, ~data.columns.str.endswith("_lat_99")]

    if "train-ticket" in data_path:
        time_col = data["time"]
        data = data.loc[:, data.columns.str.startswith("ts-")]
        data["time"] = time_col

    # handle inf
    data = data.replace([np.inf, -np.inf], np.nan)

    # handle na
    data = data.fillna(method="ffill")
    data = data.fillna(0)

    cut_length = 0
    if "rca_" in data_dir:
        normal_df = data[data["time"] < inject_time].tail(data_length)
        anomal_df = data[data["time"] >= inject_time].head(data_length)
        cut_length = min(normal_df.time) - min(data.time)
        data = pd.concat([normal_df, anomal_df], ignore_index=True)
    else:
        with open(join(data_dir, "inject_time.txt")) as f:
            inject_time = int(f.readlines()[0].strip()) + args.tdelta
        normal_df = data[data["time"] < inject_time].tail(data_length)
        anomal_df = data[data["time"] >= inject_time].head(data_length)
        cut_length = min(normal_df.time) - min(data.time)
        data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # num column, exclude time
    num_node = len(data.columns) - 1

    # select sli for certain methods
    if "my-sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_lat_90" in data:
            sli = f"{service}_lat_90"
    elif "train-ticket" in data_path:
        sli = "ts-ui-dashboard_latency-90"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "online-boutique" in data_path:
        sli = "frontend_latency-90"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"

    # == PROCESS ==
    func = globals()[args.method]

    try:
        st = datetime.now()
        out = func(
            data,
            inject_time,
            dataset=args.dataset,
            anomalies=None,
            dk_select_useful=False,
            sli=sli,
            verbose=False,
            n_iter=num_node,
            args=run_args,
        )
        root_causes = out.get("ranks")
        dump_json(filename=rp, data={0: root_causes})
    except Exception as e:
        raise e
        print(f"{args.method=} failed on {data_path=}")
        print(e)
        rp = join(result_path, f"{service}_{metric}_{case}_failed.json")
        with open(rp, "w") as f:
            json.dump({"error": str(e)}, f)


start_time = datetime.now()

for data_path in tqdm(sorted(data_paths)):
    process(data_path)

end_time = datetime.now()
time_taken = end_time - start_time
avg_speed = round(time_taken.total_seconds() / len(data_paths), 2)


# ======== EVALUTION ===========
rps = glob.glob(join(result_path, "*.json"))
services = sorted(list(set([basename(x).split("_")[0] for x in rps])))
faults = sorted(list(set([basename(x).split("_")[1] for x in rps])))

eval_data = {
    "service-fault": [],
    "top_1_service": [],
    "top_3_service": [],
    "top_5_service": [],
    "avg@5_service": [],
    "top_1_metric": [],
    "top_3_metric": [],
    "top_5_metric": [],
    "avg@5_metric": [],
}

s_evaluator_all = Evaluator()
f_evaluator_all = Evaluator()
s_evaluator_cpu = Evaluator()
f_evaluator_cpu = Evaluator()
s_evaluator_mem = Evaluator()
f_evaluator_mem = Evaluator()
s_evaluator_lat = Evaluator()
f_evaluator_lat = Evaluator()
s_evaluator_loss = Evaluator()
f_evaluator_loss = Evaluator()
s_evaluator_io = Evaluator()
f_evaluator_io = Evaluator()

for service in services:
    for fault in faults:
        s_evaluator = Evaluator()
        f_evaluator = Evaluator()

        for rp in rps:
            s, m = basename(rp).split("_")[:2]
            if s != service or m != fault:
                continue  # ignore

            data = load_json(rp)
            if "error" in data:
                continue  # ignore

            for i, ranks in data.items():
                s_ranks = [Node(x.split("_")[0].replace("-db", ""), "unknown") for x in ranks]
                # remove duplication
                old_s_ranks = s_ranks.copy()
                s_ranks = (
                    [old_s_ranks[0]]
                    + [
                        old_s_ranks[i]
                        for i in range(1, len(old_s_ranks))
                        if old_s_ranks[i] not in old_s_ranks[:i]
                    ]
                    if old_s_ranks
                    else []
                )

                f_ranks = [Node(x.split("_")[0], x.split("_")[1]) for x in ranks]

                s_evaluator.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                f_evaluator.add_case(ranks=f_ranks, answer=Node(service, fault))

                if fault == "cpu":
                    s_evaluator_cpu.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_cpu.add_case(ranks=f_ranks, answer=Node(service, fault))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, fault))

                elif fault == "mem":
                    s_evaluator_mem.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_mem.add_case(ranks=f_ranks, answer=Node(service, fault))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, fault))

                elif fault == "delay":
                    s_evaluator_lat.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_lat.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                elif fault == "loss":
                    s_evaluator_loss.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_loss.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                elif fault == "disk":
                    s_evaluator_io.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_io.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                if is_synthetic:
                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, fault))
        eval_data["service-fault"].append(f"{service}_{fault}")
        eval_data["top_1_service"].append(s_evaluator.accuracy(1))
        eval_data["top_3_service"].append(s_evaluator.accuracy(3))
        eval_data["top_5_service"].append(s_evaluator.accuracy(5))
        eval_data["avg@5_service"].append(s_evaluator.average(5))
        eval_data["top_1_metric"].append(f_evaluator.accuracy(1))
        eval_data["top_3_metric"].append(f_evaluator.accuracy(3))
        eval_data["top_5_metric"].append(f_evaluator.accuracy(5))
        eval_data["avg@5_metric"].append(f_evaluator.average(5))


print("--- Evaluation results ---")
if is_synthetic:
    print("Avg@5:", round(f_evaluator_all.average(5), 2))
else: # for real datasets
    for name, s_evaluator, f_evaluator in [
        ("cpu", s_evaluator_cpu, f_evaluator_cpu),
        ("mem", s_evaluator_mem, f_evaluator_mem),
        ("io", s_evaluator_io, f_evaluator_io),
        ("delay", s_evaluator_lat, f_evaluator_lat),
        ("loss", s_evaluator_loss, f_evaluator_loss),
    ]:
        eval_data["service-fault"].append(f"overall_{name}")
        eval_data["top_1_service"].append(s_evaluator.accuracy(1))
        eval_data["top_3_service"].append(s_evaluator.accuracy(3))
        eval_data["top_5_service"].append(s_evaluator.accuracy(5))
        eval_data["avg@5_service"].append(s_evaluator.average(5))
        eval_data["top_1_metric"].append(f_evaluator.accuracy(1))
        eval_data["top_3_metric"].append(f_evaluator.accuracy(3))
        eval_data["top_5_metric"].append(f_evaluator.accuracy(5))
        eval_data["avg@5_metric"].append(f_evaluator.average(5))

        if name == "io":
            name = "disk"

        if s_evaluator.average(5) is not None:
            print( f"Avg@5-{name.upper()}:".ljust(12), round(s_evaluator.average(5), 2))


print("---")
print("Avg speed:", avg_speed)

