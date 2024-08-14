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
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
)

download_sock_shop_1_dataset()
download_sock_shop_2_dataset()
download_train_ticket_dataset()


if is_py310():
    from RCAEval.e2e import ( 
        # here
        # causalai,
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
    print("causalrca not available")



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
    # for data
    parser.add_argument("-i", "--input-path", type=str, default="data", help="path to data")
    parser.add_argument(
        "-o", "--output-path", type=str, default="output", help="for results and reports"
    )
    parser.add_argument("--length", type=int, default=2000, help="data points, in minutes")
    parser.add_argument("--tbias", type=int, default=0)

    # for method
    parser.add_argument("-m", "--model", type=str, default="pc_pagerank", help="func name")

    # for evaluation
    parser.add_argument("-w", "--worker-num", type=int, default=1, help="number of workers")
    parser.add_argument("--iter-num", type=int, default=1)

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # check if args.model is defined here
    if args.model not in globals():
        raise ValueError(f"{args.model=} not defined. Available: {AVAILABLE_METHODS}")

    # check if args.model is defined here
    if basename(args.input_path) not in [
        "5",
        "10",
        "50",
        "online-boutique",
        "sock-shop",
        "sock-shop-1",
        "sock-shop-2",
        "train-ticket",
    ]:
        raise ValueError(
            f"{args.input_path=} should be data/<real-dataset> or data/<synthetic-dataset>/<num-node>"
        )

    if args.verbose:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
    return args


args = parse_args()


# ==== PREPARE PATHS ====
art_name = f"{basename(args.input_path)}_{args.model.replace('_', '-')}_l{args.length}_i{args.iter_num}"

output_path = f"{args.output_path}/{art_name}"

report_path = join(output_path, f"{art_name}.xlsx")
result_path = join(output_path, "results")

if not exists(args.output_path):
    os.makedirs(args.output_path)

if exists(result_path):
    shutil.rmtree(result_path)

os.makedirs(result_path, exist_ok=True)

dump_json(filename=join(output_path, "args.json"), data=vars(args))


# ==== PROCESS TO GENERATE JSON ====
data_paths = list(glob.glob(os.path.join(args.input_path, "**/data.csv"), recursive=True))
new_data_paths = []
for p in data_paths: 
    if os.path.exists(p.replace("data.csv", "simple_data.csv")):
        new_data_paths.append(p.replace("data.csv", "simple_data.csv"))
    else:
        new_data_paths.append(p)
data_paths = new_data_paths


def process(data_path):
    # for RUN method
    run_args = argparse.Namespace()
    run_args.root_path = os.getcwd()
    run_args.data_path = data_path
    
    # convert length from minutes to seconds
    data_length = args.length * 60 // 2

    data_dir = dirname(data_path)
    # for synthetic dataset
    if "rca_" in data_path:
        dataset = basename(dirname(dirname(dirname(dirname(data_path)))))
        case = basename(dirname(data_path))

        service = "SIM"
        metric = None
        with open(join(data_dir, "root_cause.txt")) as f:
            metric = f.read().splitlines()[0]

        # read inject_time
        with open(join(data_dir, "inject_time.txt")) as f:
            inject_time = int(f.readlines()[0].strip()) + args.tbias

        with open(join(data_dir, "fe_service.txt")) as f:
            sli = f.readlines()[0].strip()

        sli = "SIM_" + sli
    # for real world dataset
    else:
        dataset = basename(dirname(dirname(dirname(data_path))))
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

    # check if there is any nan or inf
    if data.isnull().values.any():
        print(f"{data_path=} has nan")

    if data.isin([np.inf, -np.inf]).values.any():
        print(f"{data_path=} has inf")

    cut_length = 0
    if "rca_" in data_dir:
        normal_df = data[data["time"] < inject_time].tail(data_length)
        anomal_df = data[data["time"] >= inject_time].head(data_length)

        cut_length = min(normal_df.time) - min(data.time)

        data = pd.concat([normal_df, anomal_df], ignore_index=True)
    else:
        with open(join(data_dir, "inject_time.txt")) as f:
            inject_time = int(f.readlines()[0].strip()) + args.tbias
        normal_df = data[data["time"] < inject_time].tail(data_length)
        anomal_df = data[data["time"] >= inject_time].head(data_length)

        cut_length = min(normal_df.time) - min(data.time)

        data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # num column, exclude time
    num_node = len(data.columns) - 1

    # == Get SLI ===
    if "my-sock-shop" in data_path or "fse-ss" in data_path:
        sli = "front-end_cpu"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_lat_90" in data:
            sli = f"{service}_lat_90"
    elif "train-ticket" in data_path or "fse-tt" in data_path:
        sli = "ts-ui-dashboard_latency-90"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "online-boutique" in data_path or "fse-ob" in data_path:
        sli = "frontend_latency-90"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"

    # == PROCESS ==
    func = globals()[args.model]

    ranks_dict = {}
    out = None

    
    try:
        for i in range(args.iter_num):
            st = datetime.now()
            out = func(
                data,
                inject_time,
                dataset=dataset,
                anomalies=None,
                dk_select_useful=False,
                sli=sli,
                verbose=args.verbose,
                n_iter=num_node,
                args=run_args,
            )
            root_causes = out.get("ranks")

            s_ranks = [i.split("_")[0] for i in root_causes]
            service_ranks = (
                [s_ranks[0]]
                + [s_ranks[i] for i in range(1, len(s_ranks)) if s_ranks[i] not in s_ranks[:i]]
                if s_ranks
                else []
            )

            ranks_dict[i] = root_causes

            with open(join(result_path, f"{service}_{metric}_{case}_{i}_time_taken.txt"), "w") as f:
                tt = datetime.now() - st
                tt = tt - timedelta(microseconds=tt.microseconds)
                f.write(f"Time taken: {tt}")

        # == SAVE ==
        dump_json(filename=rp, data=ranks_dict)
    except Exception as e:
        raise e
        print(f"{args.model=} failed on {data_path=}")
        print(e)
        rp = join(result_path, f"{service}_{metric}_{case}_failed.json")

        with open(rp, "w") as f:
            json.dump({"error": str(e)}, f)


start_time = datetime.now()

if args.worker_num > 1:
    with Pool(min(args.worker_num, os.cpu_count() - 2)) as p:
        list(tqdm(p.imap(process, data_paths), total=len(data_paths)))
else:  # single worker
    for data_path in tqdm(sorted(data_paths)):
        process(data_path)

end_time = datetime.now()
time_taken = end_time - start_time
# remove set milliseconds to 0
time_taken = time_taken - timedelta(microseconds=time_taken.microseconds)
with open(join(output_path, "time_taken.txt"), "w") as f:
    s = f"Time taken: {time_taken}"
    f.write(s)


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

                if "rca_" in args.input_path:
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


print("Evaluation results")
if "rca_" in args.input_path:
    print(round(f_evaluator_all.average(5), 2))

else:
#### THIS ONE IS FOR REAL DATASET ####
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
            print( f"s_{name}:", round(s_evaluator.average(5), 2))

    report_df = pd.DataFrame(eval_data)
    report_df.to_excel(report_path, index=False)
