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
    is_py38,
    is_py312,
    is_py314,
    load_json,
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
    download_re1_dataset,
    download_re2ob_dataset,
    download_re2ss_dataset,
    download_re2tt_dataset,
    download_re3_dataset,
    download_eventadl_dataset,
)


if is_py312() or is_py314():
    import RCAEval.e2e as e2e

    # bind every method that could be imported; one whose dependencies are
    # not installed (e.g. torch in the minimal EventADL environment) is
    # simply unavailable and rejected by parse_args with a clear error
    for _method in [
        "baro",
        "causalrca",
        "circa",
        "cloudranger",
        "cmlp_pagerank",
        "dummy",
        "e_diagnosis",
        "easyrca",
        "fci_pagerank",
        "fci_randomwalk",
        "ges_pagerank",
        "granger_pagerank",
        "granger_randomwalk",
        "lingam_pagerank",
        "lingam_randomwalk",
        "micro_diag",
        "microcause",
        "microrank",
        "mmbaro",
        "mmcirca",
        "mmnsigma",
        "mmrcd",
        "mscred",
        "nsigma",
        "rcd",
        "ntlr_pagerank",
        "ntlr_randomwalk",
        "pc_pagerank",
        "pc_randomwalk",
        "run",
        "torai",
        "tracerca",
        "eventadl",
    ]:
        if hasattr(e2e, _method):
            globals()[_method] = getattr(e2e, _method)

elif is_py38():
    from RCAEval.e2e import dummy, e_diagnosis, ht, rcd, mmrcd, torai
else:
    print("Please use Python 3.8, 3.12, or 3.14")
    exit(1)

try:
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from RCAEval.e2e.causalrca import causalrca
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    parser.add_argument("--method", type=str, help="Choose a method.")
    parser.add_argument("--dataset", type=str, help="Choose a dataset.", choices=[
        "online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket",
        "re1-ob", "re1-ss", "re1-tt", "re2-ob", "re2-ss", "re2-tt", "re3-ob", "re3-ss", "re3-tt",
        "torai-ob", "torai-ss", "torai-tt",
        "eventadl-falcon", "eventadl-flask", "eventadl-live",
    ])
    parser.add_argument("--length", type=int, default=20, help="Time series length (RQ4)")
    parser.add_argument("--tdelta", type=int, default=0, help="Specify $t_delta$ to simulate delay in anomaly detection")
    parser.add_argument("--test", action="store_true", help="Perform smoke test on certain methods without fully run on all data")
    parser.add_argument("--report-chance", action="store_true", help="Also print the Avg@5 a random ranking would reach, and the lift over it")
    args = parser.parse_args()

    # checked before the globals() lookup so the message also appears when the
    # method could not even be imported (e.g. missing patched causal-learn)
    if args.method in ("rcd", "mmrcd") and not is_py38():
        print(f"{args.method} requires the RCD environment (Python 3.8, `pip install -e .[rcd]`). See docs/SETUP.md.")
        exit(1)

    if args.method not in globals():
        raise ValueError(f"{args.method=} not defined. Please check imported methods.")

    return args


args = parse_args()

# download dataset
if "online-boutique" in args.dataset or "re1-ob" in args.dataset:
    download_online_boutique_dataset()
elif "sock-shop-1" in args.dataset:
    download_sock_shop_1_dataset()
elif "sock-shop-2" in args.dataset or "re1-ss" in args.dataset:
    download_sock_shop_2_dataset()
elif "train-ticket" in args.dataset or "re1-tt" in args.dataset:
    download_train_ticket_dataset()
elif "re2-ob" in args.dataset.lower():
    download_re2ob_dataset()
elif "re2-ss" in args.dataset.lower():
    download_re2ss_dataset()
elif "re2-tt" in args.dataset.lower():
    download_re2tt_dataset()
elif "re3" in args.dataset:
    download_re3_dataset()
elif "torai" in args.dataset:
    pass  # torai data is expected to be local
elif "eventadl" in args.dataset:
    download_eventadl_dataset(name=args.dataset.replace("eventadl-", ""))
else:
    raise Exception(f"{args.dataset} is not defined!")

DATASET_MAP = {
    "online-boutique": "data/online-boutique",
    "sock-shop-1": "data/sock-shop-1",
    "sock-shop-2": "data/sock-shop-2",
    "train-ticket": "data/train-ticket",
    "re1-ob": "data/online-boutique",
    "re1-ss": "data/sock-shop-2",
    "re1-tt": "data/train-ticket",
    "re2-ob": "data/RE2/RE2-OB",
    "re2-ss": "data/RE2/RE2-SS",
    "re2-tt": "data/RE2/RE2-TT",
    "re3-ob": "data/RE3/RE3-OB",
    "re3-ss": "data/RE3/RE3-SS",
    "re3-tt": "data/RE3/RE3-TT",
    "torai-ob": "data/torai-OB",
    "torai-ss": "data/torai-SS",
    "torai-tt": "data/torai-TT",
    "eventadl-falcon": "data/eventadl-falcon",
    "eventadl-flask": "data/eventadl-flask",
    "eventadl-live": "data/eventadl-live",
}
dataset = DATASET_MAP[args.dataset]


# prepare input paths
if "eventadl" in args.dataset:
    # EventADL test cases are event-log based (rca.json + events/{id}.json),
    # not the metrics.csv-per-case layout the other datasets use.
    data_paths = load_json(os.path.join(dataset, "rca.json"))["test_cases"]
else:
    data_paths = list(glob.glob(os.path.join(dataset, "**/data.csv"), recursive=True))
    if not data_paths:
        data_paths = list(glob.glob(os.path.join(dataset, "**/simple_metrics.csv"), recursive=True))
# new_data_paths = []
# for p in data_paths:
#     if os.path.exists(p.replace("data.csv", "simple_data.csv")):
#         new_data_paths.append(p.replace("data.csv", "simple_data.csv"))
#     elif os.path.exists(p.replace("data.csv", "simple_metrics.csv")):
#         new_data_paths.append(p.replace("data.csv", "simple_metrics.csv"))
#     else:
#         new_data_paths.append(p)
# data_paths = new_data_paths
if args.test is True:
    data_paths = data_paths[:2]


# prepare output paths
from tempfile import TemporaryDirectory
# output_path = TemporaryDirectory().name
output_path = "output"
report_path = join(output_path, f"report.xlsx")
result_path = join(output_path, "results")
os.makedirs(result_path, exist_ok=True)

if "eventadl" in args.dataset:
    # the eventadl datasets share service names and case ids, so stale result
    # files from a previous run on another eventadl dataset would collide with
    # (and leak into) this run's evaluation
    for _rp in glob.glob(join(result_path, "*_event_*.json")):
        os.remove(_rp)


def _eventadl_short_name(entity):
    """Turn a CloudTrail actor/resource identifier (ARN or similar) into a
    short label, so it fits the "{service}_{fault}_{case}.json" result
    filename convention shared with every other dataset."""
    return entity.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]


def process_eventadl(test_case):
    case_id = test_case["id"]
    ground_truth = test_case["ground_truth"]
    service = _eventadl_short_name(ground_truth)

    rp = join(result_path, f"{service}_event_{case_id}.json")

    with open(join(dataset, "events", f"{case_id}.json")) as f:
        events = json.load(f)

    func = globals()[args.method]
    try:
        out = func({"events": events}, inject_time=None, dataset=args.dataset)
        root_causes = out.get("ranks")
        dump_json(filename=rp, data={"0": root_causes, "ground_truth": ground_truth})
    except Exception as e:
        raise e


def process(data_path):
    if "eventadl" in args.dataset:
        return process_eventadl(data_path)

    run_args = argparse.Namespace()
    run_args.root_path = os.getcwd()
    run_args.data_path = data_path
    
    # convert length from minutes to seconds
    if args.length is None:
        args.length = 10
    data_length = args.length * 60 // 2

    data_dir = dirname(data_path)

    service, metric = basename(dirname(dirname(data_path))).split("_")
    case = basename(dirname(data_path))

    rp = join(result_path, f"{service}_{metric}_{case}.json")

    # == Load and Preprocess data ==
    data = pd.read_csv(data_path)
    
    # remove lat-50, only selecte lat-90 
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    
    if "mm-tt" in data_path or "torai-TT" in data_path:
        time_col = data["time"]
        data = data.loc[:, data.columns.str.startswith("ts-")]
        data["time"] = time_col
        
    # handle inf
    data = data.replace([np.inf, -np.inf], np.nan)

    # handle na
    data = data.ffill()
    data = data.fillna(0)

    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip()) + args.tdelta
    # for metrics, minutes -> seconds // 2
    normal_df = data[data["time"] < inject_time].tail(args.length * 60 // 2)
    anomal_df = data[data["time"] >= inject_time].head(args.length * 60 // 2)

    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # num column, exclude time
    num_node = len(data.columns) - 1

    # rename latency
    data = data.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in data.columns
            if c.endswith("_latency-90")
        }
    )
    
    # == Get SLI ===
    sli = None
    if "my-sock-shop" in data_path or "fse-ss" in data_path:
        sli = "front-end_cpu"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_lat_90" in data:
            sli = f"{service}_lat_90"
    elif "train-ticket" in data_path or "fse-tt" in data_path or "RE2-TT" in data_path or "RE3-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "online-boutique" in data_path or "fse-ob" in data_path or "RE2-OB" in data_path or "RE2-SS" in data_path or "RE3-OB" in data_path or "RE3-SS" in data_path:
        sli = "frontend_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
        elif "frontend_1" in data:
            sli = "frontend_1"
    elif "torai-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
    elif "torai-OB" in data_path or "torai-SS" in data_path:
        sli = "frontend_latency"
    else:
        raise ValueError("SLI not implemented")

    # == Multi-source data loading (multi-source methods and torai) ==
    MM_METHODS = ("mmbaro", "mmnsigma", "mmrcd", "mmcirca")
    if args.method in MM_METHODS or ("torai" in args.dataset and args.method == "torai"):
        logts = pd.read_csv(os.path.join(data_dir, "logts.csv"))

        traces_err = pd.DataFrame()
        traces_lat = pd.DataFrame()
        try:
            traces_err = pd.read_csv(os.path.join(data_dir, "tracets_err.csv"))
            traces_lat = pd.read_csv(os.path.join(data_dir, "tracets_lat.csv"))
        except FileNotFoundError:
            pass

        # window logts and trace ts to match metric window
        logts_length = args.length * 4 // 2 if args.length else 20
        a = logts[logts["time"] < inject_time].tail(logts_length)
        b = logts[logts["time"] >= inject_time].head(logts_length)
        logts = pd.concat([a, b], ignore_index=True)

        if traces_err.shape[0] > 0:
            a = traces_err[traces_err["time"] < inject_time].tail(logts_length)
            b = traces_err[traces_err["time"] >= inject_time].head(logts_length)
            traces_err = pd.concat([a, b], ignore_index=True)

        if traces_lat.shape[0] > 0:
            a = traces_lat[traces_lat["time"] < inject_time].tail(logts_length)
            b = traces_lat[traces_lat["time"] >= inject_time].head(logts_length)
            traces_lat = pd.concat([a, b], ignore_index=True)

        data = {
            "metric": data.copy(deep=True) if isinstance(data, pd.DataFrame) else data,
            "logts": logts,
            "tracets_err": traces_err,
            "tracets_lat": traces_lat,
        }

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
        # print("==============")
        # print(f"{data_path=}")
        # print(root_causes[:5])
        dump_json(filename=rp, data={0: root_causes})
    except Exception as e:
        raise e
        print(f"{args.method=} failed on {data_path=}")
        print(e)
        rp = join(result_path, f"{service}_{metric}_{case}_failed.json")
        with open(rp, "w") as f:
            json.dump({"error": str(e)}, f)


start_time = datetime.now()

sort_key = (lambda x: x["id"]) if "eventadl" in args.dataset else None
for data_path in tqdm(sorted(data_paths, key=sort_key)):
    process(data_path)

end_time = datetime.now()
time_taken = end_time - start_time
avg_speed = round(time_taken.total_seconds() / len(data_paths), 2)


# ======== EVALUTION ===========
rps = glob.glob(join(result_path, "*.json"))

if "eventadl" in args.dataset:
    # EventADL result files only, keyed on the literal "event" fault token
    # written by process_eventadl(); the physical cpu/mem/io/socket/delay/loss
    # evaluation below doesn't apply and is skipped entirely.
    rps = [rp for rp in rps if basename(rp).split("_")[1] == "event"]

    s_evaluator_event = Evaluator()

    for rp in rps:
        data = load_json(rp)
        if "error" in data:
            continue  # ignore

        ground_truth = data["ground_truth"]
        ranks = data["0"]
        answer = Node(ground_truth, "unknown")
        event_ranks = [Node(x, "unknown") for x in ranks]
        s_evaluator_event.add_case(ranks=event_ranks, answer=answer)

    print("--- Evaluation results ---")
    if s_evaluator_event.average(5) is not None:
        print("AC1:".ljust(12), round(s_evaluator_event.accuracy(1), 2))
        print("AC3:".ljust(12), round(s_evaluator_event.accuracy(3), 2))
        print("AC5:".ljust(12), round(s_evaluator_event.accuracy(5), 2))
        print("Avg@5:".ljust(12), round(s_evaluator_event.average(5), 2))
        if args.report_chance:
            print("Chance@5:".ljust(12), round(s_evaluator_event.chance_average(5), 2))
            print("Lift@5:".ljust(12), round(s_evaluator_event.lift(5), 2))
    print("---")
    print("Avg speed:", avg_speed)
    exit(0)

# non-eventadl result files only; stale eventadl result files from a
# previous run must not leak into this generic per-service/fault evaluation
rps = [rp for rp in rps if basename(rp).split("_")[1] != "event"]
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
s_evaluator_socket = Evaluator()
f_evaluator_socket = Evaluator()

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

                f_ranks = [Node(x.split("_")[0], x.split("_")[1] if "_" in x else "unknown") for x in ranks]

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
                    f_evaluator_io.add_case(ranks=f_ranks, answer=Node(service, "diskio"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "diskio"))
                elif fault == "socket":
                    s_evaluator_socket.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_socket.add_case(ranks=f_ranks, answer=Node(service, "socket"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "socket"))


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
for name, s_evaluator, f_evaluator in [
    ("cpu", s_evaluator_cpu, f_evaluator_cpu),
    ("mem", s_evaluator_mem, f_evaluator_mem),
    ("io", s_evaluator_io, f_evaluator_io),
    ("socket", s_evaluator_socket, f_evaluator_socket),
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
        if args.report_chance:
            print(f"Chance@5-{name.upper()}:".ljust(12), round(s_evaluator.chance_average(5), 2))
            print(f"Lift@5-{name.upper()}:".ljust(12), round(s_evaluator.lift(5), 2))

print("---")
print("Avg speed:", avg_speed)

