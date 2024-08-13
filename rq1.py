import argparse
import glob
import math
import itertools
import json
import os
import pickle
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from os.path import abspath, basename, dirname, exists, join

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.score.LocalScoreFunction import local_score_BIC
from tqdm import tqdm

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.benchmark.metrics import F1, SHD, F1_Skeleton
from RCAEval.classes.graph import MemoryGraph, Node
from RCAEval.graph_heads import finalize_directed_adj
from RCAEval.io.time_series import drop_constant, drop_extra, drop_time
from RCAEval.utility import (
    dump_json,
    download_syn_rcd_dataset,
    download_syn_circa_dataset,
    download_syn_causil_dataset,
    is_py310,
    load_json,
)

if is_py310():
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.search.FCMBased.lingam import DirectLiNGAM, ICALiNGAM, VARLiNGAM
    from causallearn.search.ScoreBased.GES import ges
    from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
    from RCAEval.graph_construction.granger import granger
    from RCAEval.graph_construction.pcmci import pcmci
    from RCAEval.graph_construction.cmlp import cmlp
    try:
        from RCAEval.graph_construction.dag_gnn import dag_gnn
        from RCAEval.graph_construction.dag_gnn import notears_low_rank as ntlr
        from RCAEval.graph_construction.notears import notears
    except Exception as e:
        print(e)
    
else:
    from RCAEval.graph_construction.fges import fges

AVAILABLE_METHODS = sorted(
    [
        "pc",
        "ppc",
        "pcmci",
        "fci",
        "fges",
        "notears",
        "ntlr",
        "DirectLiNGAM",
        "VARLiNGAM",
        "ICALiNGAM",
        "ges",
        "granger",
    ]
)


def adj2generalgraph(adj):
    G = GeneralGraph(nodes=[f"X{i + 1}" for i in range(len(adj))])
    for row_idx in range(len(adj)):
        for col_idx in range(len(adj)):
            if adj[row_idx, col_idx] == 1:
                G.add_directed_edge(f"X{col_idx + 1}", f"X{row_idx + 1}")
    return G


def score_g(Data, G, parameters=None):
    # calculate the score for the current G
    # here G is a DAG
    parameters = {"lambda_value": 0}

    score = 0
    for i, node in enumerate(G.get_nodes()):
        PA = G.get_parents(node)

        # for granger
        if len(PA) > 0 and isinstance(PA[0], str):
            pass  # already in str format
        else:
            PA = [p.name for p in PA]

        if len(PA) > 0 and isinstance(PA[0], str):
            # this is for FCI, bc it doesn't have node_names param
            # remove X from list ['X6', 'X10']
            PA = [int(p[1:]) - 1 for p in PA]


        delta_score = local_score_BIC(Data, i, PA, parameters)

        # delta_score is nan, ignore
        if np.isnan(delta_score):
            continue

        score = score + delta_score
    return score.sum()


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    parser.add_argument("--dataset", type=str, help="Input dataset.")
    parser.add_argument("--output-path", type=str, default="output", help="Output cache.")
    parser.add_argument("--length", type=int, default=10, help="Time series length.")
    parser.add_argument("--model", type=str, default="pc_pagerank", help="func name")

    args = parser.parse_args()

    # check if args.model is defined here
    if args.model not in globals():
        raise ValueError(f"{args.model=} not defined. Available: {AVAILABLE_METHODS}")
    
    # assert dataset
    if args.dataset not in ["circa10", "circa50", "rcd10", "rcd50", "causil10", "causil50"]:
        print(f"{args.dataset=} not defined. Available: circa10, circa50, rcd10, rcd50, causil10, causil50")
        exit()

    return args


args = parse_args()


# download dataset
if "circa" in args.dataset:
    download_syn_circa_dataset()
elif "rcd" in args.dataset:
    download_syn_rcd_dataset()
elif "causil" in args.dataset:
    download_syn_causil_dataset()


# prepare output path
output_path = f"{args.output_path}"
report_path = join(output_path, "report.xlsx")
result_path = join(output_path, "results")
if not exists(args.output_path):
    os.makedirs(args.output_path)
os.makedirs(result_path, exist_ok=True)
# dump_json(filename=join(output_path, "args.json"), data=vars(args))


data_paths = list(glob.glob(os.path.join(args.input_path, "**/data.csv"), recursive=True))

def evaluate():
    eval_data = {
        "Case": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": [],
        "Precision-Skel": [],
        "Recall-Skel": [],
        "F1-Skel": [],
        "BIC": [],
        "SHD": [],
    }

    for data_path in data_paths:
        if "circa" in data_path or "rcd" in data_path:
            num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
            graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
            case_idx = int(basename(dirname(data_path)))
        elif "causil" in data_path:
            graph_idx = int(basename(dirname(data_path))[-1:])
            case_idx = 0

        # ===== READ RESULT =====
        est_graph_name = f"{graph_idx}_{case_idx}_est_graph.json"
        est_graph_path = join(result_path, est_graph_name)

        if not exists(est_graph_path):
            continue
        est_graph = MemoryGraph.load(est_graph_path)

        # ====== READ TRUE GRAPH =====
        if "circa" in data_path:
            true_graph_path = join(dirname(dirname(dirname(data_path))), "graph.json")
            true_graph = MemoryGraph.load(true_graph_path)

        elif "causil" in data_path:
            dag_gt = pickle.load(open(join(dirname(data_path), "DAG.gpickle"), "rb"))
            true_graph = MemoryGraph(dag_gt)

        elif "rcd" in data_path:
            dag_gt = pickle.load(
                open(join(dirname(dirname(dirname(data_path))), "g_graph.pkl"), "rb")
            )
            true_graph = MemoryGraph(dag_gt)
            true_graph = MemoryGraph.load(
                join(dirname(dirname(dirname(data_path))), "true_graph.json")
            )

        e = F1(true_graph, est_graph)
        e_skel = F1_Skeleton(true_graph, est_graph)
        shd = SHD(true_graph, est_graph)

        eval_data["Case"].append(est_graph_name)
        eval_data["Precision"].append(e["precision"])
        eval_data["Recall"].append(e["recall"])
        eval_data["F1-Score"].append(e["f1"])
        eval_data["Precision-Skel"].append(e_skel["precision"])
        eval_data["Recall-Skel"].append(e_skel["recall"])
        eval_data["F1-Skel"].append(e_skel["f1"])
        eval_data["SHD"].append(shd)

    avg_precision = np.mean(eval_data["Precision"])
    avg_recall = np.mean(eval_data["Recall"])
    avg_f1 = np.mean(eval_data["F1-Score"])
    avg_precision_skel = np.mean(eval_data["Precision-Skel"])
    avg_recall_skel = np.mean(eval_data["Recall-Skel"])
    avg_f1_skel = np.mean(eval_data["F1-Skel"])

    avg_shd = np.mean(eval_data["SHD"])
    avg_bic = np.mean(eval_data["BIC"])

    eval_data["Case"].insert(0, "Average")
    eval_data["Precision"].insert(0, avg_precision)
    eval_data["Recall"].insert(0, avg_recall)
    eval_data["F1-Score"].insert(0, avg_f1)
    eval_data["Precision-Skel"].insert(0, avg_precision_skel)
    eval_data["Recall-Skel"].insert(0, avg_recall_skel)
    eval_data["F1-Skel"].insert(0, avg_f1_skel)
    eval_data["BIC"].insert(0, avg_bic)
    eval_data["SHD"].insert(0, avg_shd)

    print(f"F1:   {avg_f1:.2f}")
    print(f"F1-S: {avg_f1_skel:.2f}")
    print(f"SHD:  {math.floor(avg_shd)}")

    report_df = pd.DataFrame(eval_data)
    report_df.to_excel(report_path, index=False)


def process(data_path):
    if "circa" in data_path:
        num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
        graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
        case_idx = int(basename(dirname(data_path)))
    elif "causil" in data_path:
        num_node = int(basename(dirname(dirname(dirname(data_path)))).split("_")[0])
        graph_idx = int(basename(dirname(data_path))[-1:])
        case_idx = 0
    elif "rcd" in data_path:
        num_node = int(basename(dirname(dirname(dirname(dirname(data_path))))))
        graph_idx = int(basename(dirname(dirname(dirname(data_path)))))
        case_idx = int(basename(dirname(data_path)))

    if "circa" in data_path:  
        data = pd.read_csv(data_path, header=None)
        data.header = list(map(str, range(0, data.shape[1])))
    else:
        data = pd.read_csv(data_path)

    # == PROCESS ==
    data = data.fillna(method="ffill")
    data = data.fillna(value=0)
    np_data = np.absolute(data.to_numpy().astype(float))

    if args.length is not None:
        np_data = np_data[: args.length, :]

    adj = []
    G = None

    st = datetime.now()
    try:
        if args.model == "pc":
            indep_test = fisherz 
            adj = pc(
                np_data,
                alpha=args.alpha,
                indep_test=indep_test,
                stable=args.stable,
                show_progress=False,
            ).G.graph

        elif args.model == "fci":
            indep_test = fisherz
            adj = fci(
                np_data,
                independence_test_method=indep_test,
                alpha=args.alpha,
                show_progress=False,
                verbose=False,
            )[0].graph
        elif args.model == "fges":
            adj = fges(pd.DataFrame(np_data))
        elif args.model == "ICALiNGAM":
            model = ICALiNGAM()
            model.fit(np_data)
            adj = model.adjacency_matrix_
            adj = adj.astype(bool).astype(int)
        elif args.model == "DirectLiNGAM":
            model = DirectLiNGAM()
            model.fit(np_data)
            adj = model.adjacency_matrix_
            adj = adj.astype(bool).astype(int)
        elif args.model == "ges":
            record = ges(np_data)
            adj = record["G"].graph
        elif args.model == "granger":
            # assert args.test in [None, "ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"]
            # adj = granger(data, test=args.test, maxlag=args.tau, p_val_threshold=args.alpha)
            adj = granger(data)
        elif args.model == "pcmci":
            adj = pcmci(pd.DataFrame(np_data))
        elif args.model == "cmlp":
            adj = cmlp(pd.DataFrame(np_data))
        elif args.model == "notears":
            adj = notears(pd.DataFrame(np_data))
        elif args.model == "ntlr":
            adj = ntlr(pd.DataFrame(np_data))
        elif args.model == "dag_gnn":
            adj = dag_gnn(pd.DataFrame(np_data))
        else:
            raise ValueError(f"{args.model=} not defined. Available: {AVAILABLE_METHODS}")

        if "circa" in data_path:
            est_graph = MemoryGraph.from_adj(
                adj, nodes=[Node("SIM", str(i)) for i in range(len(adj))]
            )
        else:
            est_graph = MemoryGraph.from_adj(adj, nodes=data.columns.to_list())

        est_graph.dump(join(result_path, f"{graph_idx}_{case_idx}_est_graph.json"))
    except Exception as e:
        raise e
        print(f"{args.model=} failed on {data_path=}")
        est_graph = MemoryGraph.from_adj([], nodes=[])
        est_graph.dump(join(result_path, f"{graph_idx}_{case_idx}_failed.json"))

    with open(join(result_path, f"{graph_idx}_{case_idx}_time_taken.txt"), "w") as f:
        tt = datetime.now() - st
        tt = tt - timedelta(microseconds=tt.microseconds)

        f.write(f"Time taken: {tt}")


start_time = datetime.now()

for data_path in tqdm(data_paths):
    output = process(data_path)

end_time = datetime.now()
time_taken = end_time - start_time
time_taken = time_taken - timedelta(microseconds=time_taken.microseconds)
with open(join(output_path, "time_taken.txt"), "w") as f:
    s = f"Time taken: {time_taken}"
    print(s)
    f.write(s)


evaluate()
