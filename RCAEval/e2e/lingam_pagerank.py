import warnings

import numpy as np

from RCAEval.graph_construction.lingam import DirectLiNGAM, ICALiNGAM
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)
from RCAEval.e2e import rca

warnings.filterwarnings("ignore")


@rca
def lingam_pagerank(data, inject_time=None, dataset=None, num_loop=None, sli=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    model = ICALiNGAM()
    model.fit(data.to_numpy().astype(float))
    adj = model.adjacency_matrix_
    adj = adj.astype(bool).astype(int)

    # check if adj values are all 0
    if len(adj) == 0 or adj.sum().sum() == 0:
        return {
            "adj": adj,
            "node_names": node_names,
            "ranks": node_names, 
        }

    ranks = page_rank(adj, node_names=node_names)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


def micro_diag(data, inject_time=None, dataset=None, num_loop=None, sli=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # DirectLiNGAM + Pagerank
    node_names = data.columns.to_list()
    model = DirectLiNGAM()
    model.fit(data.to_numpy().astype(float))
    # print(model.causal_order_)
    adj = model.adjacency_matrix_
    adj = adj.astype(bool).astype(int)

    # check if adj values are all 0
    if adj.sum().sum() == 0:
        return {
            "adj": adj,
            "node_names": node_names,
            # "ranks": [],
            "ranks": node_names,  # hmm should we return empty or random?
        }

    ranks = page_rank(adj, node_names=node_names)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
