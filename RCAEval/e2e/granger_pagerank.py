from RCAEval.graph_construction.granger import granger
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    preprocess,
    drop_time,
    select_useful_cols,
)
from RCAEval.e2e import rca


@rca
def granger_pagerank(data, inject_time=None, dataset=None, num_loop=None, sli=None, **kwargs):
    data = preprocess(
        data=data,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()
    adj = granger(data)

    # check if adj values are all 0
    if adj.sum().sum() == 0:
        return {
            "adj": adj,
            "node_names": node_names,
            "ranks": node_names,
        }

    ranks = page_rank(adj, node_names=data.columns.to_list())
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
