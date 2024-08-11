from cfm.graph_construction.granger import granger
from cfm.graph_heads.page_rank import page_rank
from cfm.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    preprocess,
    drop_time,
    select_useful_cols,
)


def granger_pagerank(data, inject_time=None, dataset=None, num_loop=None, sli=None, **kwargs):
    data = preprocess(
        data=data,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False)
    )


    node_names = data.columns.to_list()
    
    adj = []
    try:
        adj = granger(data)
    except Exception as e:
        print(e)
        return {
            "adj": adj,
            "node_names": node_names,
            "ranks": node_names,
        }



    # check if adj values are all 0
    if adj.sum().sum() == 0:
        return {
            "adj": adj,
            "node_names": node_names,
            # "ranks": [],
            "ranks": node_names,  # hmm should we return empty or random?
        }

    ranks = page_rank(adj, node_names=data.columns.to_list())
    # return ranks
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
