from cfm.graph_construction.fci import fci_default
from cfm.graph_heads.page_rank import page_rank
from cfm.io.time_series import preprocess


def fci_pagerank(data, inject_time=None, dataset=None, dk_select_useful=False, n_iter=10, **kwargs):
    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    node_names = data.columns.to_list()

    adj = []
    try:
        adj = fci_default(data)
        ranks = page_rank(adj, node_names=node_names, n_iter=n_iter)
    except Exception as e:
        print(e)
        return {"adj": [], "node_names": node_names, "ranks": node_names}

    # # check if adj values are all 0
    # if adj.sum().sum() == 0:
    #     return {
    #         "adj": adj,
    #         "node_names": node_names,
    #         "ranks": node_names,  # hmm should we return empty or random?
    #     }

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
