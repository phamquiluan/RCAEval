from RCAEval.graph_construction.fci import fci_default
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.io.time_series import preprocess
from RCAEval.e2e import rca


@rca
def fci_pagerank(data, inject_time=None, dataset=None, dk_select_useful=False, n_iter=10, **kwargs):
    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    node_names = data.columns.to_list()

    adj = fci_default(data)
    ranks = page_rank(adj, node_names=node_names, n_iter=n_iter)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
