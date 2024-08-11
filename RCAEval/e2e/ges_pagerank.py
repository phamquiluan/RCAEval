from RCAEval.graph_construction.ges import ges
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.graph_heads.random_walk import random_walk
from RCAEval.io.time_series import preprocess

try:
    from RCAEval.graph_construction.fges import fges
except:
    print("fges not available")


def ges_pagerank(data, inject_time=None, dataset=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    record = ges(data)
    G = record["G"]
    ranks = page_rank(G.graph, node_names=data.columns.to_list())
    return {
        "adj": G.graph,
        "node_names": data.columns.to_list(),
        "ranks": ranks,
    }


def fges_pagerank(
    data, inject_time=None, dataset=None, dk_select_useful=False, with_bg=False, n_iter=10, **kwargs
):
    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    node_names = data.columns.to_list()

    adj = []
    try:
        adj = fges(data)
        ranks = page_rank(adj.T, node_names=node_names, n_iter=n_iter)
    except ValueError as e:
        print(e)
        return {"adj": [], "node_names": node_names, "ranks": node_names}
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


def fges_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    try:
        adj = fges(data)
        ranks = random_walk(adj.T, node_names, num_loop=n_iter)
    except Exception as e:
        print(e)
        return {"adj": [], "node_names": node_names, "ranks": node_names}

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
