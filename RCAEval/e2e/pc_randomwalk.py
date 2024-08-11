from cfm.graph_construction.fci import fci_default
from cfm.graph_construction.granger import granger
from cfm.graph_construction.lingam import DirectLiNGAM, ICALiNGAM
from cfm.graph_construction.pc import pc_default
from cfm.graph_heads.random_walk import random_walk
from cfm.io.time_series import drop_constant, drop_extra, drop_near_constant, drop_time, preprocess


def pc_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    try:
        adj = pc_default(data)
        ranks = random_walk(adj, node_names, num_loop=n_iter)
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


def ntlr_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    from cfm.graph_construction.dag_gnn import notears_low_rank

    node_names = data.columns.to_list()

    adj = []
    try:
        adj = notears_low_rank(data)
        ranks = random_walk(adj, node_names, num_loop=n_iter)
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


def fci_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    try:
        adj = fci_default(data)
        ranks = random_walk(adj, node_names, num_loop=n_iter)
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


def lingam_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    try:
        model = ICALiNGAM()
        model.fit(data.to_numpy().astype(float))
        adj = model.adjacency_matrix_
        adj = adj.astype(bool).astype(int)
        ranks = random_walk(adj, node_names, num_loop=n_iter)
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


def granger_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    try:
        adj = granger(data)
        adj = adj.astype(bool).astype(int)
        ranks = random_walk(adj, node_names, num_loop=n_iter)
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
