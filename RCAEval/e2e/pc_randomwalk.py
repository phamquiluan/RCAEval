from RCAEval.graph_construction.fci import fci_default
from RCAEval.graph_construction.granger import granger
from RCAEval.graph_construction.lingam import DirectLiNGAM, ICALiNGAM
from RCAEval.graph_construction.pc import pc_default
from RCAEval.graph_heads.random_walk import random_walk
from RCAEval.io.time_series import drop_constant, drop_extra, drop_near_constant, drop_time, preprocess
from RCAEval.e2e import rca


@rca
def pc_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    adj = pc_default(data)
    ranks = random_walk(adj, node_names, num_loop=n_iter)

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def ntlr_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    from RCAEval.graph_construction.dag_gnn import notears_low_rank

    node_names = data.columns.to_list()

    adj = notears_low_rank(data)
    ranks = random_walk(adj, node_names, num_loop=n_iter)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def fci_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    adj = fci_default(data)
    ranks = random_walk(adj, node_names, num_loop=n_iter)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def lingam_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    model = ICALiNGAM()
    model.fit(data.to_numpy().astype(float))
    adj = model.adjacency_matrix_
    adj = adj.astype(bool).astype(int)
    ranks = random_walk(adj, node_names, num_loop=n_iter)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }



@rca
def granger_randomwalk(data, inject_time=None, dataset=None, n_iter=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    adj = granger(data)
    adj = adj.astype(bool).astype(int)
    ranks = random_walk(adj, node_names, num_loop=n_iter)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
