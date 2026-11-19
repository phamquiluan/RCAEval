from RCAEval.graph_construction.fci import fci_default
from RCAEval.graph_construction.granger import granger
from RCAEval.graph_construction.lingam import DirectLiNGAM, ICALiNGAM
from RCAEval.graph_construction.pc import pc_default
from RCAEval.graph_heads.random_walk import random_walk, sli_correlation
from RCAEval.io.time_series import drop_constant, drop_extra, drop_near_constant, drop_time, preprocess
from RCAEval.e2e import rca


def _walk_ranks(adj, data, sli=None, n_iter=None):
    """Rank the columns of `data` by a random walk on their causal graph `adj`.

    Without an SLI the nodes weigh the same and the walk starts at a random
    node, so only the graph shapes it. With one, the walk starts at the SLI and
    prefers the neighbours that correlate with it (|Pearson r|).
    """
    node_names = data.columns.to_list()
    scores = None
    if sli is not None:
        if sli not in node_names:
            raise ValueError(f"the SLI {sli!r} is not among the metrics (misspelt, or dropped as constant)")
        scores = sli_correlation(data, sli)
    ranks = random_walk(adj, node_names, sli=sli, num_loop=n_iter, previous_scores=scores)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranks]


@rca
def pc_randomwalk(data, inject_time=None, dataset=None, n_iter=None, sli=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    adj = pc_default(data)
    ranks = _walk_ranks(adj, data, sli=sli, n_iter=n_iter)
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def ntlr_randomwalk(data, inject_time=None, dataset=None, n_iter=None, sli=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )
    from RCAEval.graph_construction.dag_gnn import notears_low_rank

    node_names = data.columns.to_list()

    adj = notears_low_rank(data)
    # NOTEARS sets adj[cause, effect]; the walk reads 0/1 entries as adj[effect, cause]
    ranks = _walk_ranks(adj.T, data, sli=sli, n_iter=n_iter)
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def fci_randomwalk(data, inject_time=None, dataset=None, n_iter=None, sli=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    adj = fci_default(data)
    ranks = _walk_ranks(adj, data, sli=sli, n_iter=n_iter)
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def lingam_randomwalk(data, inject_time=None, dataset=None, n_iter=None, sli=None, **kwargs):
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
    ranks = _walk_ranks(adj, data, sli=sli, n_iter=n_iter)
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }



@rca
def granger_randomwalk(data, inject_time=None, dataset=None, n_iter=None, sli=None, **kwargs):
    data = preprocess(
        data=data, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    node_names = data.columns.to_list()

    if n_iter is None:
        n_iter = len(node_names)

    adj = granger(data)
    adj = adj.astype(bool).astype(int)
    ranks = _walk_ranks(adj, data, sli=sli, n_iter=n_iter)
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
