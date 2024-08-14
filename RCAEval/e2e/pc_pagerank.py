import networkx as nx
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, gsq, kci, mv_fisherz
from sknetwork.ranking import PageRank

from RCAEval.graph_construction.pc import pc_default
from RCAEval.graph_heads.page_rank import page_rank
from RCAEval.io.time_series import preprocess
from RCAEval.e2e import rca

@rca
def pc_pagerank(
    data, inject_time=None, dataset=None, dk_select_useful=False, with_bg=False, n_iter=10, **kwargs
):
    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    node_names = data.columns.to_list()

    cg = pc(data.to_numpy())
    adj = cg.G.graph
    G = nx.DiGraph()
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i, j] == -1:
                G.add_edge(i, j)
            if adj[i, j] == 1:
                G.add_edge(j, i)
    nodes = sorted(G.nodes())
    adj = np.asarray(nx.to_numpy_matrix(G, nodelist=nodes))

    pagerank = PageRank()
    scores = pagerank.fit_transform(adj.T)
    ranks = list(zip(node_names, scores))
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def cmlp_pagerank(
    data, inject_time=None, dataset=None, dk_select_useful=False, with_bg=False, n_iter=10, **kwargs
):
    from RCAEval.graph_construction.cmlp import cmlp

    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    node_names = data.columns.to_list()

    adj = cmlp(data, max_iter=20000)

    pagerank = PageRank()
    scores = pagerank.fit_transform(adj.T)
    ranks = list(zip(node_names, scores))
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }


@rca
def ntlr_pagerank(
    data, inject_time=None, dataset=None, dk_select_useful=False, with_bg=False, n_iter=10, **kwargs
):
    from RCAEval.graph_construction.dag_gnn import notears_low_rank

    data = preprocess(data=data, dataset=dataset, dk_select_useful=dk_select_useful)
    node_names = data.columns.to_list()

    adj = notears_low_rank(data)
    pagerank = PageRank()
    scores = pagerank.fit_transform(adj.T)
    ranks = list(zip(node_names, scores))
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": node_names,
        "ranks": ranks,
    }
