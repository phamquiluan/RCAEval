# Contributors: Siddharth Jain, Aayush Makharia, Vineet Malik, Sourav Suman, Ayush Chauhan, Gaurav Sinha
# Owned by: Adobe Corporation

# Implements bayesian networks comparison metrics

from itertools import combinations

from python.bnutils import (
    _has_any_edge,
    _has_both_edges,
    _has_no_edge,
    _has_one_edge,
    _has_only_edge,
    get_CPDAG,
)
from python.scores import conditional_gaussian_score


def SHD(G1, G2):
    """Calculates Structural Hamming Distance between two graphs.
    Args:
        G1: a graph (as a networkx.Graph or networkx.DiGraph)
        G2: a graph (as a networkx.Graph or networkx.DiGraph)
    Returns:
        SHD between graphs G1 and G2
    """
    shd = 0
    for i, j in combinations(G1.nodes(), 2):
        # Edge present in G1, but missing in G2.
        if _has_any_edge(G1, i, j) and _has_no_edge(G2, i, j):
            shd += 1
        # Edge missing in G1, but present in G2.
        elif _has_no_edge(G1, i, j) and _has_any_edge(G2, i, j):
            shd += 1
        # Edge undirected in G1, but directed in G2.
        elif _has_both_edges(G1, i, j) and _has_one_edge(G2, i, j):
            shd += 1
        # Edge directed in G1, but undirected or reversed in G2.
        elif (_has_only_edge(G1, i, j) and G2.has_edge(j, i)) or (
            _has_only_edge(G1, j, i) and G2.has_edge(i, j)
        ):
            shd += 1
    return shd


def SHD_C(G1, G2):
    """Calculates Structural Hamming Distance between the CPDAGs of two graphs.
    Args:
        G1: a graph (as a networkx.Graph or networkx.DiGraph)
        G2: a graph (as a networkx.Graph or networkx.DiGraph)
    Returns:
        SHD between the CPDAGs of graphs G1 and G2
    """
    return SHD(get_CPDAG(G1), get_CPDAG(G2))


def adj_stats(dag, cpdag):
    """calculate adjacency precision and recall
    Args:
        dag: ground truth graph
        cpdag: estimated cpdag
    Returns:
        adjacency precision
        adjacency recall
    """
    true_pos = 0
    prec_den = 0
    rec_den = dag.number_of_edges()
    for i, j in combinations(dag.nodes(), 2):
        if _has_any_edge(cpdag, i, j) and _has_any_edge(dag, i, j):
            true_pos += 1
        if _has_any_edge(cpdag, i, j):
            prec_den += 1

    if prec_den == 0 and rec_den != 0:
        return float("NaN"), 0
    elif rec_den == 0:
        if prec_den == 0:
            return float("NaN"), float("NaN")
        else:
            return 0, float("NaN")
    return {"precision": round(true_pos / prec_den, 3), "recall": round(true_pos / rec_den, 3)}


def arrow_stats(dag, cpdag):
    """calculate arrow_head precision and recall
    Args:
        dag: ground truth graph
        cpdag: estimated cpdag
    Returns:
        arrow_head precision
        arrow_head recall
    """
    true_pos = 0
    prec_den = 0
    rec_den = 0
    for i, j in combinations(dag.nodes(), 2):
        if (_has_only_edge(cpdag, i, j) and dag.has_edge(i, j)) or (
            _has_only_edge(cpdag, j, i) and dag.has_edge(j, i)
        ):
            true_pos += 1
        if _has_one_edge(cpdag, i, j):
            prec_den += 1
        if _has_any_edge(dag, i, j) and _has_any_edge(cpdag, i, j):
            rec_den += 1
    if prec_den == 0 and rec_den != 0:
        return float("NaN"), 0
    elif rec_den == 0:
        if prec_den == 0:
            return float("NaN"), float("NaN")
        else:
            return 0, float("NaN")
    return {"precision": round(true_pos / prec_den, 3), "recall": round(true_pos / rec_den, 3)}


def overall_stats(dag, cpdag):
    """calculate overall precision and recall
    Args:
        dag: ground truth graph
        cpdag: estimated cpdag
    Returns:
        overall precision
        overall recall
    """
    true_pos = 0
    prec_den = 0
    rec_den = 0
    for i, j in combinations(dag.nodes(), 2):
        if (_has_only_edge(cpdag, i, j) and dag.has_edge(i, j)) or (
            _has_only_edge(cpdag, j, i) and dag.has_edge(j, i)
        ):
            true_pos += 1
        if _has_any_edge(cpdag, i, j):
            prec_den += 1
        if _has_any_edge(dag, i, j):
            rec_den += 1
    if prec_den == 0 and rec_den != 0:
        return float("NaN"), 0
    elif rec_den == 0:
        if prec_den == 0:
            return float("NaN"), float("NaN")
        else:
            return 0, float("NaN")
    return {"precision": round(true_pos / prec_den, 3), "recall": round(true_pos / rec_den, 3)}


def false_disc_rate(true_dag, dag):
    """calculate false discovery rate
    Args:
        true_dag: ground truth DAG
        dag: estimated DAG
    Returns:
        false discovery rate
    """
    false_disc = 0
    total_disc = 0
    for i, j in combinations(true_dag.nodes(), 2):
        if _has_any_edge(dag, i, j) and _has_no_edge(true_dag, i, j):
            false_disc += 1
        elif (true_dag.has_edge(i, j) and _has_only_edge(dag, j, i)) or (
            true_dag.has_edge(j, i) and _has_only_edge(dag, i, j)
        ):
            false_disc += 1
        if _has_any_edge(dag, i, j):
            total_disc += 1
    if total_disc == 0:
        return float("NaN")
    return round(false_disc / total_disc, 3)


def bic_ratio(df, dag, true_dag, sparsity=1):
    return conditional_gaussian_score(df, dag, sparsity=sparsity) / conditional_gaussian_score(
        df, true_dag, sparsity=sparsity
    )
