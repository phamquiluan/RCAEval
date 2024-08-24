import queue

import networkx as nx
import numpy as np
from libraries.FGES.graph_util import *
from libraries.FGES.meekrules import MeekRules


def get_undir_edge(g):
    """Find an undirected edge in Graph g, or None if none exist"""
    for x, y in g.edges():
        if has_undir_edge(g, x, y):
            return (x, y)
    return None


def dagFromPattern(graph):
    """
    Construct a specific graph from an equivalence class
    :param graph: Equivalence class (i.e. DAG with unoriented edges)
    :return: Candidate dag, or None if unsuccessful
    """

    dag = graph.copy()
    graph_colliders = get_all_collider_triples(graph)
    rules = MeekRules(undirect_unforced_edges=False)

    if detect_cycle(dag):
        return None

    def check_graph(dag, node):
        """
        Check if a dag is okay.
        :param dag: dag to check
        :param node: node to check for cycles
        :return: dag with implied edges if valid, else None
        """
        rules.orient_implied(dag)
        new_colliders = get_all_collider_triples(dag)
        if new_colliders == graph_colliders and not detect_cycle_at_node(dag, node):
            return dag
        else:
            return None

    def try_to_solve(pattern):
        """
        Try to find a fully oriented dag from a pattern using a DFS
        :param pattern: the pattern to try
        :return: dag if successful, else None
        """
        if pattern is None:
            return None

        edge = get_undir_edge(pattern)

        if edge is None:
            # No more undirected edges
            return pattern

        (node1, node2) = edge
        new_g = pattern.copy()
        if node2 not in get_ancestors(new_g, node1):
            # Orient edge node1 -> node2
            new_g.remove_edge(node2, node1)
            result = try_to_solve(check_graph(new_g, node1))
            if result is not None:
                return result

        new_g = pattern.copy()
        if node1 not in get_ancestors(new_g, node2):
            # Orient edge node2 -> node1
            new_g.remove_edge(node1, node2)
            result = try_to_solve(check_graph(new_g, node2))
            if result is not None:
                return result

        return None

    return try_to_solve(graph)


# TODO: use smarter penalty function than simply a count
def dagFromPatternWithColliders(graph):
    """
    Construct a DAG from the pattern graph that minimizes the number of new colliders added.
    :param graph: The pattern to use
    :return: DAG
    """
    pattern = graph.copy()

    assert not detect_cycle(pattern), "Pattern must be acyclic to start"

    optimal_graph = None
    optimal_penalty = np.inf

    # Stack of (colliders before messing with the edge,
    #           penalty before messing with the edge,
    #           edge to mess with,
    #           status)
    history = queue.LifoQueue(maxsize=len(pattern.edges()))
    history.put((get_all_collider_triples(pattern), 0, get_undir_edge(pattern), 0))

    while not history.empty():
        c, p, edge, status = history.get()

        if edge is None:
            if p < optimal_penalty:
                optimal_penalty = p
                optimal_graph = pattern.copy()
            continue

        (x, y) = edge

        if status == 0:
            # Orient x -> y
            pattern.remove_edge(y, x)
            c1 = check_for_colliders(pattern, y)
            p1 = p + len(c1 - c)  # set difference
            history.put((c, p, edge, 1))

            if not detect_cycle_at_node(pattern, x) and p1 < optimal_penalty:
                history.put((c.union(c1), p1, get_undir_edge(pattern), 0))

        elif status == 1:
            # Orient y -> x
            pattern.add_edge(y, x)
            pattern.remove_edge(x, y)
            c1 = check_for_colliders(pattern, x)
            p1 = p + len(c1 - c)
            history.put((c, p, edge, 2))

            if not detect_cycle_at_node(pattern, y) and p1 < optimal_penalty:
                history.put((c.union(c1), p1, get_undir_edge(pattern), 0))

        elif status == 2:
            pattern.add_edge(x, y)

    assert optimal_graph is not None

    return (optimal_graph, optimal_penalty)


def mean_shift_data(data):
    """Shift all variables in a dataset to have mean zero"""
    return data - np.mean(data, axis=0)


def estimate_parameters(dag, data):
    """
    Estimate the parameters of a DAG to fit the data.
    :return: matrix of edge coefficients, and diagonal matrix of residuals
    For the parameters matrix, p[i, j] is the weight of edge i -> j
    """

    assert get_undir_edge(dag) is None

    data = mean_shift_data(data)
    num_nodes = len(dag.nodes())

    edge_parameters = np.zeros((num_nodes, num_nodes))
    residuals = np.zeros((num_nodes, num_nodes))

    for j in range(num_nodes):
        inbound_nodes = [i for i in range(num_nodes) if has_dir_edge(dag, i, j)]

        if len(inbound_nodes) == 0:
            residuals[j, j] = np.var(data[:, j])
            continue

        assert j not in inbound_nodes

        a = data[:, inbound_nodes]
        b = data[:, j]

        params, r, _, _ = np.linalg.lstsq(a, b)

        residuals[j, j] = r / (data.shape[0] - 1)

        for i in range(len(inbound_nodes)):
            edge_parameters[inbound_nodes[i], j] = params[i]
            # v = edge_parameters * v + e

    return np.array(edge_parameters), np.array(residuals)


def get_covariance_matrix(params, resids):
    """
    Get the covariance matrix from edge parameters
     (representing a DAG) and the residuals.

    For the equation, see "Causal Mapping of Emotion Networks in the Human Brain" (p. 15)
    The params matrix is taken with orientation p[i, j] is the weight for edge i -> j
    """
    id = np.identity(params.shape[0])
    a = np.linalg.inv(id - params.transpose())

    return np.matmul(np.matmul(a, resids), np.transpose(a))
