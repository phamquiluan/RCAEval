# Contributors: Siddharth Jain, Aayush Makharia, Vineet Malik, Sourav Suman, Ayush Chauhan, Gaurav Sinha
# Owned by: Adobe Corporation

# Implements bayesian network simulation methods

import itertools
import math
import random

import networkx as nx
import numpy as np
from scipy.stats import multinomial, norm


def get_random_DAG(n, num_edges=0, graph_density=None):
    """Simulate random DAG with given num_edges/graph_density.

    Args:
        n: number of nodes
        num_edges: number of edges
        graph_density: expected graph density

    Returns:
        DG: Randomly generated DAG
    """
    N = num_edges
    if graph_density is not None:
        N = int(np.random.normal(graph_density * n / 2, n / 4))
    if N < 0:
        raise Exception("Number of edges can't be negative")
    if N > n * (n - 1) / 2:
        raise Exception("Too many edges for a DAG")

    DG = nx.DiGraph()
    DG.add_nodes_from(range(n))
    edges = random.sample([(i, j) for i in range(n) for j in range(i)], N)
    weights = [x + 0.5 if x >= 0 else x - 0.5 for x in np.random.uniform(-1, 1, N)]
    weighted_edges = [(*edge[0], edge[1]) for edge in zip(edges, weights)]
    DG.add_weighted_edges_from(weighted_edges)
    # random permutation
    W = nx.to_numpy_matrix(DG)
    Perm_matrix = np.random.permutation(np.eye(n, n))
    W = np.dot(np.dot(Perm_matrix.T, W), Perm_matrix)
    DG = nx.DiGraph(W)
    return DG


def assign_node_type(G, p, min_cardinality, max_cardinality):
    """Assign node type to each node given a bias parameter

    Args:
        G: DAG
        p: probability of getting continuous node, greater p implies more continuous nodes
        min_cardinality: min number of categories variable can have
        max_cardinality: max number of categories variable can have

    Returns:
        DG: DAG with mixed variable type nodes
    """
    if (min_cardinality <= 1) or (max_cardinality < min_cardinality):
        raise Exception("Cardinality range not valid")

    for i in list(G.nodes()):
        type = np.random.binomial(1, p)
        if type == 0:
            G.nodes[i]["type"] = "disc"
            G.nodes[i]["num_categories"] = random.randint(min_cardinality, max_cardinality)
        else:
            G.nodes[i]["type"] = "cont"
            G.nodes[i]["num_categories"] = "NA"


def get_random_DAG_with_weight(nodes, deg, g_type, w_range):
    """Simulate random DAG with some expected degree.

    Args:
        nodes: number of nodes
        deg: expected node degree, in + out
        g_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)

    Returns:
        DG: weighted DAG
    """

    if g_type == "erdos-renyi":
        prob = float(deg) / (nodes - 1)
        temp = np.random.rand(nodes, nodes) < prob
        LTM = np.tril(temp.astype(float), k=-1)
    elif g_type == "barabasi-albert":
        m = int(round(deg / 2))
        LTM = np.zeros([nodes, nodes])
        bag = [0]
        for i in range(1, nodes):
            dest = np.random.choice(bag, size=m)
            for j in dest:
                LTM[i, j] = 1
            bag.append(i)
            bag.extend(dest)
    elif g_type == "full":  # ignore degree, only for experimental use
        temp = np.ones([nodes, nodes])
        LTM = np.tril(temp, k=-1)
    else:
        raise ValueError("unknown graph type")

    # random permutation
    Perm_matrix = np.random.permutation(np.eye(nodes, nodes))  # permutes first axis only
    LTM_perm = np.dot(np.dot(Perm_matrix.T, LTM), Perm_matrix)

    Uniform_matrix = np.random.uniform(low=w_range[0], high=w_range[1], size=[nodes, nodes])
    temp = np.random.rand(nodes, nodes) < 0.5
    Uniform_matrix[temp] *= -1

    W = (LTM_perm != 0).astype(float) * Uniform_matrix
    DG = nx.DiGraph(W)
    return DG


def uniform_sampler(val_ranges):
    """generate non-zero uniform sample within given range

    Args:
        val_ranges : list of tuples where each tuple denotes an interval. The different intervals must be disjoint. The range does not contain 0.

    Returns:
        uniformly sampled value from the provided range.
    """
    interval_len = np.array([float(interval[1] - interval[0]) for interval in val_ranges])
    total_len = sum(interval_len)
    # maps the discontinuous range into range (0,total_len)
    sample = random.uniform(0, total_len)
    cum_interval_len = np.cumsum(interval_len)
    # converts the sampled value from mapped range to the value needed from provided range
    for i in range(len(interval_len)):
        if sample < cum_interval_len[i]:
            return val_ranges[i][0] + sample - (cum_interval_len[i] - interval_len[i])
    return


def random_parameter_generator(
    dag, cont_cpd_type="lin-gaussian", disc_cpd_type="multi-class-logistic", val_ranges=[(0.1, 1)]
):
    """generate random parameters for conditional probability distributions

    Args:
        dag: The Input Directed Acyclic Graph
        cont_cpd_type: value {lin-gaussian} type of distribution being considered for cont variables
        disc_cpd_type: value {multi-class-logistic} type of distribution being considered for disc variables
        val_ranges : range from which parameters have to be generated. list of tuples where each tuple denotes an interval.
        The different intervals must be disjoint. The range does not contain 0.
    """

    def discrete_parent_settings(node_list, dag):
        """used to generate all possible combinations of discrete parents"""
        categories_arr = []
        # gets a list of categories list for each discrete parent
        for k in node_list:
            if dag.nodes[k]["type"] == "disc":
                categories_arr.append(list(range(dag.nodes[k]["num_categories"])))
        # return the cross_product of all lists present in categories_arr
        return list(itertools.product(*categories_arr))

    for i in val_ranges:
        if i[0] <= 0:
            raise Exception("parameter range can't contain non-positive numbers")

    node_ids = list(dag.nodes())
    for target_node in node_ids:
        param = {}
        parents = list(dag.predecessors(target_node))
        parent_values = discrete_parent_settings(parents, dag)

        if dag.nodes[target_node]["type"] == "cont":
            # generates list of parameters of length |cont_par|+1 for each setting of disc parents.
            if cont_cpd_type == "lin-gaussian":
                for par_val in parent_values:
                    weights = [
                        uniform_sampler(val_ranges)
                        for par in parents
                        if dag.nodes[par]["type"] == "cont"
                    ]
                    weights.append(uniform_sampler(val_ranges))
                    param[par_val] = weights
            else:
                raise NotImplementedError("cpd_type not implemeted yet")
        else:
            # generates list of parameters of dimension (|target_node|)*(|cont_par|+1) for each setting of disc parents.
            if disc_cpd_type == "multi-class-logistic":
                for par_val in parent_values:
                    weights = []
                    for j in range(dag.nodes[target_node]["num_categories"]):
                        arr = [
                            uniform_sampler(val_ranges)
                            for par in parents
                            if dag.nodes[par]["type"] == "cont"
                        ]
                        arr.append(uniform_sampler(val_ranges))
                        weights.append(arr)
                    param[par_val] = weights
            else:
                raise NotImplementedError("cpd_type not implemeted yet")

        dag.nodes[target_node]["cpd_weights"] = param

    # Note: an extra parameter is generated so the linear combination of cont parents will be written as sigma(alphai*xi) + alpha0 where alpha are parameters and xi are cont parents.
    return


def lin_gaussian_cpd(dag, target_node, samples):
    """sampler for simulated linear gaussian distribution

    Args:
        dag: input graph
        target_node: variable for which a sample of cpd is to be generated
        samples: value of variables sampled so far, None for the variables left to sample.

    Returns:
        sample for lin-gauss distribution
    """
    node_ids = list(dag.nodes())
    disc_par_val = []
    cont_par_val = [1]
    for k in dag.predecessors(target_node):
        disc_par_val.append(samples[node_ids.index(k)]) if dag.nodes[k][
            "type"
        ] == "disc" else cont_par_val.append(samples[node_ids.index(k)])

    return (
        sum(
            [
                iterator[0] * iterator[1]
                for iterator in zip(
                    dag.nodes[target_node]["cpd_weights"][tuple(disc_par_val)], cont_par_val
                )
            ]
        )
        + norm.rvs()
    )


def multi_class_softmax_cpd(dag, target_node, samples):
    """sampler for simulated multi-class-logistic using softmax distribution

    Args:
        dag: input graph
        target_node: variable for which a sample of cpd is to be generated
        samples: value of variables sampled so far, None for the variables left to sample.

    Returns:
        sample for multinomial distribution
    """
    node_ids = list(dag.nodes())
    disc_par_val = []
    cont_par_val = [1]
    for k in dag.predecessors(target_node):
        disc_par_val.append(samples[node_ids.index(k)]) if dag.nodes[k][
            "type"
        ] == "disc" else cont_par_val.append(samples[node_ids.index(k)])

    param_cont_product_sum = [
        sum([iterator[0] * iterator[1] for iterator in zip(param, cont_par_val)])
        for param in dag.nodes[target_node]["cpd_weights"][tuple(disc_par_val)]
    ]
    mult_sample = multinomial.rvs(
        1,
        [
            math.exp(param_cont_product_sum[i]) / np.sum(np.exp(param_cont_product_sum))
            for i in range(len(param_cont_product_sum))
        ],
    )
    return np.where(mult_sample == 1)[0][0]
