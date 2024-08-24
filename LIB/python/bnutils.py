# Contributors: Siddharth Jain, Aayush Makharia, Vineet Malik, Sourav Suman, Ayush Chauhan, Gaurav Sinha
# Owned by: Adobe Corporation

# Implements utility functions for bayesian networks

import itertools
from itertools import combinations
from math import pi

import networkx as nx
import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder
from python.simulate import (
    assign_node_type,
    get_random_DAG,
    lin_gaussian_cpd,
    multi_class_softmax_cpd,
    random_parameter_generator,
)
from tqdm import tqdm


def _has_both_edges(graph, x, y):
    """Conditions on the presence of some edges in the graph.
    Args:
        graph: a graph (as a networkx.Graph or networkx.DiGraph)
        x: node in graph
        y: node in graph
    Returns:
        True: if graph has both edges: x-->y and y-->x
        False: otherwise
    """
    return graph.has_edge(x, y) and graph.has_edge(y, x)


def _has_any_edge(graph, x, y):
    """Conditions on the presence of some edges in the graph.
    Args:
        graph: a graph (as a networkx.Graph or networkx.DiGraph)
        x: node in graph
        y: node in graph
    Returns:
        True: if graph has atleast one of the edges: x-->y and y-->x
        False: otherwise
    """
    return graph.has_edge(x, y) or graph.has_edge(y, x)


def _has_only_edge(graph, x, y):
    """Conditions on the presence of some edges in the graph.
    Args:
        graph: a graph (as a networkx.Graph or networkx.DiGraph)
        x: node in graph
        y: node in graph
    Returns:
        True: if graph has edge: x-->y, but doesn't have edge: y-->x
        False: otherwise
    """
    return graph.has_edge(x, y) and (not graph.has_edge(y, x))


def _has_no_edge(graph, x, y):
    """Conditions on the presence of some edges in the graph.
    Args:
        graph: a graph (as a networkx.Graph or networkx.DiGraph)
        x: node in graph
        y: node in graph
    Returns:
        True: if graph doesn't have both the edges: x-->y and y-->x
        False: otherwise
    """
    return not (graph.has_edge(x, y) or graph.has_edge(y, x))


def _has_one_edge(graph, x, y):
    """Conditions on the presence of some edges in the graph.
    Args:
        graph: a graph (as a networkx.Graph or networkx.DiGraph)
        x: node in graph
        y: node in graph
    Returns:
        True: if graph has exactly one of the two edges: x-->y and y-->x
        False: otherwise
    """
    return graph.has_edge(x, y) ^ graph.has_edge(y, x)


def get_CPDAG(graph):
    """Generates Completed Partially DAG from graph.
    Args:
        graph: a graph (as a networkx.Graph or networkx.DiGraph)
    Returns:
        CPDAG, a directed graph (as a networkx.DiGraph)
    """
    nodes = graph.nodes()
    dag = graph.to_undirected().to_directed()

    for i, j in combinations(nodes, 2):
        # Checks whether k is an unshielded collider.
        if _has_no_edge(graph, i, j):
            for k in nodes:
                if _has_only_edge(graph, i, k) and _has_only_edge(graph, j, k):
                    if dag.has_edge(k, i):
                        dag.remove_edge(k, i)
                    if dag.has_edge(k, j):
                        dag.remove_edge(k, j)

    old_dag = dag.copy()
    while True:
        for i, j in combinations(nodes, 2):
            # Rule 1: Orient i-j into i->j whenever there is an arrow k->i such that k and j are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Look all the predecessors of i.
                for k in dag.predecessors(i):
                    # Skip if there is an arrow i->k.
                    if dag.has_edge(i, k):
                        continue
                    # Skip if k and j are adjacent.
                    if _has_any_edge(dag, k, j):
                        continue
                    # Make i-j into i->j
                    dag.remove_edge(j, i)
                    break
            # Rule 2: Orient i-j into i->j whenever there is a chain i->k->j.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where k is i->k.
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                # Find nodes j where j is k->j.
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(succs_i & preds_j) > 0:
                    # Make i-j into i->j
                    dag.remove_edge(j, i)
            # Rule 3: Orient i-j into i->j whenever there are two chains i-k->j and i-l->j such that k and l are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                # For all the pairs of nodes in adj_i,
                for k, l in combinations(adj_i, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->j.
                    if not _has_only_edge(dag, k, j):
                        continue
                    # Skip if not l->j.
                    if not _has_only_edge(dag, l, j):
                        continue
                    # Make i-j into i->j.
                    dag.remove_edge(j, i)
                    break
        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag


def ancestral_sampling(dag, cpd_sampler_functions, num_samples=1):
    """performs ancestral sampling on dag based on CPD
    Args:
        dag: Directed Acyclic Graph for which samples have to be generated
        cpd_sampler_functions: list of functions. A list of cpds where cpd at index i corresponds to cpd for node i.
        num_samples: Number of samples that have to be generated
    Returns:
        Pandas dataframe object containing samples
    """
    if num_samples <= 0:
        raise ValueError("Not a valid sample size")
    sorted_nodes = list(nx.topological_sort(dag))
    node_ids = list(dag.nodes())
    samples = []
    nodes_idx = {node: idx for idx, node in enumerate(node_ids)}

    for sam_num in tqdm(range(num_samples)):
        # Sampling for each node in topological sorted manner
        cur_sample = [None] * len(node_ids)
        for target_node in sorted_nodes:
            cur_sample[nodes_idx[target_node]] = cpd_sampler_functions[nodes_idx[target_node]](
                dag, target_node, cur_sample
            )
        samples.append(cur_sample)

    return pd.DataFrame(samples, columns=node_ids)


def one_hot_encoder(x):
    """Encode categorical variables as one-hot encoding
    Args:
        x: categorical variable x
    Returns:
        one hot encoding of x
    """
    enc = OneHotEncoder(cols=[0], handle_unknown="indicator").fit(x)
    x = enc.transform(x).to_numpy()
    return x


def conditional_gaussian_likelihood(df, node, z_disc, z_cont, levels, nodes):
    """Calculates likelihood and degree of freedom for the variable 'node' conditioned on set Z
    Args:
        df: Pandas dataframe object for the sampled data
        node: target node of the graph
        z_disc, z_cont: set of discrete and continuous variables in Z respectively
        levels: list of number of categories of the discrete variables in Z
        nodes: dictionary of nodes and its attributes in graph

    Returns:
        likelihood, degree of freedom
    """

    def discrete_parent(node_list, levels):
        """used to generate all possible combinations of discrete parents"""
        categories_arr = []
        # gets a list of categories list for each discrete parent
        for k in node_list:
            categories_arr.append(list(range(levels[node_list.index(k)])))
        # return the cross_product of all lists present in categories_arr
        return list(itertools.product(*categories_arr))

    def get_likelihood_and_dof(disc_list, cont_list, levels):
        """used to calculate likelihood and degree of freedom given the list of discrete and continuous variables"""
        ll, dof = 0, 0
        par_val = discrete_parent(disc_list, levels)
        for i in par_val:
            Df_req = df
            for itr in zip(disc_list, i):
                Df_req = Df_req.loc[Df_req[itr[0]] == itr[1]]
            Df_req = Df_req[cont_list].to_numpy()
            d = Df_req.shape[1]
            Np = Df_req.shape[0]
            if Np == 0:
                continue
            mu = np.mean(Df_req, axis=0)
            Df_req = Df_req - mu
            #             covariance = np.zeros((Df_req.shape[1], Df_req.shape[1]))
            #             for row in Df_req:
            #                 covariance += np.multiply.outer(row, row)
            covariance = np.dot(np.transpose(Df_req), Df_req)
            covariance = covariance / Df_req.shape[0]
            det = np.linalg.det(covariance)
            if det != 0:
                ll += -(Np / 2) * (np.log(abs(det)) + d * np.log(2 * pi) + d) + Np * np.log(
                    Np / df.shape[0]
                )
                dof += (d * (d + 1) / 2) + 1
            else:
                ll += Np * np.log(Np / df.shape[0])
                dof += (d * (d + 1) / 2) + 1
        dof -= 1
        return ll, dof

    ll0, dof0 = get_likelihood_and_dof(z_disc, z_cont, levels)
    z_disc_with_node = z_disc.copy()
    z_cont_with_node = z_cont.copy()
    levels_with_node = levels.copy()
    z_disc_with_node.append(node) if nodes[node]["type"] == "disc" else z_cont_with_node.append(
        node
    )
    if nodes[node]["type"] == "disc":
        levels_with_node.append(nodes[node]["num_categories"])
    ll1, dof1 = get_likelihood_and_dof(z_disc_with_node, z_cont_with_node, levels_with_node)
    return ll1 - ll0, dof1 - dof0


def data_generator(nodes, graph_density, node_type_ratio, sample_size):
    """
    Generates a random DAG and samples data from it

    Args:
        nodes: number of nodes required in the random DAG
        graph_density: required graph density of the random DAG to be generated
        node_type_ratio: ratio of continuous type nodes to total nodes in graph where value 1 means all continuous nodes
        sample_size: required size of data sampled from the DAG
    Returns:
        dag: netwrokx object of random DAG
        df: pandas dataframe object of sampled data from the DAG
    """
    dag = get_random_DAG(nodes, graph_density=graph_density)
    assign_node_type(dag, node_type_ratio, 3, 5)
    random_parameter_generator(dag)
    func_list = []
    for node in dag.nodes():
        func_list.append(multi_class_softmax_cpd) if (
            dag.nodes[node]["type"] == "disc"
        ) else func_list.append(lin_gaussian_cpd)
    df = ancestral_sampling(dag, func_list, sample_size)
    return dag, df


def column_mapping(data):
    """
    Maps column names from 0 to total column count

    Args:
        data: pandas dataframe object with string type names
    Returns:
        forward_col_mapping: dictionary with mapping from original column names to dummy names
        backward_col_mapping: dictionary with mapping from dummy names to original column names
    """
    forward_col_mapping = {}
    backward_col_mapping = {}
    for i, col in enumerate(data.columns):
        forward_col_mapping[col] = i
        backward_col_mapping[i] = col
    return (forward_col_mapping, backward_col_mapping)


def pre_processing(data, custom=False, cont_columns=[], max_categories=20):
    """
    Preprocesses the names and data_types of columns to make it runnable on algorithms

    Args:
        data: pandas dataframe object with string type names
        custom: True if user has prior knowledge about which columns are continuous
        cont_columns: If custom is true, then user can specify the names of continuous columns as list
        max_categories: If custom is false, then user can specify the maximum number of categories of categorical variables in the dataset
    Returns:
        data: pandas dataframe after processing
        args: networkx object containing the data type and number of categories of different variables
        mappers: tuple containing (forward mapping, backward mapping) for data columns
        cat_mappers: dict containing mapping of categorical values
    """
    nodes = []
    args = nx.DiGraph()
    cat_mappers = dict()
    mappers = column_mapping(data)
    data.rename(columns=mappers[0], inplace=True)
    args.add_nodes_from(list(data.columns))
    if not custom:
        for col in data.columns:
            categories = len(pd.Series.unique(data[col]))
            if categories > max_categories:
                args.nodes[col]["type"] = "cont"
                args.nodes[col]["num_categories"] = "NA"
            else:
                args.nodes[col]["type"] = "disc"
                args.nodes[col]["num_categories"] = categories
                cat_mappers[col] = (
                    pd.unique(data[col]),
                    list(range(pd.unique(data[col]).shape[0])),
                )
                data = data.replace({col: cat_mappers[col][0]}, {col: cat_mappers[col][1]})
    else:
        for col in data.columns:
            categories = len(pd.Series.unique(data[col]))
            if mappers[1][col] in cont_columns:
                args.nodes[col]["type"] = "cont"
                args.nodes[col]["num_categories"] = "NA"
            else:
                args.nodes[col]["type"] = "disc"
                args.nodes[col]["num_categories"] = categories
                cat_mappers[col] = (
                    pd.unique(data[col]),
                    list(range(pd.unique(data[col]).shape[0])),
                )
                data = data.replace({col: cat_mappers[col][0]}, {col: cat_mappers[col][1]})

    return (data, args, mappers, cat_mappers)
