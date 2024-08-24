# import sys
# sys.path.append('../python/')
import networkx as nx
from python.ci_tests import fast_conditional_ind_test
from python.discretize import Data_Driven_Discretizer, get_mlameva
from python.scores import bdeu_score
from python.structure_learning.constraint_based.mdepcs import MDEPCS
from python.structure_learning.score_based.fges import fges


def d3hges(
    data,
    nodes,
    alpha=0.1,
    blacklist=False,
    verbose=False,
    max_process=5,
    method=get_mlameva,
    score=bdeu_score,
):
    """Performs D3HGES on the data set and returns the learned DAG
     D3HGES is a three stage algorithm where the first stage is dependency learning,
     second stage is the data-driven discretization
     and the last stage is search-and-score stage which finally outputs the DAG

    Args:
        data: pandas Dataframe containing samples
        nodes: dictionary containing the node type
        alpha: significance level for the conditional independence test
        cit: Conditional independence test
        verbose: print output in file
        max_process: maximum number of parallel processes
        method: discretization score to be used in data-driven discretization
        score: score to be used in search-and-score phase

    Returns:
        DAG (networkx DiGraph object)
    """
    # stage 1: dependency learning
    skel_learn = MDEPCS(data, nodes, alpha, verbose=verbose, max_process=max_process)
    skel = skel_learn.mdepcs()
    PCS_neigh = skel_learn.get_PCS_neighbors()
    # print("Dependency learning phase complete")
    # stage 2: data-driven discretization
    disc_data = Data_Driven_Discretizer(
        data.copy(), skel, PCS_neigh, nodes, alpha, max_process, method=method
    ).discretize()
    # print("Data-driven discretization done")
    # stage 3: search-and-score
    blacklist_skel = None
    if blacklist:
        blacklist_skel = skel_learn.skel_union()
        blacklist_skel = blacklist_skel.to_directed()
    dag = fges(disc_data, nodes, black_list_skel=blacklist_skel, disc=None, score=score)
    disc_data.to_pickle(
        "./Discretized_Data_" + str(len([node[0] for node in nodes])) + "_nodes.pickle"
    )
    nx.write_gpickle(dag, "./DAG_" + str(len([node[0] for node in nodes])) + "_nodes.gpickle")
    return dag
