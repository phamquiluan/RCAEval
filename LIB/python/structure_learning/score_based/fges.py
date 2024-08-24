
import networkx as nx
from libraries.FGES.knowledge import Knowledge

# sys.path.append('../libraries/FGES/')
from libraries.FGES.runner import fges_runner

# sys.path.append('../python/')
from python.scores import bdeu_score


def fges(
    data,
    nodes,
    white_list=None,
    black_list=None,
    black_list_skel=None,
    white_list_skel=None,
    disc=None,
    n_bins=5,
    score=bdeu_score,
):
    """wrapper function for FGES"""
    knowledge = Knowledge()
    if white_list is not None or black_list is not None:
        knowledge.set_forbidden(black_list)
        knowledge.set_required(white_list)
    elif black_list_skel is not None:
        knowledge.set_forbidden_from_skeleton(black_list_skel)
    elif white_list_skel is not None:
        knowledge.set_required_from_skeleton(white_list_skel)
    else:
        knowledge = None
    result = fges_runner(data, nodes, score, knowledge, disc=disc, n_bins=n_bins)

    # create a graph to be returned by algorithm
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(result["graph"].edges())
    return g
