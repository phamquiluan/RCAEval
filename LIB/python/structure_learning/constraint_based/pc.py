
import pandas as pd

# sys.path.append('../libraries/')
from libraries.pclib import estimate_cpdag, estimate_skeleton
from python.bnutils import one_hot_encoder

# sys.path.append('../python/')
from python.ci_tests import fast_conditional_ind_test
from sklearn.preprocessing import KBinsDiscretizer


def pcalg(data, nodes, alpha=0.05, cit=fast_conditional_ind_test, disc=None, n_bins=5):
    """wrapper for pc algorithm
    Args:
        data: pandas dataframe object
        nodes: list of nodes with attributes
        cit: condiditional independence test to be used
        disc: type of discretization to be used for continuous data
    Returns:
        CPDAG (as a networkx.Graph)
    """
    if disc is not None:
        # discretize continuous data
        discretizer = KBinsDiscretizer(n_bins, encode="ordinal", strategy=disc)
        nodes_list = []
        for node in nodes:
            if node[1]["type"] == "cont":
                nodes_list.append(node[0])
        data[nodes_list] = discretizer.fit_transform(data[nodes_list])
    onehot_dict = {}
    if cit == fast_conditional_ind_test:
        # create dictionary of one-hot encoded features in dataframe
        for node in nodes:
            if node[1]["type"] == "disc":
                onehot_dict[node[0]] = one_hot_encoder(data[:][node[0]].to_numpy())
    skel, sep_set = estimate_skeleton(
        cit, data.to_numpy(), alpha, nodes=nodes, onehot_dict=onehot_dict
    )
    return estimate_cpdag(skel, sep_set)
