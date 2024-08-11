import itertools
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# np.random.seed(24)
import pandas as pd
from causallearn.score.LocalScoreFunction import local_score_BIC
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import fisherz
from libraries.FGES.knowledge import Knowledge
from libraries.FGES.runner import fges_runner
from python.bnmetrics import *
from python.bnutils import *
from python.ci_tests import *
from python.discretize import *
from python.scores import *
from python.structure_learning.score_based.fges import fges

from cfm.classes.data import CaseData, DataLoader, MemoryDataLoader
from cfm.classes.graph import Graph, MemoryGraph, Node
from cfm.graph_heads.random_walk import Score, Scorer
from cfm.graph_heads.rht import RHTScorer
from cfm.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    select_useful_cols,
)
from cfm.utility.visualization import visualize_metrics

# from utils import compute_stats, read_data
warnings.filterwarnings("ignore")


import networkx as nx
import pandas as pd
from python.bnmetrics import *
from python.bnutils import *
from python.ci_tests import *
from python.discretize import *


def fges(data, score_func=None, **kwargs):
    assert score_func in [None, "linear", "p2", "p3"]
    if score_func is None or score_func == "linear":
        score_func = linear_gaussian_score_iid
    elif score_func == "p2":
        score_func = polynomial_2_gaussian_score_iid
    elif score_func == "p3":
        score_func = polynomial_3_gaussian_score_iid

    data.columns = range(data.shape[1])

    g = nx.DiGraph()
    g.add_nodes_from(list(data.columns))
    for col in data.columns:
        g.nodes[col]["type"] = "cont"
        g.nodes[col]["num_categories"] = "NA"

    result = fges_runner(
        data, g.nodes(data=True), n_bins=1, disc=None, score=score_func, knowledge=None
    )
    G = result["graph"]

    adj = nx.adjacency_matrix(G).todense()
    # adj = np.zeros((data.shape[1], data.shape[1]))
    # for n, neighbors in G.adjacency():
    #     for i in neighbors.keys():
    #         adj[i, n] = 1
    return adj
