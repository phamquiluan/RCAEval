# Contributors: Siddharth Jain, Aayush Makharia, Vineet Malik, Sourav Suman, Ayush Chauhan, Gaurav Sinha
# Owned by: Adobe Corporation
from multiprocessing import Pool

import numpy as np
import pandas as pd
from feature_engine.discretisation import decision_tree as dsc

# sys.path.append('./libraries/')
from libraries.caim_test import CAIMD
from python.bnutils import one_hot_encoder
from python.ci_tests import fast_conditional_ind_test as FCIT
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

# Implements multiple discretization techniques for continuous variables


class Decision_Tree_Discretizer:

    """Discretizes a continuous variable using Decision tree classifier
    Args:
        score: score to be considered for discretization
        **kwargs: dictionary of parameters for DecisionTreeDiscretiser

        kwargs format with default values: {'cv': 10, 'regression': False, 'max_depth': [1,2,3], 'max_samples_leaf': [10, 4]}
    """

    def __init__(self, score="accuracy", **kwargs):
        self.cv = kwargs.get("cv", 10)
        self.scoring = score
        self.regression = (kwargs.get("regression", False),)
        self.param_grid = {
            "max_depth": kwargs.get("max_depth", [1, 2, 3]),
            "min_samples_leaf": kwargs.get("max_samples_leaf", [10, 4]),
        }

    def fit(self, data, node, target, **kwargs):
        self.node = node
        self.disc = dsc.DecisionTreeDiscretiser(
            cv=self.cv,
            scoring=self.scoring,
            variables=[node],
            regression=False,
            param_grid=self.param_grid,
        )
        self.disc.fit(data[[node, target]], data[target])
        print(self.disc.scores_dict_[node])
        return self, self.disc.scores_dict_[node]

    def transform(self, data):
        print(data)
        return self.disc.transform(data[[data.columns[0], data.columns[1]]])[self.node]


def unsupervised_discretization(df, node_list, bins, discretization_type):
    """Bins continuous data into intervals.
    Args:
        df : pandas dataframe object wtih mixed data
        node_list : list of continuous nodes
        bins : number of intervals with equal width
        discretization_type : takes one of the following values -
            uniform : generates bins of equal width
            frequency : generates bins of equal frequency
            K-means : generates bins using kmeans algorithm
    Returns:
        dataframe with discretized columns appended in df
    """
    discretizer = None
    if discretization_type == "uniform":
        discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="uniform")
    elif discretization_type == "quantile":
        discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile")
    elif discretization_type == "kmeans":
        discretizer = KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="kmeans")
    else:
        raise NotImplementedError("Invalid discretization type")
    df[node_list] = discretizer.fit_transform(df[node_list])
    return df


def get_laim(sp, scheme, xi, y):
    """LAIM score for discretization
    Args:
        sp : indexes of x corresponding to each bin
        scheme : set of thresholds for the discretized bins
        xi : attribute being discretized
        y : target to be used for discretization of xi
    Returns:
        LAIM score
    """
    sp.insert(0, 0)
    sp.append(xi.shape[0])
    n = len(sp) - 1

    M = 0
    laim = 0
    for r in range(n):
        init = sp[r]
        fin = sp[r + 1]
        val, counts = np.unique(y[init:fin], return_counts=True)
        if val[0] == -1e10:
            val = val[1:]
            counts = counts[1:]

        Mr = counts.sum()
        maxr = counts.max()
        laim += (maxr / Mr) * maxr
        M += Mr

    laim /= n * M
    return laim


def get_ameva(sp, scheme, xi, y):
    """Ameva score for discretization
    Args:
        sp : indexes of x corresponding to each bin
        scheme : set of thresholds for the discretized bins
        xi : attribute being discretized
        y : target to be used for discretization of xi
    Returns:
        Ameva score
    """
    sp.insert(0, 0)
    sp.append(xi.shape[0])
    n = len(sp) - 1

    label, label_counts = np.unique(y, return_counts=True)
    M_label = dict()
    for j in range(len(label)):
        M_label[label[j]] = label_counts[j]

    M = 0
    ameva = 0
    for r in range(n):
        init = sp[r]
        fin = sp[r + 1]
        val, counts = np.unique(y[init:fin], return_counts=True)

        Mr = counts.sum()
        for j in range(len(val)):
            ameva += (counts[j] / Mr) * (counts[j] / M_label[val[j]])
        M += Mr

    ameva = (M * (ameva - 1)) / (n * (len(label) - 1))
    return ameva


def get_mlameva(sp, scheme, xi, y):
    """Multi Label Ameva score for discretization
    Args:
        sp : indexes of x corresponding to each bin
        scheme : set of thresholds for the discretized bins
        xi : attribute being discretized
        y : target to be used for discretization of xi
    Returns:
        MLAmeva score
    """
    sp.insert(0, 0)
    sp.append(xi.shape[0])
    n = len(sp) - 1

    label, label_counts = np.unique(y, return_counts=True)
    if label[0] == -1e10:
        label = label[1:]
        label_counts = label_counts[1:]
    M_label = dict()
    for j in range(len(label)):
        M_label[label[j]] = label_counts[j]

    M = 0
    mlameva = 0
    for r in range(n):
        init = sp[r]
        fin = sp[r + 1]
        val, counts = np.unique(y[init:fin], return_counts=True)
        if val[0] == -1e10:
            val = val[1:]
            counts = counts[1:]

        Mr = counts.sum()
        for j in range(len(val)):
            mlameva += (counts[j] / Mr) * (counts[j] / M_label[val[j]])
        M += Mr

    mlameva = (M * (mlameva - 1)) / (n * (len(label) - 1))
    return mlameva


def parallel(args):
    if FCIT(args[0], args[3], args[4], args[5], nodes=args[1], onehot_dict=args[2]) > args[6]:
        return args[4]
    return -1


class Data_Driven_Discretizer:
    """Used to discretize a set of variables using the inter-dependence information available before hand.
    Args:
        data : data to be discretized
        skel : the inter-dependence knowledge available before-hand
        nodes : a dict containing extra information about the variables
        max_process : max no of processes to create during parallelization
        method : core discretization technique to be used
    Returns:
        discretized data
    """

    def __init__(
        self,
        data,
        skel,
        cond_check_skel,
        nodes,
        alpha=0.1,
        max_process=10,
        discretizer=CAIMD,
        method=get_mlameva,
    ):
        self.data = data
        self.alpha = alpha
        self.max_process = max_process
        self.skel = skel
        self.cond_check_skel = cond_check_skel
        self.nodes = nodes
        self.disc_data = data.copy()
        self.cont_list = [node[0] for node in self.nodes if node[1]["type"] == "cont"]
        self.disc_list = [node[0] for node in self.nodes if node[1]["type"] == "disc"]
        self.n_samples = self.data.shape[0]
        self.onehot_dict = {
            node[0]: one_hot_encoder(data[:][node[0]].to_numpy())
            for node in self.nodes
            if node[1]["type"] == "disc"
        }
        self.discretizer = discretizer(score=method, max_process=max_process)

    def cond_check(self, node, neigh, neighbors, scheme):
        """Returns the no of nodes that have either changed to independent or dependent on "node" due to discretization"""
        pool = Pool(self.max_process)
        PCS = set(self.cont_list + self.disc_list) - set([node])
        data_disc = self.data.copy()

        data_disc[node] = scheme.transform(self.data[[node, neigh]])
        data_disc = data_disc.replace(
            {node: pd.unique(data_disc[node])},
            {node: list(range(pd.unique(data_disc[node]).shape[0]))},
        )
        self.onehot_dict[node] = one_hot_encoder(data_disc[node].to_numpy())

        args = [
            (self.data, self.nodes, self.onehot_dict, node, neigh, [], self.alpha) for neigh in PCS
        ]
        PCS = PCS - set(pool.map(parallel, args))
        args = [
            (self.data, self.nodes, self.onehot_dict, node, X, [Z], self.alpha)
            for X in PCS
            for Z in PCS - set([X])
        ]
        new_neighbors = PCS - set(pool.map(parallel, args))

        pool.close()
        pool.join()

        return len(new_neighbors - set(neighbors)) + len(set(neighbors) - new_neighbors)

    def discretize(self):
        """Entry point into the algorithm"""
        cont_queue = []
        for node in self.cont_list:
            if len(list(self.skel.neighbors(node))) != 0:
                ratio = len(set(self.skel.neighbors(node)) - set(self.cont_list)) / len(
                    list(self.skel.neighbors(node))
                )
            else:
                ratio = 0
            cont_queue.append((ratio, len(list(self.skel.neighbors(node))), node))
        cont_queue = sorted(cont_queue, key=lambda x: (-x[0], -x[1]))

        while cont_queue:
            (ratio, _, node) = cont_queue.pop(0)
            best_score = -1
            best_scheme = None
            main_list = []
            if ratio == 0:
                iter_set = set(self.disc_list)
            else:
                iter_set = set(self.skel.neighbors(node)) - set(self.cont_list)
            for neigh in iter_set:
                scheme, score = self.discretizer.fit(self.data[[node, neigh]], node, neigh)
                main_list.append(
                    (
                        self.cond_check(node, neigh, list(self.cond_check_skel[node]), scheme),
                        score,
                        scheme,
                        neigh,
                    )
                )
            (best_shd, best_score, best_scheme, best_neigh) = sorted(
                main_list, key=lambda i: (i[0], -i[1])
            )[0]

            self.data[node] = best_scheme.transform(self.data[[node, best_neigh]])
            self.data = self.data.replace(
                {node: pd.unique(self.data[node])},
                {node: list(range(pd.unique(self.data[node]).shape[0]))},
            )
            self.cont_list.remove(node)
            self.disc_list.append(node)
            for i in range(len(cont_queue)):
                if cont_queue[i][2] in set(self.skel.neighbors(node)):
                    cont_queue[i] = (
                        cont_queue[i][0] + 1 / len(list(self.skel.neighbors(cont_queue[i][2]))),
                        cont_queue[i][1],
                        cont_queue[i][2],
                    )
            cont_queue = sorted(cont_queue, key=lambda x: (-x[0], -x[1]))

        return self.data


def PCA_discretizer(data, skel, PCS_neigh, nodes, alpha, max_process, threshold=90):
    cont_nodes = [node[0] for node in nodes if node[1]["type"] == "cont"]
    data_cont = data.loc[:, cont_nodes].values
    data_transformed = StandardScaler().fit_transform(data_cont)
