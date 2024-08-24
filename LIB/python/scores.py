# Contributors: Siddharth Jain, Aayush Makharia, Vineet Malik, Sourav Suman, Ayush Chauhan, Gaurav Sinha
# Owned by: Adobe Corporation

# Implements bayesian network scoring functions

from math import lgamma, log

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pygobnilp.scoring import BGe, ContinuousData
from python.bnutils import conditional_gaussian_likelihood
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.gam.api import BSplines, GLMGam
from statsmodels.tools.eval_measures import bic as BIC
from statsmodels.tools.tools import pinv_extended


def bdeu_score(data, dag, local=False, equivalent_sample_size=10, **kwargs):
    """Calculates the bdeu score for given graph and data.
    Args:
        dag: the graph for which score is to be calculated
        data: the pandas dataframe
        state_names: A dict indicating, for each variable, the discrete set of states (or values) that the variable can take.
        equivalent_sample_size: The equivalent/imaginary sample size (of uniform pseudo samples) for the dirichlet hyperparameters. The score is sensitive to this value, runs with different values might be useful.
    Returns:
        the bdeu score
    """

    def _collect_state_names(variable):
        "Return a list of states that the variable takes in the data"
        states = sorted(list(data.loc[:, variable].dropna().unique()))
        return states

    def state_counts(variable, parents):
        """counts how often each state of 'variable' occured, conditional on parents' states"""
        if not parents:
            # count how often each state of 'variable' occured
            state_count_data = data.loc[:, variable].value_counts()
            return state_count_data.reindex(state_names[variable]).fillna(0).to_frame()
        else:
            parents_states = [state_names[parent] for parent in parents]
            # count how often each state of 'variable' occured, conditional on parents' states
            state_count_data = data.groupby([variable] + parents).size().unstack(parents)
            if not isinstance(state_count_data.columns, pd.MultiIndex):
                state_count_data.columns = pd.MultiIndex.from_arrays([state_count_data.columns])
            return state_count_data.reindex(
                index=state_names[variable],
                columns=pd.MultiIndex.from_product(parents_states, names=parents),
            ).fillna(0)

    def local_score(variable, parents=[]):
        """computes local score for the given variable"""
        var_states = state_names[variable]
        var_cardinality = len(var_states)
        state_counts_var = state_counts(variable, parents)
        num_parents_states = float(len(state_counts_var.columns))
        local_score = 0
        # iterate over df columns (only 1 if no parents)
        for parents_state in state_counts_var:
            conditional_sample_size = sum(state_counts_var[parents_state])
            local_score += lgamma(equivalent_sample_size / num_parents_states) - lgamma(
                conditional_sample_size + equivalent_sample_size / num_parents_states
            )
            for state in var_states:
                if state_counts_var[parents_state][state] > 0:
                    local_score += lgamma(
                        state_counts_var[parents_state][state]
                        + equivalent_sample_size / (num_parents_states * var_cardinality)
                    ) - lgamma(equivalent_sample_size / (num_parents_states * var_cardinality))
        return local_score

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if "state_names" not in kwargs:
        state_names = dict()
        variables = list(data.columns.values)
        for var in variables:
            if var in state_names:
                if not set(_collect_state_names(var)) <= set(state_names[var]):
                    raise ValueError("Data contains unexpected states for variable: {var}.")
                state_names[var] = state_names[var]
            else:
                state_names[var] = _collect_state_names(var)
    else:
        state_names = kwargs.get("state_names")

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return local_score(node, parents)

    return sum(
        [
            local_score(variable, list(dag.predecessors(variable)))
            for variable in data.columns.values.tolist()
        ]
    )


def conditional_gaussian_score(df, dag, local=False, sparsity=1, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def conditional_gaussian_local_score(node, parents):
        parents_disc = [i for i in parents if dag.nodes[i]["type"] == "disc"]
        parents_cont = [i for i in parents if dag.nodes[i]["type"] == "cont"]
        levels = [dag.nodes[node]["num_categories"] for node in parents_disc]
        log_lik, dof = conditional_gaussian_likelihood(
            df, node, parents_disc, parents_cont, levels, dag.nodes(data=True)
        )
        return 2 * log_lik - sparsity * dof * np.log(df.shape[0])

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return conditional_gaussian_local_score(node, parents)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = conditional_gaussian_local_score(node, parents)
        score += temp
    return score


score_dict_global = dict()
is_calculated = False
dot_matrix = None
col_map = {}


def linear_gaussian_score_iid(df, dag, penalty=2, local=False, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def linear_gaussian_local_score_iid(node, parents):
        parents_data = df[parents].to_numpy()
        parents_data = sm.add_constant(parents_data, prepend=False)
        node_data = df[node].to_numpy()
        result = sm.OLS(node_data, parents_data).fit()
        return 2 * result.llf - penalty * (parents_data.shape[1] + 1) * np.log(
            parents_data.shape[0]
        )

    #         return -result.bic

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return linear_gaussian_local_score_iid(node, parents)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = linear_gaussian_local_score_iid(node, parents)
        score += temp
    return score


def polynomial_2_gaussian_score_iid(df, dag, penalty=2, local=False, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def polynomial_gaussian_local_score_iid(node, parents):
        parents_data = df[parents].to_numpy()
        if parents_data.shape[1] == 0:
            parents_data = sm.add_constant(parents_data, prepend=False)
        else:
            polynomial_features = PolynomialFeatures(degree=2)
            parents_data = polynomial_features.fit_transform(parents_data)

        node_data = df[node].to_numpy()

        result = linear_model.BayesianRidge(compute_score=True).fit(parents_data, node_data)
        return 2 * result.scores_[-1] - penalty * (parents_data.shape[1] + 1) * np.log(
            parents_data.shape[0]
        )

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return polynomial_gaussian_local_score_iid(node, parents)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = polynomial_gaussian_local_score_iid(node, parents)
        score += temp
    return score


def polynomial_3_gaussian_score_iid(df, dag, penalty=2, local=False, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def polynomial_gaussian_local_score_iid(node, parents):
        parents_data = df[parents].to_numpy()
        if parents_data.shape[1] == 0:
            parents_data = sm.add_constant(parents_data, prepend=False)
        else:
            polynomial_features = PolynomialFeatures(degree=3)
            parents_data = polynomial_features.fit_transform(parents_data)

        node_data = df[node].to_numpy()

        result = linear_model.BayesianRidge(compute_score=True).fit(parents_data, node_data)
        return 2 * result.scores_[-1] - penalty * (parents_data.shape[1] + 1) * np.log(
            parents_data.shape[0]
        )

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return polynomial_gaussian_local_score_iid(node, parents)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = polynomial_gaussian_local_score_iid(node, parents)
        score += temp
    return score


def linear_splines_score_iid(df, dag, local=False, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    def linear_splines_local_score_iid(node, parents):
        local_data = df[parents]
        local_data = sm.add_constant(local_data, prepend=False)
        bs = BSplines(local_data, df=[3] * local_data.shape[1], degree=[1] * local_data.shape[1])

        local_data[node] = df[node]
        formula = node + " ~ "
        for col in parents:
            formula += col + " + "
        formula += "const"
        print(formula)
        gam_bs = GLMGam.from_formula(formula, data=local_data, smoother=bs)
        res_bs = gam_bs.fit()
        bic = 2 * res_bs.llf - (len(parents) + 1) * np.log(df.shape[0])
        return bic

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return linear_splines_local_score_iid(node, parents)

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = linear_splines_local_score_iid(node, parents)
        score += temp
    return score


def bge_score(df, dag, local=False, **kwargs):
    score = 0
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    if local:
        if "node" not in kwargs or "parents" not in kwargs:
            raise Exception("Must specify a node and its parents for calculating local score")
        node = kwargs.get("node")
        parents = kwargs.get("parents")
        return BGe(ContinuousData(df)).bge_score(node, parents)[0]

    nodes = list(dag.nodes())

    for node in nodes:
        parents = list(dag.predecessors(node))
        temp = BGe(ContinuousData(df)).bge_score(node, parents)[0]
        score += temp
    return score


def bic_score_discrete(G, D):
    """Calculates BIC score of a graph and discrete data.

    Args:
        G: graph (as a networkx.Graph or networkx.DiGraph)
        D: data (as a pandas.DataFrame), shape = num_samples * num_nodes

    Returns:
        BIC score
    """

    def local_bic_score(np_data, target, parents):
        """Computes a score that measures how much a given variable is influenced by a given list of potential parents."""
        sample_size = np_data.shape[0]

        # build dictionary and populate
        count = dict()
        for data_ind in range(sample_size):
            parent_combination = tuple(np_data[data_ind, parents].reshape(1, -1)[0])
            self_value = tuple(np_data[data_ind, target].reshape(1, -1)[0])
            if parent_combination in count:
                if self_value in count[parent_combination]:
                    count[parent_combination][self_value] += 1.0
                else:
                    count[parent_combination][self_value] = 1.0
            else:
                count[parent_combination] = dict()
                count[parent_combination][self_value] = 1.0

        # compute likelihood
        log_lik = 0.0
        num_parent_state = np.prod(np.amax(np_data[:, parents], axis=0) + 1)
        num_self_state = np.amax(np_data[:, target], axis=0) + 1
        for parents_state in count:
            local_count = sum(count[parents_state].values())
            for self_state in count[parents_state]:
                log_lik += count[parents_state][self_state] * (
                    log(count[parents_state][self_state] + 0.1) - log(local_count)
                )

        score = log_lik - 0.5 * log(sample_size) * num_parent_state * (num_self_state - 1)
        return score

    D = D.to_numpy()
    G = nx.to_numpy_matrix(G)
    num_var = G.shape[0]
    bic_score = 0
    for i in range(num_var):
        parents = np.where(G[:, i] != 0)
        bic_score += local_bic_score(D, i, parents)
    return bic_score
