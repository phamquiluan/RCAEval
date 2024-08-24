from itertools import combinations

import networkx as nx
from pgmpy.utils.mathext import powerset

# sys.path.append('python/structure_learning/')
from python.ci_tests import fast_conditional_ind_test

# sys.path.append('python/structure_learning/score_based/')
from python.structure_learning.score_based.hc import Hill_Climb


def Hybrid_HPC(data, cit, nodes, alpha):
    """Performs H2PC on the data set and returns set of neighbors for each node

    Args:
        data: pandas Dataframe containing samples
        cit: Conditional independence test
        nodes: dictionary containing the node type
        alpha: significance level for the conditional independence test

    Returns:
        skeleton of the Bayesian network consisiting of the nodes in dataset
    """

    cond = dict()

    def helper(X, Y, Z):
        """Returns the CIT value for X, Y conditioned on Z

        Args:
            X,Y: The pair between which conditional indepence is to be checked
            Z: List containing the set of nodes on which the independence test is to be conditioned on

        Returns:
            p-value obtained after independence test
        """

        if not ((X, Y, tuple(Z)) in cond):
            cond[(X, Y, tuple(Z))] = cit(data, X, Y, Z, nodes=nodes)
        return cond[(X, Y, tuple(Z))]

    def DEPCS(target):
        """Performs a data efficient search to get the parent children set of the target node

        Args:
            target: node for which the parent children superset is to be searched

        Returns:
            PCS: parent children superset of the target node
            sep_set: separating set for the nodes independent of the target node
        """
        sep_set = dict()
        PCS = list(set(nodes_list) - set([target]))
        temp1 = []
        for X in PCS:
            Z = []
            if helper(target, X, Z) > alpha:
                temp1.append(X)
                sep_set[X] = []

        PCS = list(set(PCS) - set(temp1))
        temp1 = []
        for X in PCS:
            temp2 = set(PCS) - set(temp1)
            sep_set[X] = []
            for Z in set(temp2) - set([X]):
                if helper(target, X, [Z]) > alpha:
                    sep_set[X].append(Z)
            if not len(sep_set[X]) == 0:
                temp1.append(X)

        PCS = list(set(PCS) - set(temp1))
        return PCS, sep_set

    def DESPS(target, PCS, sep_set):
        """Performs a data efficient search to get the spouse set of the target node

        Args:
            target: node for which the spouse set is to be searched
            PCS: parent children superset of the target node
            sep_set: list containing the separating set of the nodes independent from the target node

        Returns:
            SPS: Spouse set of the target node
        """
        nodes_list = [node[0] for node in list(nodes)]
        SPS = []
        for X in PCS:
            for Y in set(nodes_list) - set([target] + PCS):
                if helper(target, Y, list(set(sep_set[Y] + [X]))) < alpha:
                    SPS.append(Y)

            SPS = list(set(SPS))
            temp1 = []
            for Y in SPS:
                temp2 = set(SPS) - set(temp1)
                for Z in set(temp2) - set([Y]):
                    if helper(target, Y, list(set([Z] + [X]))) > alpha:
                        temp1.append(Y)
                        break
            SPS = list(set(SPS) - set(temp1))

        return SPS

    def Inter_IAPC(target, valid_nodes):
        """Performs weak conditional independence test between a target node and its probable neighbors nodes

        Args:
            target: node for which the parent children set is to be searched
            valid_nodes: probable neighbors of the target node

        Returns:
            PC: list containing the parent children set of the target node
        """

        MB = []
        change = True
        while change:
            MBprev = MB
            min_assoc = alpha
            best_Y = None
            for Y in set(valid_nodes) - set([target] + MB):
                assoc = helper(target, Y, MB)
                if assoc < min_assoc:
                    best_Y = Y
                    min_assoc = assoc
            if best_Y is not None:
                MB.append(best_Y)
            temp = []
            for X in MB:
                if helper(target, X, list(set(MB) - set([X]))) > alpha:
                    temp.append(X)
            MB = list(set(MB) - set(temp))
            change = not (set(MB) == set(MBprev))
        PC = MB
        for X in MB:
            assoc = max(helper(target, X, Z) for Z in powerset(list(set(MB) - set([X]))))
            if assoc > alpha:
                PC.remove(X)
        return PC

    def HPC(target):
        """Performs HPC on the target node

        Args:
            target: node for which the parent children set is to be searched

        Returns:
            Final parent children set of the target node
        """

        PCS, sep_set = DEPCS(target)
        SPS = DESPS(target, PCS, sep_set)
        PC = Inter_IAPC(target, list(set([target] + PCS + SPS)))
        temp = PC
        for X in set(PCS) - set(temp):
            if target in Inter_IAPC(X, list(set([target] + PCS + SPS))):
                PC.append(X)
        return PC

    nodes_list = [node[0] for node in list(nodes)]
    skel = nx.Graph()
    HPCdict = dict()
    for X, Y in combinations(nodes_list, 2):
        print(X, Y)
        if X not in HPCdict:
            HPCdict[X] = HPC(X)
        if Y not in HPCdict:
            HPCdict[Y] = HPC(Y)
        if X in HPCdict[Y] and Y in HPCdict[X]:
            skel.add_edge(X, Y)
    return skel


def H2HC(
    data,
    nodes,
    alpha=0.05,
    cit=fast_conditional_ind_test,
    verbose=False,
    tabu_length=0,
    max_indegree=None,
    epsilon=1e-4,
    max_iter=1e2,
    score=None,
):
    skel = Hybrid_HPC(data, cit, nodes, alpha)
    return Hill_Climb(
        data,
        nodes,
        tabu_length,
        max_indegree,
        epsilon,
        max_iter,
        score=score,
        white_list=skel.to_undirected().edges(),
    )._estimate()
