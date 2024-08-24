
# sys.path.append('../python/')
from python.ci_tests import fast_conditional_ind_test

# sys.path.append('../python/structure_learning/')
from python.structure_learning.constraint_based.mmpc import MMPC

# sys.path.append('../python/structure_learning/score_based/')
from python.structure_learning.score_based.hc import hill_climb


def mmhc(
    data,
    nodes,
    alpha=0.05,
    cit=fast_conditional_ind_test,
    verbose=False,
    tabu_length=0,
    max_indegree=None,
    epsilon=1e-4,
    max_iter=1e3,
    score=None,
    max_process=5,
):
    skel = MMPC(data, nodes, alpha, cit, verbose, max_process=max_process).mmpc()
    return hill_climb(
        data,
        nodes,
        tabu_length,
        max_indegree,
        epsilon,
        max_iter,
        score=score,
        white_list=skel.to_undirected().edges(),
    )
