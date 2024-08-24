
# sys.path.append('../libraries/score_based/')
from libraries.hclib import Hill_Climb

# sys.path.append('../python/')
from python.scores import bdeu_score, conditional_gaussian_score


def hill_climb(
    data,
    nodes,
    tabu_length=0,
    max_indegree=None,
    epsilon=1e-4,
    max_iter=1e2,
    score=bdeu_score,
    black_list=None,
    white_list=None,
    start=None,
):
    "wrapper function for Hill Climb search"
    return Hill_Climb(
        data, nodes, score, black_list, white_list, start, max_iter, tabu_length, max_indegree
    )._estimate()
