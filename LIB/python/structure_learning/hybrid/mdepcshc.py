
# sys.path.append('../python/')
from python.ci_tests import fast_conditional_ind_test

# sys.path.append('../python/structure_learning/')
from python.structure_learning.constraint_based.mdepcs import MDEPCS

# sys.path.append('../python/structure_learning/score_based/')
from python.structure_learning.score_based.hc import Hill_Climb


class MDEPCSHC:
    def __init__(
        self,
        data,
        nodes,
        score,
        alpha=0.1,
        cit=fast_conditional_ind_test,
        verbose=False,
        tabu_length=10,
        max_indegree=None,
        epsilon=-1e4,
        max_iter=300,
        max_process=5,
    ):
        self.data = data
        self.nodes = nodes
        self.alpha = alpha
        self.cit = cit
        self.tabu_length = tabu_length
        self.max_indegree = max_indegree
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_process = max_process
        self.score = score
        self.verbose = verbose
        self.logger = None
        skel = MDEPCS(
            self.data, self.nodes, self.alpha, self.cit, self.verbose, self.max_process
        ).mdepcs()
        return Hill_Climb(
            self.data,
            self.nodes,
            self.tabu_length,
            self.max_indegree,
            self.epsilon,
            self.max_iter,
            score=self.score,
            white_list=skel.to_undirected().edges(),
        )._estimate()
