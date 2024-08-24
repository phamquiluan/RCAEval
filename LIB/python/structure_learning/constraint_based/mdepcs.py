import logging
import os
from multiprocessing import Pool

import networkx as nx

# sys.path.append('../python/')
from python.bnutils import one_hot_encoder

# sys.path.append('../python/')
from python.ci_tests import fast_conditional_ind_test


class MDEPCS:
    def __init__(
        self, data, nodes, alpha=0.1, cit=fast_conditional_ind_test, verbose=False, max_process=5
    ):
        self.data = data
        self.nodes = nodes
        self.nodes_list = [node[0] for node in nodes]
        self.alpha = alpha
        self.assoc = dict()
        self.cit = cit
        self.PCS_neigh = None
        self.max_process = max_process
        self.verbose = verbose
        self.logger = None
        if self.verbose:
            self.log_folder_path = "../../../logs/mdepcs"
            if not os.path.exists(self.log_folder_path):
                os.mkdir(self.log_folder_path)
        if self.cit == fast_conditional_ind_test:
            # create dictionary of one-hot encoded features in dataframe
            self.onehot_dict = {}
            for node in self.nodes:
                if node[1]["type"] == "disc":
                    self.onehot_dict[node[0]] = one_hot_encoder(data[:][node[0]].to_numpy())

    def helper(self, X, Y, Z):
        if not ((X, Y, tuple(Z)) in self.assoc):
            self.assoc[(X, Y, tuple(Z))] = self.cit(
                self.data, X, Y, Z, nodes=self.nodes, onehot_dict=self.onehot_dict
            )
        return self.assoc[(X, Y, tuple(Z))]

    def PCSuperset(self, target):
        PCS = set(self.nodes_list) - set([target])
        PCS = PCS - set([X for X in PCS if (self.helper(target, X, []) > self.alpha)])
        indep_nodes = []
        for X in PCS:
            sep_set = [
                Z for Z in PCS - set(indep_nodes + [X]) if self.helper(target, X, [Z]) > self.alpha
            ]
            if not len(sep_set) == 0:
                indep_nodes.append(X)
        return list(PCS - set(indep_nodes))

    def parallel(self, node):
        "performs forward and backward phase on a node. Each call to this function is parallelized"
        if self.verbose:
            logging.basicConfig(
                filename=self.log_folder_path + "/process_" + str(node) + ".log",
                format="%(asctime)s %(message)s",
            )
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
        # PCSuperset Phase
        neighbors = self.PCSuperset(node)
        if self.verbose:
            self.logger.info("[INFO] : End of PCSuperset of " + str(node) + " : " + str(neighbors))
        return neighbors

    def get_neighbors_set(self):
        pool = Pool(min(self.max_process, len(self.nodes_list)))
        self.PCS_neigh = pool.map(self.parallel, self.nodes_list)
        pool.close()
        pool.join()
        return

    def mdepcs(self):
        if self.PCS_neigh is None:
            self.get_neighbors_set()
        neighbors = self.PCS_neigh.copy()
        for node in self.nodes_list:
            for neigh in neighbors[node].copy():
                if node not in neighbors[neigh]:
                    neighbors[node].remove(neigh)

        skel = nx.Graph()
        skel.add_nodes_from(self.nodes)
        for node in self.nodes_list:
            skel.add_edges_from([(node, neigh) for neigh in neighbors[node]])

        return skel

    def skel_union(self):
        if self.PCS_neigh is None:
            self.get_neighbors_set()
        neighbors = self.PCS_neigh.copy()
        skel = nx.Graph()
        skel.add_nodes_from(self.nodes)
        for node in self.nodes_list:
            skel.add_edges_from([(node, neigh) for neigh in neighbors[node]])

        return skel

    def get_PCS_neighbors(self):
        return self.PCS_neigh
