import logging
import os
from multiprocessing import Manager, Pool

import networkx as nx
from pgmpy.estimators.CITests import chi_square
from pgmpy.utils.mathext import powerset
from python.bnutils import one_hot_encoder

# sys.path.append('../python')
from python.ci_tests import chi_square_test, fast_conditional_ind_test


class MMPC:
    def __init__(
        self, data, nodes, alpha=0.05, cit=fast_conditional_ind_test, verbose=False, max_process=5
    ):
        self.data = data
        if cit == chi_square_test:
            self.data = self.data.astype(int)
        self.data_str = self.data.copy()
        self.data_str.columns = self.data_str.columns.astype(str)
        self.nodes = nodes
        self.nodes_list = [node[0] for node in nodes]
        self.alpha = alpha
        self.assoc = dict()
        self.max_process = max_process
        self.cit = cit
        self.verbose = verbose
        self.logger = None
        if self.verbose:
            self.log_folder_path = "../../../logs/mmpc"
            if not os.path.exists(self.log_folder_path):
                os.mkdir(self.log_folder_path)
        if self.cit == fast_conditional_ind_test:
            # create dictionary of one-hot encoded features in dataframe
            self.onehot_dict = {}
            for node in self.nodes:
                if node[1]["type"] == "disc":
                    self.onehot_dict[node[0]] = one_hot_encoder(data[:][node[0]].to_numpy())

    def mmpc(self):
        "Main entry point into the algorithm. Returns skeleton by learning parent-child set of each node"
        m = Manager()
        q = m.Queue()
        pool = Pool(min(self.max_process, len(self.nodes_list)))
        args_list = list(zip(self.nodes_list, [q] * len(self.nodes_list)))
        pool.map(self.parallel, args_list)
        pool.close()
        pool.join()
        q.put("EOQ")

        neighbors = dict()
        while True:
            neighbor = q.get()
            if neighbor == "EOQ":
                break
            neighbors[neighbor[0]] = neighbor[1]

        # correct for false positives
        for node in self.nodes_list:
            for neigh in neighbors[node].copy():
                if node not in neighbors[neigh]:
                    neighbors[node].remove(neigh)

        skel = nx.Graph()
        skel.add_nodes_from(self.nodes)
        for node in self.nodes_list:
            skel.add_edges_from([(node, neigh) for neigh in neighbors[node]])
        return skel

    def parallel(self, args):
        "performs forward and backward phase on a node. Each call to this function is parallelized"
        node = args[0]
        q = args[1]
        neighbors = []
        if self.verbose:
            logging.basicConfig(
                filename=self.log_folder_path + "/process_" + str(node) + ".log",
                format="%(asctime)s %(message)s",
            )
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
        # Forward Phase

        while True:
            new_neighbor, new_neighbor_min_assoc = self.max_min_heuristic(node, neighbors)
            if new_neighbor_min_assoc > 1 - self.alpha:
                neighbors.append(new_neighbor)
            else:
                break
        if self.verbose:
            self.logger.info(
                "[INFO] : End of forward Phase of " + str(node) + " : " + str(neighbors)
            )

        # Backward Phase
        for neigh in neighbors.copy():
            other_neighbors = [n for n in neighbors if n != neigh]
            for sep_set in powerset(other_neighbors):
                if 1 - self.association(node, neigh, sep_set) > self.alpha:
                    neighbors.remove(neigh)
                    break
        if self.verbose:
            self.logger.info(
                "[INFO] : End of backward Phase of " + str(node) + " : " + str(neighbors)
            )
        q.put((node, neighbors))
        return

    def association(self, X, Y, Z):
        "association of X, Y given set Z"
        if not ((X, Y, Z) in self.assoc):
            if self.cit == chi_square_test:
                self.assoc[(X, Y, Z)] = (
                    1 - chi_square(str(X), str(Y), [str(z) for z in Z], self.data_str)[1]
                )
            else:
                self.assoc[(X, Y, Z)] = 1 - self.cit(
                    self.data, X, Y, Z, nodes=self.nodes, onehot_dict=self.onehot_dict
                )
        if self.verbose:
            self.logger.info(
                "[INFO] : assoc call "
                + str(X)
                + " , "
                + str(Y)
                + " , "
                + str(Z)
                + " : "
                + str(self.assoc[(X, Y, Z)])
            )
        return self.assoc[(X, Y, Z)]

    def min_assoc(self, X, Y, Zs):
        "Minimal association of X, Y given any subset of Zs."
        min_assoc_val = 1
        for s in powerset(Zs):
            assoc_val = self.association(X, Y, s)
            if assoc_val <= 1 - self.alpha:
                return assoc_val
            min_assoc_val = min(min_assoc_val, assoc_val)
        return min_assoc_val

    def max_min_heuristic(self, X, Zs):
        "Finds variable that maximizes min_assoc with `node` relative to `neighbors`."
        max_min_assoc = 1 - self.alpha
        best_Y = None

        for Y in set(self.nodes_list) - set(Zs + [X]):
            min_assoc_val = self.min_assoc(X, Y, Zs)
            if min_assoc_val >= max_min_assoc:
                best_Y = Y
                max_min_assoc = min_assoc_val
                break
        return (best_Y, max_min_assoc)
