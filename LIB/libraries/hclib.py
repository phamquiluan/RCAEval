from itertools import permutations

import networkx as nx

# sys.path.append('../python/')
from python.scores import bdeu_score, conditional_gaussian_score


class Hill_Climb:
    def __init__(
        self,
        data,
        nodes,
        score=conditional_gaussian_score,
        black_list=None,
        white_list=None,
        start=None,
        max_iter=1e2,
        tabu_length=0,
        max_indegree=None,
    ):
        """Args:
            data: pandas dataframe object (No missing values assumed)
            nodes: dict of nodes and their types
            score: the scoring function to be used
            black_list: list of edges to not be considered
            white_list: list of edges to be considered for edge_addition
            start: initial state of graph to start with
            max_iter: maximum iterations to run the algorithm for
            tabu_length: maximum operations to store that is to not be repeated
            max_indegree: maximum parents allowed for a node

        Returns:
            object of class
        """
        if start is None:
            self.start = nx.DiGraph()
            self.start.add_nodes_from(nodes)
        elif not set(start.nodes()) == set(nodes):
            raise ValueError(
                "'start' should be a DAG with the same variables as the data set, or 'None'."
            )
        else:
            self.start = start
        self.score = score
        self.data = data
        self.white_list = white_list
        self.black_list = black_list
        self.max_iter = max_iter
        self.nodes = nodes
        self.max_indegree = max_indegree
        self.tabu_length = tabu_length

    def _legal_operations(self, model, tabu_list=[], black_list_operation=[]):
        """Finds out the best operation that can be applied to the model to improve the score"""
        nodes_list = list(model.nodes())
        potential_new_edges = (
            set(permutations(nodes_list, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )

        for X, Y in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y)])):
                operation = ("+", (X, Y))
                if (
                    operation not in tabu_list
                    and (self.black_list is None or (X, Y) not in self.black_list)
                    and (self.white_list is None or (X, Y) in self.white_list)
                    and operation not in black_list_operation
                ):
                    old_parents = list(model.predecessors(Y))
                    new_parents = old_parents + [X]
                    if self.max_indegree is None or len(new_parents) <= self.max_indegree:
                        score_delta = self.score(
                            self.data, model, local=True, node=Y, parents=new_parents
                        ) - self.score(self.data, model, local=True, node=Y, parents=old_parents)
                        yield (operation, score_delta)

        for X, Y in model.edges():  # (2) remove single edge
            operation = ("-", (X, Y))
            if operation not in tabu_list and operation not in black_list_operation:
                old_parents = list(model.predecessors(Y))
                new_parents = old_parents[:]
                new_parents.remove(X)
                score_delta = self.score(
                    self.data, model, local=True, node=Y, parents=new_parents
                ) - self.score(self.data, model, local=True, node=Y, parents=old_parents)
                yield (operation, score_delta)

        for X, Y in model.edges():  # (3) flip single edge
            new_edges = list(model.edges()) + [(Y, X)]
            new_edges.remove((X, Y))
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                operation = ("flip", (X, Y))
                if (
                    operation not in tabu_list
                    and ("flip", (Y, X)) not in tabu_list
                    and (self.black_list is None or (Y, X) not in self.black_list)
                    and (self.white_list is None or (Y, X) in self.white_list)
                    and operation not in black_list_operation
                ):
                    old_X_parents = list(model.predecessors(X))
                    old_Y_parents = list(model.predecessors(Y))
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if self.max_indegree is None or len(new_X_parents) <= self.max_indegree:
                        score_delta = (
                            self.score(self.data, model, local=True, node=X, parents=new_X_parents)
                            + self.score(
                                self.data, model, local=True, node=Y, parents=new_Y_parents
                            )
                            - self.score(
                                self.data, model, local=True, node=X, parents=old_X_parents
                            )
                            - self.score(
                                self.data, model, local=True, node=Y, parents=old_Y_parents
                            )
                        )
                        yield (operation, score_delta)

    def _estimate(self):
        """entry point of algorithm. Does the best operations to the model to achieve a better fit to the data"""
        tabu_list = []
        current_model = self.start
        black_list_operation = []

        iter_no = 0
        while iter_no <= self.max_iter:
            iter_no += 1

            best_score_delta = None
            if self.score == bdeu_score:
                best_score_delta = 0
            else:
                best_score_delta = -1e4
            best_operation = None

            for operation, score_delta in self._legal_operations(
                current_model, tabu_list, black_list_operation
            ):
                if score_delta > best_score_delta:
                    best_operation = operation
                    best_score_delta = score_delta

            if self.score == conditional_gaussian_score:
                if best_score_delta == 0.0:
                    black_list_operation.append(best_operation)
                    continue
                else:
                    black_list_operation = []

            if best_operation is None:
                break
            elif best_score_delta == 0:
                black_list_operation.append(best_operation)
            elif best_operation[0] == "+":
                current_model.add_edge(*best_operation[1])
                tabu_list = ([("-", best_operation[1])] + tabu_list)[: self.tabu_length]
            elif best_operation[0] == "-":
                current_model.remove_edge(*best_operation[1])
                tabu_list = ([("+", best_operation[1])] + tabu_list)[: self.tabu_length]
            elif best_operation[0] == "flip":
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list = ([best_operation] + tabu_list)[: self.tabu_length]

        return current_model
