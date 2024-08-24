import itertools
import logging
import time

import dill
import networkx as nx
from libraries.FGES import graph_util
from libraries.FGES.knowledge import Knowledge
from libraries.FGES.meekrules import MeekRules
from python.scores import (
    bdeu_score,
    bge_score,
    bic_score_discrete,
    conditional_gaussian_score,
    linear_gaussian_score_iid,
    polynomial_2_gaussian_score_iid,
    polynomial_3_gaussian_score_iid,
)
from sortedcontainers import SortedListWithKey

logging.basicConfig(filename="fges.log", format="%(asctime)s %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Arrow:
    __slots__ = ["a", "b", "na_y_x", "h_or_t", "bump", "index"]

    def __init__(self, a, b, na_y_x, hOrT, bump, arrow_index):
        self.a = a
        self.b = b
        self.na_y_x = na_y_x
        self.h_or_t = hOrT
        self.bump = bump
        self.index = arrow_index


class FGES:
    def __init__(
        self,
        variables,
        nodes,
        data,
        filename="",
        checkpoint_frequency=0,
        save_name=None,
        knowledge=None,
        verbose=False,
        score=conditional_gaussian_score,
    ):
        self.top_graphs = []
        # self.last_checkpoint = time.time()
        # How often fges-py will save a checkpoint of the data
        self.checkpoint_frequency = checkpoint_frequency
        self.save_name = save_name
        # List of the nodes, in order
        self.variables = variables
        self.nodes = nodes
        self.data = data
        # Meant to be a map from the node to its column in the dataset,
        # but in this implementation, this should always be a map
        # from x -> x, i.e. {1:1, 2:2, ...}
        # self.node_dict = {}
        self.sorted_arrows = SortedListWithKey(key=lambda val: -val.bump)
        self.arrow_dict = {}
        self.arrow_index = 0
        self.total_score = 0
        self.local_score = score
        # Only needed for their `heuristic speedup`, it tells
        # you if two edges even have an effect on each other
        # the way we use this is effect_edges_graph[node] gives you
        # an iterable of nodes {w_1, w_2, w_3...} where node and
        # w_i have a non-zero total effect
        self.effect_edges_graph = {}
        self.cycle_bound = -1
        self.stored_neighbors = {}
        self.graph = None
        self.removed_edges = set()
        self.filename = filename
        self.in_bes = False
        self.knowledge = knowledge
        self.verbose = verbose

    def set_knowledge(self, knowledge):
        if not isinstance(knowledge, Knowledge):
            raise TypeError("knowledge must be of type Knowledge")
        else:
            self.knowledge = knowledge

    def get_dict(self):
        return {
            "graph": self.graph,
            "filename": self.filename,
            "nodes": len(self.variables),
            "knowledge": self.knowledge,
        }

    @classmethod
    def load_checkpoint(cls, filename):
        with open(filename, "rb") as f:
            return dill.load(f)

    #     def do_profile(follow=[]):
    #         def inner(func):
    #             def profiled_func(*args, **kwargs):
    #                 try:
    #                     profiler = LineProfiler()
    #                     profiler.add_function(func)
    #                     for f in follow:
    #                         profiler.add_function(f)
    #                     profiler.enable_by_count()
    #                     return func(*args, **kwargs)
    #                 finally:
    #                     profiler.print_stats()
    #             return profiled_func
    #         return inner

    def fes(self):
        """The basic workflow of FGES is to first consider add all edges with positive bump, as defined
        by the SEMBicScore, to a sorted list (sorted by bump).
        Edges are popped off this list and added to the graph, after which point the Meek rules are utilized to
        orient edges in the graph that can be oriented. Then, all relevant bumps are recomputed and
        the list is resorted. This process is repeated until there remain no edges to add with positive bump.
        """

        while len(self.sorted_arrows) > 0:
            max_bump_arrow = self.sorted_arrows.pop(
                0
            )  # Pops the highest bump edge off the sorted list
            x = max_bump_arrow.a
            y = max_bump_arrow.b
            # print("Popped arrow: " + str(x) + " -> " + str(y))

            if graph_util.adjacent(self.graph, x, y):
                continue

            na_y_x = graph_util.get_na_y_x(self.graph, x, y)

            # TODO: max degree checks
            # print(na_y_x)

            if max_bump_arrow.na_y_x != na_y_x:
                continue

            # print("Past crucial step")

            if not graph_util.get_t_neighbors(self.graph, x, y).issuperset(max_bump_arrow.h_or_t):
                continue

            if not self.valid_insert(x, y, max_bump_arrow.h_or_t, na_y_x):
                # print("Not valid insert")
                continue

            T = max_bump_arrow.h_or_t
            bump = max_bump_arrow.bump

            # TODO: Insert should return a bool that we check here
            inserted = self.insert(x, y, T, bump)  # Insert highest bump edge into the graph
            if not inserted:
                continue

            self.total_score += bump
            # print("Edge set before reapplying orientation: " + str(self.graph.edges()))
            visited_nodes = self.reapply_orientation(
                x, y, None
            )  # Orient edges appropriately following insertion
            # print("Edge set after reapplying orientation: " + str(self.graph.edges()))
            to_process = set({})

            # check whether the (undirected) neighbors of each node in
            # visited_nodes changed compared to stored neighbors
            for node in visited_nodes:
                # gets undirected neighbors
                new_neighbors = graph_util.neighbors(self.graph, node)
                stored_neighbors = self.stored_neighbors.get(node)
                if stored_neighbors != new_neighbors:
                    to_process.add(node)  # Reevaluate neighbor nodes

            to_process.add(x)  # Reevaluate edges relating to node x
            to_process.add(y)  # Reevaluate edges relating to node y

            self.reevaluate_forward(to_process, max_bump_arrow)  # Do actual reevaluation

    def bes(self):
        """BES removes edges from the graph generated by FGES, as added edges can now have negative bump in light
        of the additions to the graph after those edges were added."""

        while len(self.sorted_arrows) > 0:
            if (
                self.checkpoint_frequency > 0
                and (time.time() - self.last_checkpoint) > self.checkpoint_frequency
            ):
                self.create_checkpoint()
                self.last_checkpoint = time.time()

            arrow = self.sorted_arrows.pop(0)
            x = arrow.a
            y = arrow.b

            if (
                (not (arrow.na_y_x == graph_util.get_na_y_x(self.graph, x, y)))
                or (not graph_util.adjacent(self.graph, x, y))
                or (graph_util.has_dir_edge(self.graph, y, x))
            ):
                continue
                if not self.valid_delete(x, y, arrow.h_or_t, arrow.na_y_x):
                    continue

            H = arrow.h_or_t
            bump = arrow.bump

            self.delete(x, y, H)

            meek_rules = MeekRules(knowledge=self.knowledge)
            meek_rules.orient_implied_subset(self.graph, set([x, y]))

            self.total_score += bump
            self.clear_arrow(x, y)
            if self.verbose:
                print("BES: Removed arrow " + str(x) + " -> " + str(y) + " with bump -" + str(bump))
            logger.info(
                "BES: Removed arrow " + str(x) + " -> " + str(y) + " with bump -" + str(bump)
            )
            visited = self.reapply_orientation(x, y, H)

            to_process = set()

            for node in visited:
                neighbors = graph_util.neighbors(self.graph, node)
                str_neighbors = self.stored_neighbors[node]

                if str_neighbors != neighbors:
                    to_process.update([node])

            to_process.add(x)
            to_process.add(y)
            to_process.update(graph_util.get_common_adjacents(self.graph, x, y))

            # TODO: Store graph
            self.reevaluate_backward(to_process)

    #     @do_profile(follow=[])
    def search(self):
        """
        The main entry point into the algorithm.
        """
        # Create an empty directed graph
        if self.graph is None:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(self.nodes)
            self.add_required_edges()
            self.initialize_forward_edges_from_empty_graph()  # Adds all edges that have positive bump

        # Step 1: Run FES and BES with heuristic
        # mode. The mode is used in reevaluate_forward
        self.mode = "heuristic"
        if not self.in_bes:
            self.fes()
            self.sorted_arrows = SortedListWithKey(key=lambda val: -val.bump)
            self.arrow_dict = {}
            self.stored_neighbors = {}
            self.initialize_arrows_backwards()
            self.in_bes = True

            if self.checkpoint_frequency > 0:
                self.create_checkpoint()

        self.bes()

        # Step 1: Run FES and BES with covernoncolliders
        # mode. The mode is used in reevaluate_forward

        # self.mode = "covernoncolliders"
        # self.fes()
        # self.bes()
        # print(self.graph.edges())
        return self.get_dict()

    def initialize_arrows_backwards(self):
        for node_1, node_2 in self.graph.edges():
            if self.knowledge is not None and not self.knowledge.no_edge_required(node_1, node_2):
                continue
            self.clear_arrow(node_1, node_2)
            self.clear_arrow(node_2, node_1)

            self.calculate_arrows_backward(node_1, node_2)

            self.stored_neighbors[node_1] = graph_util.neighbors(self.graph, node_1)
            self.stored_neighbors[node_2] = graph_util.neighbors(self.graph, node_2)

    def calculate_arrows_backward(self, a, b):
        """Finds all edges with negative bump"""

        if self.knowledge is not None and not self.knowledge.no_edge_required(a, b):
            return

        na_y_x = graph_util.get_na_y_x(self.graph, a, b)
        _na_y_x = list(na_y_x)
        _depth = len(_na_y_x)

        for i in range(_depth + 1):
            choices = itertools.combinations(range(0, _depth), i)
            for choice in choices:
                diff = set([_na_y_x[k] for k in choice])
                h = set(_na_y_x)
                h = h - diff

                if self.knowledge is not None and not self.valid_set_by_knowledge(b, h):
                    continue

                bump = self.delete_eval(a, b, diff, na_y_x)

                if bump > 0:
                    if self.verbose:
                        print(
                            "Evaluated removal of an arrow "
                            + str(a)
                            + " -> "
                            + str(b)
                            + " with bump: "
                            + str(bump)
                        )
                    logger.info(
                        "[INFO] Evaluated removal of an arrow "
                        + str(a)
                        + " -> "
                        + str(b)
                        + " with bump: "
                        + str(bump)
                    )
                    self.add_arrow(a, b, na_y_x, h, bump)

    def delete_eval(self, x, y, diff, na_y_x):
        """Evaluates the bump of removing edge X-->Y"""
        a = set(diff)
        a.update(graph_util.get_parents(self.graph, y))
        a = a - {x}
        return -1 * self.score_graph_change(y, a, x)

    def reevaluate_forward(self, to_process, arrow):
        # print("Re-evaluate forward with " + str(to_process) + " " + str(arrow))
        for node in to_process:
            if self.mode == "heuristic":
                nzero_effect_nodes = self.effect_edges_graph.get(node)
                # print("Re-evaluate forward. Currently on node: " + str(node))
                # print("nzero-effect-nodes: " + str(nzero_effect_nodes))
            elif self.mode == "covernoncolliders":
                g = set()
                for n in graph_util.adjacent_nodes(self.graph, node):
                    for m in graph_util.adjacent_nodes(self.graph, n):
                        if graph_util.adjacent(self.graph, n, m):
                            continue

                        if graph_util.is_def_collider(self.graph, m, n, node):
                            continue

                        g.update(m)

                nzero_effect_nodes = list(g)
            if nzero_effect_nodes is not None:
                for w in nzero_effect_nodes:
                    if w == node:
                        continue
                    if not graph_util.adjacent(self.graph, node, w):
                        self.clear_arrow(w, node)
                        self.calculate_arrows_forward(w, node)

    def reevaluate_backward(self, to_process):
        for node in to_process:
            self.stored_neighbors[node] = graph_util.neighbors(self.graph, node)
            adjacent_nodes = graph_util.adjacent_nodes(self.graph, node)

            for adj_node in adjacent_nodes:
                if graph_util.has_dir_edge(self.graph, adj_node, node):
                    self.clear_arrow(adj_node, node)
                    self.clear_arrow(node, adj_node)

                    self.calculate_arrows_backward(adj_node, node)
                elif graph_util.has_undir_edge(self.graph, adj_node, node):
                    self.clear_arrow(adj_node, node)
                    self.clear_arrow(node, adj_node)
                    self.calculate_arrows_backward(adj_node, node)
                    self.calculate_arrows_backward(node, adj_node)

    def reapply_orientation(self, x, y, new_arrows):
        to_process = {x, y}
        if new_arrows is not None:
            to_process.update(new_arrows)

        return self.meek_orient_restricted(to_process)

    def meek_orient_restricted(self, nodes):
        # Runs meek rules on the changed adjacencies
        meek_rules = MeekRules(undirect_unforced_edges=True, knowledge=self.knowledge)
        meek_rules.orient_implied_subset(self.graph, nodes)
        return meek_rules.get_visited()

    def valid_insert(self, x, y, T, na_y_x):
        union = set(T)

        if self.knowledge is not None:
            if self.knowledge.is_forbidden(x, y):
                return False
            for node in union:
                if self.knowledge.is_forbidden(node, y):
                    return False

        if na_y_x != set([]):
            union.update(na_y_x)
        return graph_util.is_clique(
            self.graph, union
        ) and not graph_util.exists_unblocked_semi_directed_path(
            self.graph, y, x, union, self.cycle_bound
        )

    def valid_delete(self, x, y, H, na_y_x):
        if self.knowledge is not None:
            for h in H:
                if self.knowledge.is_forbidden(x, h):
                    return False
                if self.knowledge.is_forbidden(y, h):
                    return False

        diff = set(na_y_x)
        diff = diff - H
        return graph_util.is_clique(self.graph, diff)

    def add_required_edges(self):
        """Tetrad implementation is really confusing and seems to be
        mostly checks to ensure required edges don't form a cycle"""
        if self.knowledge is None:
            return

        for edge in self.knowledge.required_edges:
            # Make sure the required edges aren't a cycle
            if not edge[0] in graph_util.get_ancestors(self.graph, edge[1]):
                graph_util.remove_dir_edge(self.graph, edge[1], edge[0])
                graph_util.add_dir_edge(self.graph, edge[0], edge[1])
                if self.verbose:
                    print("Adding edge from knowledge: {edge[0]} -> {edge[1]}")
                logger.info("Adding edge from knowledge: {edge[0]} -> {edge[1]}")
        for edge in self.knowledge.required_connections:
            graph_util.add_undir_edge(self.graph, edge[1], edge[0])

    def initialize_two_step_edges(self, nodes):
        for node in nodes:
            g = set()

            for n in graph_util.adjacent_nodes(self.graph, node):
                for m in graph_util.adjacent_nodes(self.graph, n):
                    if node == m:
                        continue

                    if graph_util.adjacent(self.graph, node, m):
                        continue

                    if graph_util.is_def_collider(self.graph, m, n, node):
                        continue

                    g.update(m)

            for x in g:
                assert x is not node
                if self.knowledge is not None:
                    if self.knowledge.is_forbidden(node, x) or self.knowledge.is_forbidden(x, node):
                        continue
                    # again, what's the point?
                    if not self.valid_set_by_knowledge(node, set()):
                        continue

                # TODO: Adjacencies

                if (x, node) in self.removed_edges:
                    continue

                self.calculate_arrows_forward(x, node)

    def initialize_forward_edges_from_empty_graph(self):
        """
        Initializes the state of the graph before executing fes()
        This is called from search().
        TODO:
        - This seems easily parallelizable
        - There is a check for symmetricFirstStep here, which essentially
        adds the bump between child <-> parent instead of just parent <-> child.
        - This code also checks for boundGraph, which directly enforces
        what kind of edges can be `bound`. In effect, this is a type of background knowledge.
        Unknowns:
        - Confused by the same-reference emptySet() in the Java implementation for Arrow. Does that
        mean that if one gets modified, all will?
        """
        for i in range(len(self.variables)):
            self.stored_neighbors[self.variables[i]] = set()
            for j in range(i + 1, len(self.variables)):
                if self.knowledge is not None:
                    if self.knowledge.is_forbidden(
                        self.variables[i], self.variables[j]
                    ) and self.knowledge.is_forbidden(self.variables[j], self.variables[i]):
                        continue

                # Not sure.. Need to check
                bump1, bump2 = self.local_score_diff(self.variables[j], self.variables[i])

                if self.verbose:
                    print(
                        "Evaluated starting arrow "
                        + str(self.variables[j])
                        + " -> "
                        + str(self.variables[i])
                        + " with bump: "
                        + str(bump1)
                    )
                logger.info(
                    "[INFO] Evaluated starting arrow "
                    + str(self.variables[j])
                    + " -> "
                    + str(self.variables[i])
                    + " with bump: "
                    + str(bump1)
                    + " "
                    + str(bump2)
                )
                if bump1 >= 0 or bump2 >= 0:
                    self.mark_nonzero_effect(self.variables[i], self.variables[j])
                if bump2 >= 0:
                    parent_node = self.variables[j]
                    child_node = self.variables[i]
                    self.add_arrow(parent_node, child_node, set(), set(), bump2)
                if bump1 >= 0:
                    parent_node = self.variables[i]
                    child_node = self.variables[j]
                    self.add_arrow(parent_node, child_node, set(), set(), bump1)

        if self.verbose:
            print("Initialized forward edges from empty graph")
        logging.info("[INFO] Initialized forward edges from empty graph")

    def mark_nonzero_effect(self, node_1, node_2):
        """
        Adds node_1 to the (instance-wide) effect edges list for node_2,
        and vice versa.
        """
        if self.effect_edges_graph.get(node_1) is None:
            self.effect_edges_graph[node_1] = [node_2]
        else:
            self.effect_edges_graph[node_1].append(node_2)

        if self.effect_edges_graph.get(node_2) is None:
            self.effect_edges_graph[node_2] = [node_1]
        else:
            self.effect_edges_graph[node_2].append(node_1)

    def insert_eval(self, x, y, T, na_y_x):
        """Evaluates bump for adding edge x->y given conditioning sets T and na_y_x"""
        assert x is not y
        _na_y_x = set(na_y_x)
        _na_y_x.update(T)
        _na_y_x.update(graph_util.get_parents(self.graph, y))
        return self.score_graph_change(y, _na_y_x, x)

    def local_score_diff(self, node1, node2):
        bump1 = self.local_score(
            self.data, self.graph, local=True, node=node1, parents=[node2]
        ) - self.local_score(self.data, self.graph, local=True, node=node1, parents=[])
        bump2 = self.local_score(
            self.data, self.graph, local=True, node=node2, parents=[node1]
        ) - self.local_score(self.data, self.graph, local=True, node=node2, parents=[])
        return bump1, bump2

    def insert(self, x, y, T, bump):
        """T is a subset of the neighbors of Y that are not adjacent to
        (connected by a directed or undirected edge) to X, this should
        connect X -> Y and for t in T, direct T -> Y if it's not already
        directed
        Definition 12
        """
        if self.verbose:
            print(
                "Doing an actual insertion with "
                + str(x)
                + " -> "
                + str(y)
                + " with T: "
                + str(T)
                + " and bump: "
                + str(bump)
            )
        logger.info(
            "[INFO] Doing an actual insertion with "
            + str(x)
            + " -> "
            + str(y)
            + " with T: "
            + str(T)
            + " and bump: "
            + str(bump)
        )
        if graph_util.adjacent(self.graph, x, y):
            return False

        # Adds directed edge
        self.graph.add_edge(x, y)

        for node in T:
            graph_util.undir_to_dir(self.graph, node, y)

        return True

    def delete(self, x, y, H):
        # Remove any edge between x and y
        graph_util.remove_dir_edge(self.graph, x, y)
        graph_util.remove_dir_edge(self.graph, y, x)

        # H is the set of neighbors of y that are adjacent to x
        for node in H:
            if graph_util.has_dir_edge(self.graph, node, y) or graph_util.has_dir_edge(
                self.graph, node, x
            ):
                continue

            # Direct the edge y --- node as y --> node
            graph_util.undir_to_dir(self.graph, y, node)

            # If x --- node is undirected, direct it as x --> node
            if graph_util.has_undir_edge(self.graph, x, node):
                graph_util.undir_to_dir(self.graph, x, node)

        self.removed_edges.add((x, y))

    def add_arrow(self, a, b, na_y_x, h_or_t, bump):
        """Add arrow a->b with bump "bump" and conditioning sets na_y_x and h_or_t to sorted arrows list"""
        # print("Added arrow: " + str(a) + " ->  " + str(b) + " with bump " + \
        #  str(bump) + " and na_y_x " + str(na_y_x) + " and h_or_t " + str(h_or_t))
        arrow = Arrow(a, b, na_y_x, h_or_t, bump, self.arrow_index)
        self.sorted_arrows.add(arrow)

        pair = (a, b)
        if self.arrow_dict.get(pair) is None:
            self.arrow_dict[pair] = [arrow]
        else:
            self.arrow_dict[pair].append(arrow)

        self.arrow_index += 1

    def clear_arrow(self, a, b):
        """Remove arrow a->b from sorted arrows list"""
        pair = (a, b)
        # print("Clearing arrow " + str(pair))
        lookup_arrows = self.arrow_dict.get(pair)
        # print(lookup_arrows)
        if lookup_arrows is not None:
            for arrow in lookup_arrows:
                # print("Removing " + str(arrow) + " from sorted_arrows")
                self.sorted_arrows.discard(arrow)

        self.arrow_dict[pair] = None

    def score_graph_change(self, y, parents, x):
        """Evaluate change in score from adding x->y"""
        assert x is not y
        assert y not in parents
        y_index = y

        parent_indices = list()
        for parent_node in parents:
            parent_indices.append(parent_node)

        return self.local_score_diff_parents(x, y_index, parent_indices)

    def local_score_diff_parents(self, x, y, parents):
        bump_without_x = self.local_score(
            self.data, self.graph, local=True, parents=parents, node=y
        )
        parents.append(x)
        bump_with_x = self.local_score(self.data, self.graph, local=True, parents=parents, node=y)
        return bump_with_x - bump_without_x

    def calculate_arrows_forward(self, a, b):
        # print("Calculate Arrows Forward: " + str(a) + " " + str(b))
        if b not in self.effect_edges_graph[a] and self.mode == "heuristic":
            print("Returning early...")
            return

        if self.knowledge is not None and self.knowledge.is_forbidden(a, b):
            return

        # print("Get neighbors for " + str(b) + " returns " + str(graph_util.neighbors(self.graph, b)))

        self.stored_neighbors[b] = graph_util.neighbors(self.graph, b)

        na_y_x = graph_util.get_na_y_x(self.graph, a, b)
        _na_y_x = list(na_y_x)

        if not graph_util.is_clique(self.graph, na_y_x):
            return

        t_neighbors = list(graph_util.get_t_neighbors(self.graph, a, b))
        # print("tneighbors for " + str(a) + ", " + str(b) + " returns " + str(t_neighbors))
        len_T = len(t_neighbors)

        def outer_loop():
            previous_cliques = set()  # set of sets of nodes
            previous_cliques.add(frozenset())
            new_cliques = set()  # set of sets of nodes
            for i in range(len_T + 1):
                choices = itertools.combinations(range(len_T), i)
                choices2 = itertools.combinations(range(len_T), i)
                # print("All choices: ", list(choices2), " TNeighbors: ", t_neighbors)
                for choice in choices:
                    T = frozenset([t_neighbors[k] for k in choice])
                    # print("Choice:", T)
                    union = set(na_y_x)
                    union.update(T)

                    found_a_previous_clique = False

                    for clique in previous_cliques:
                        # basically if clique is a subset of union
                        if union >= clique:
                            found_a_previous_clique = True
                            break

                    if not found_a_previous_clique:
                        # Break out of the outer for loop
                        return

                    if not graph_util.is_clique(self.graph, union):
                        continue

                    new_cliques.add(frozenset(union))

                    bump = self.insert_eval(a, b, T, na_y_x)
                    logger.info(
                        "Evaluated arrow "
                        + str(a)
                        + " -> "
                        + str(b)
                        + " with T: "
                        + str(T)
                        + " and bump: "
                        + str(bump)
                    )

                    if bump > 0:
                        self.add_arrow(a, b, na_y_x, T, bump)

                previous_cliques = new_cliques
                new_cliques = set()

        outer_loop()

    def valid_set_by_knowledge(self, x, subset):
        """Use knowledge to decide if an insert of delete does not orient edges
        in a forbidden way. If some orientation in the subset is forbidden,
        the whole subset is forbidden"""
        for node in subset:
            if self.knowledge.is_forbidden(x, node):
                return False
        return True

    def create_checkpoint(self):
        with open(self.save_name + "-checkpoint.pkl", "wb") as f:
            dill.dump(self, f, dill.HIGHEST_PROTOCOL)
