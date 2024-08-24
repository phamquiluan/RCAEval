import itertools

from libraries.FGES import graph_util


class MeekRules:
    __slots__ = [
        "undirect_unforced_edges",
        "init_nodes",
        "node_subset",
        "visited",
        "direct_stack",
        "oriented",
        "knowledge",
    ]

    def __init__(self, undirect_unforced_edges=True, knowledge=None):
        # Unforced parents should be undirected before orienting
        self.undirect_unforced_edges = undirect_unforced_edges
        self.init_nodes = []
        self.node_subset = {}
        self.visited = set()
        self.direct_stack = []
        self.oriented = set()
        self.knowledge = knowledge

    def orient_implied_subset(self, graph, node_subset):
        self.node_subset = node_subset
        self.visited.update(node_subset)
        self.orient_using_meek_rules_locally(graph)

    def orient_implied(self, graph):
        self.orient_implied_subset(graph, graph.nodes())

    def orient_using_meek_rules_locally(self, graph):
        """Orient graph using the four Meek rules"""
        if self.undirect_unforced_edges:
            for node in self.node_subset:
                self.undirect_unforced_edges_func(node, graph)
                self.direct_stack.extend(graph_util.adjacent_nodes(graph, node))

        # TODO: Combine loops
        for node in self.node_subset:
            self.run_meek_rules(node, graph)
        if self.direct_stack != []:
            last_node = self.direct_stack.pop()
        else:
            last_node = None

        while last_node is not None:
            # print(last_node)
            if self.undirect_unforced_edges:
                self.undirect_unforced_edges_func(last_node, graph)

            self.run_meek_rules(last_node, graph)
            # print("past run_meek_rules")
            if len(self.direct_stack) > 0:
                last_node = self.direct_stack.pop()
            else:
                last_node = None

    def undirect_unforced_edges_func(self, node, graph):
        """Removes directed edges that are not forced by an unshielded collider about node"""
        node_parents = graph_util.get_parents(graph, node)
        parents_to_undirect = set(node_parents)

        # Find any unshielded colliders in node_parents, and orient them
        for p1, p2 in itertools.combinations(node_parents, 2):
            if not graph_util.adjacent(graph, p1, p2):
                # Have an unshielded collider p1 -> node <- p2, which forces orientation
                self.oriented.update([(p1, node), (p2, node)])
                parents_to_undirect.difference_update([p1, p2])

        did_unorient = False

        for parent in parents_to_undirect:
            if self.knowledge is not None:
                must_orient = self.knowledge.is_required(
                    parent, node
                ) or self.knowledge.is_forbidden(node, parent)
            else:
                must_orient = False
            if not (parent, node) in self.oriented and not must_orient:
                # Undirect parent -> node
                graph_util.remove_dir_edge(graph, parent, node)
                graph_util.add_undir_edge(graph, parent, node)
                self.visited.add(node)
                self.visited.add(parent)
                # print(f"unorienting {parent} -> {node}")
                did_unorient = True

        if did_unorient:
            for adjacent in graph_util.adjacent_nodes(graph, node):
                self.direct_stack.append(adjacent)

            self.direct_stack.append(node)

    def run_meek_rules(self, node, graph):
        self.run_meek_rule_one(node, graph)
        self.run_meek_rule_two(node, graph)
        self.run_meek_rule_three(node, graph)
        self.run_meek_rule_four(node, graph)

    def run_meek_rule_one(self, node, graph):
        """
        Meek's rule R1: if a-->b, b---c, and a not adj to c, then a-->c
        """
        # print("Running meek rule one", node)
        adjacencies = graph_util.adjacent_nodes(graph, node)
        if len(adjacencies) < 2:
            return
        all_combinations = itertools.combinations(range(0, len(adjacencies)), 2)
        for index_one, index_two in all_combinations:
            node_a = adjacencies[index_one]
            node_c = adjacencies[index_two]

            # TODO: Parallelize these flipped versions?
            self.r1_helper(node_a, node, node_c, graph)
            self.r1_helper(node_c, node, node_a, graph)

    def r1_helper(self, node_a, node_b, node_c, graph):
        if (
            (not graph_util.adjacent(graph, node_a, node_c))
            and (
                graph_util.has_dir_edge(graph, node_a, node_b) or (node_a, node_b) in self.oriented
            )
            and graph_util.has_undir_edge(graph, node_b, node_c)
        ):
            if not graph_util.is_unshielded_non_collider(graph, node_a, node_b, node_c):
                return

            if self.is_arrowpoint_allowed(node_b, node_c):
                # print("R1: " + str(node_b) + " " + str(node_c))
                if (
                    (node_a, node_c) not in self.oriented
                    and (node_c, node_a) not in self.oriented
                    and (node_b, node_c) not in self.oriented
                    and (node_c, node_b) not in self.oriented
                ):
                    self.direct(node_b, node_c, graph)

    def direct(self, node_1, node_2, graph):
        # print("Int Directing " + str(node_1) + " " + str(node_2))
        if self.knowledge is not None:
            if self.knowledge.is_forbidden(node_1, node_2):
                return

        graph_util.remove_dir_edge(graph, node_1, node_2)
        graph_util.remove_dir_edge(graph, node_2, node_1)
        graph_util.add_dir_edge(graph, node_1, node_2)
        self.visited.update([node_1, node_2])
        # node_1 -> node_2 edge
        if (node_1, node_2) not in self.oriented and (node_2, node_1) not in self.oriented:
            self.oriented.add((node_1, node_2))
            self.direct_stack.append(node_2)

    def run_meek_rule_two(self, node_b, graph):
        # print("Running meek rule two", node_b)
        adjacencies = graph_util.adjacent_nodes(graph, node_b)
        if len(adjacencies) < 2:
            return
        all_combinations = itertools.combinations(range(0, len(adjacencies)), 2)
        for index_one, index_two in all_combinations:
            node_a = adjacencies[index_one]
            node_c = adjacencies[index_two]
            self.r2_helper(node_a, node_b, node_c, graph)
            self.r2_helper(node_b, node_a, node_c, graph)
            self.r2_helper(node_a, node_c, node_b, graph)
            self.r2_helper(node_c, node_a, node_b, graph)

    def r2_helper(self, a, b, c, graph):
        if (
            graph_util.has_dir_edge(graph, a, b)
            and graph_util.has_dir_edge(graph, b, c)
            and graph_util.has_undir_edge(graph, a, c)
        ):
            if self.is_arrowpoint_allowed(a, c):
                self.direct(a, c, graph)

    def run_meek_rule_three(self, node, graph):
        """
        A --- B, A --- X, A --- C, B --> X <-- C, B -/- C => A --> X
        The parameter node = X
        """
        # print("Running meek rule three", node)
        adjacencies = graph_util.adjacent_nodes(graph, node)

        if len(adjacencies) < 3:
            return

        for node_a in adjacencies:
            if graph_util.has_undir_edge(graph, node, node_a):
                copy_adjacencies = [a for a in adjacencies if a != node_a]
                all_combinations = itertools.combinations(copy_adjacencies, 2)

                for node_b, node_c in all_combinations:
                    if (
                        graph_util.is_kite(graph, node, node_a, node_b, node_c)
                        and self.is_arrowpoint_allowed(node_a, node)
                        and graph_util.is_unshielded_non_collider(graph, node_c, node_a, node_b)
                    ):
                        # print("R3: " + str(node_a) + " " + str(node))
                        self.direct(node_a, node, graph)

    #     def run_meek_rule_four(self, a, graph):
    #         if self.knowledge is None:
    #             return

    #         adjacent_nodes = graph_util.adjacent_nodes(graph, a)

    #         for (b, c, d) in itertools.permutations(adjacent_nodes, 3):
    #             if (graph_util.has_undir_edge(graph, a, b) and
    #                     graph_util.has_dir_edge(graph, b, c) and
    #                     graph_util.has_dir_edge(graph, c, d) and
    #                     graph_util.has_undir_edge(graph, d, a) and
    #                     self.is_arrowpoint_allowed(a, d)):
    #                 self.direct(a, d, graph)

    def run_meek_rule_four(self, a, graph):
        adjacent_nodes = graph_util.adjacent_nodes(graph, a)
        if len(adjacent_nodes) < 3:
            return
        neighbours_a = graph_util.neighbors(graph, a)
        for c in adjacent_nodes:
            list_b = graph_util.get_parents(graph, c)
            list_d = graph_util.get_children(graph, c)
            list_ba = [x for x in list_b if x in neighbours_a]
            list_da = [x for x in list_d if x in neighbours_a]
            dir_list = []
            for d in list_da:
                flag = 0
                for b in list_ba:
                    if not graph_util.adjacent(graph, b, d):
                        flag = 1
                        break
                if flag == 1:
                    dir_list.append(d)
            for d in dir_list:
                if self.is_arrowpoint_allowed(a, d):
                    self.direct(a, d, graph)

    def is_arrowpoint_allowed(self, from_node, to_node):
        if self.knowledge is None:
            return True

        return not self.knowledge.is_required(
            to_node, from_node
        ) and not self.knowledge.is_forbidden(from_node, to_node)

    def get_visited(self):
        """This is what FGES actually uses"""
        return self.visited
