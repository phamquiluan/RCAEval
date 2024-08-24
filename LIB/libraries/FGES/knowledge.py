"""
knowledge.py

Description from Tetrad

Stores information about required and forbidden edges and common causes for
use in algorithm. This information can be set edge by edge or else globally
via temporal tiers. When setting temporal tiers, all edges from later tiers
to earlier tiers are forbidden.

For this class, all variable names are referenced by name only. This is
because the same Knowledge object is intended to plug into different graphs
with MyNodes that possibly have the same names. Thus, if the Knowledge object
forbids the edge X --> Y, then it forbids any edge which connects a MyNode
named "X" to a MyNode named "Y", even if the underlying MyNodes themselves
named "X" and "Y", respectively, are not the same.
"""
from itertools import permutations

from libraries.FGES import graph_util


class Knowledge:
    def __init__(self):
        self.required_edges = set()
        self.forbidden_edges = set()
        self.required_connections = set()
        self.tier_map = {}
        self.forbidden_within_tiers = {}

    def set_forbidden(self, x, y):
        self.forbidden_edges.add((x, y))

    def remove_forbidden(self, x, y):
        self.forbidden_edges -= {(x, y)}

    def set_required(self, x, y):
        self.required_edges.add((x, y))

    def set_required_from_skeleton(self, skeleton):
        for edge in list(skeleton.edges()):
            self.required_edges.add(edge)

    def set_forbidden_from_skeleton(self, skeleton):
        for i, j in permutations(skeleton.nodes(), 2):
            if (i, j) not in list(skeleton.edges()):
                self.forbidden_edges.add((i, j))

    def remove_required(self, x, y):
        self.required_edges -= {(x, y)}

    def set_required_connection(self, x, y):
        self.required_connections.add((x, y))

    def remove_required_connection(self, x, y):
        self.required_connections -= {(x, y)}

    def set_tier(self, node, tier):
        self.tier_map[node] = tier

    def is_forbidden_by_tiers(self, x, y):
        if not x in self.tier_map.keys() or not y in self.tier_map.keys():
            return False
        if self.tier_map[x] == self.tier_map[y]:
            return self.forbidden_within_tiers[x] == self.forbidden_within_tiers[y]
        return self.tier_map[x] > self.tier_map[y]

    def set_tier_forbidden_within(self, tier, forbidden):
        self.forbidden_within_tiers[tier] = forbidden

    def is_forbidden(self, x, y):
        if self.is_forbidden_by_tiers(x, y):
            return True
        else:
            return (x, y) in self.forbidden_edges

    def no_edge_required(self, x, y):
        return not (
            self.is_required(x, y)
            or self.is_required(y, x)
            or (x, y) in self.required_connections
            or (y, x) in self.required_connections
        )

    def is_required(self, x, y):
        return (x, y) in self.required_edges

    def is_violated_by(self, graph):
        for edge in self.required_edges:
            if not graph_util.has_dir_edge(graph, edge[0], edge[1]):
                return True

        for edge in graph.edges:
            if graph_util.has_undir_edge(graph, edge[0], edge[1]):
                continue

            if self.is_forbidden(edge[0], edge[1]):
                return True

        return False
