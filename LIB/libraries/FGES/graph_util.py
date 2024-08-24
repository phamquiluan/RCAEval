"""
Graph Terminology
-----------------
 * DAG: Directed Acyclic Graph
 * PDAG: Partially Directed Acyclic Graph
 * Directed edge: x --> y
 * Undirected edge: x --- y is represented as the pair x --> y and y --> x
 * Parents(x): nodes y such that there is a directed edge y --> x
 * Neighbors(x): nodes y such that there is an undirected edge y --- x
 * Adjacents(x): nodes y such there is some edge between x and y
   (i.e. x --- y, x --> y, or x <-- y)
 * Directed path: A path of entirely directed edges in the same direction
 * Ancestors(x): nodes y such that there is a directed path y ~~> x
 * Unshielded collider: triple of vertices x, y, z with directed edges
   x --> y <-- z, and x and z are not adjacent
 * Semi-directed path: A path of directed and undirected edges, where no
   directed edge is going in the opposite direction
"""

import itertools
import queue

import networkx as nx


def add_undir_edge(g, x, y):
    """Adds an undirected edge (x,y) to Graph g.
    Undirected edges are stores as pairs of directed edges (x,y) and (y,x)"""
    g.add_edge(x, y)
    g.add_edge(y, x)


def add_dir_edge(g, x, y):
    """Adds an directed edge (x,y) to Graph g."""
    g.add_edge(x, y)


def undir_to_dir(g, x, y):
    """Keep only x --> y"""
    if g.has_edge(y, x) and g.has_edge(x, y):
        # Current edge is x --- y
        g.remove_edge(y, x)
        # print("Directing " + str(x) + " -> " + str(y))
    elif g.has_edge(x, y):
        # Current edge is x --> y
        pass
        # print("No-op Directing " + str(x) + " -> " + str(y))
    elif g.has_edge(y, x):
        # Current edge is y --> x
        raise AssertionError("undir_to_dir: trying to reverse a directed edge")
    else:
        raise AssertionError("undir_to_dir: no such edge")


def get_parents(g, x):
    """Returns immediate parents of node x in graph g"""
    parents = []
    for node in adjacent_nodes(g, x):
        if has_dir_edge(g, node, x):
            parents.append(node)
    return parents


def get_children(g, x):
    """Returns immediate children of node x in graph g"""
    children = []
    for node in adjacent_nodes(g, x):
        if has_dir_edge(g, x, node):
            children.append(node)
    return children


def get_ancestors(g, x, accum=None):
    """Returns all ancestors of a node x in graph g"""
    if accum is None:
        accum = set()
    if x in accum:
        return accum

    accum.add(x)
    parents = get_parents(g, x)
    for p in parents:
        accum = get_ancestors(g, p, accum)

    return accum


def has_undir_edge(g, x, y):
    """Returns whether there is an undirected edge from x to y in Graph g"""
    return g.has_edge(x, y) and g.has_edge(y, x)


def has_dir_edge(g, x, y):
    """Returns whether there is a directed edge from x to y in Graph g"""
    return g.has_edge(x, y) and not g.has_edge(y, x)


def is_unshielded_non_collider(g, node_a, node_b, node_c):
    """Returns whether nodes a,b,c form an unshielded non-coliding triple"""
    if not adjacent(g, node_a, node_b):
        return False

    if not adjacent(g, node_c, node_b):
        return False

    if adjacent(g, node_a, node_c):
        return False

    if is_ambiguous_triple(g, node_a, node_b, node_c):
        return False

    return not (has_dir_edge(g, node_a, node_b) and has_dir_edge(g, node_c, node_b))


def is_def_collider(g, node_1, node_2, node_3):
    """Returns whether nodes a,b,c form a collider a -> b <- c"""
    return has_dir_edge(g, node_1, node_2) and has_dir_edge(g, node_3, node_2)


def is_unshielded_collider(g, node_1, node_2, node_3):
    """Returns whether nodes a,b,c form an unshielded collider a -> b <- c"""
    return (
        has_dir_edge(g, node_1, node_2)
        and has_dir_edge(g, node_3, node_2)
        and not adjacent(g, node_1, node_3)
    )


def check_for_colliders(g, n):
    """Searches for an unshielded collider a -> n <- b"""
    adj = [m for m in g.nodes() if has_dir_edge(g, m, n)]
    new_colliders = set()

    for pair in itertools.combinations(adj, 2):
        if is_unshielded_collider(g, pair[0], n, pair[1]):
            new_colliders.add((pair[0], n, pair[1]))

    return new_colliders


def get_all_collider_triples(g):
    """Returns set of all collider triples in a Graph g"""
    colliders = [check_for_colliders(g, n) for n in g.nodes()]
    return set.union(*colliders)


def is_ambiguous_triple(g, node_a, node_b, node_c):
    # TODO: Actually write this. I'm having a tough time finding
    # where this logic is implemented, because it seems like
    # ambiguous triples is not populated for an EdgeListGraph that is
    # populated with just a List<Node>s
    return False


def traverseSemiDirected(g, x, y):
    """Returns y if there is a directed"""
    if has_undir_edge(g, x, y) or g.has_edge(x, y):
        return y
    return None


def adjacent(g, x, y):
    """Returns whether nodes x and y are adjacent"""
    return g.has_edge(x, y) or g.has_edge(y, x)


def undir_edge_neighbors(g, x, y):
    """Alias for has_undir_edge"""
    return has_undir_edge(g, x, y)


def adjacent_nodes(g, x):
    """Returns all adjacent nodes to x in Graph g"""
    return list(set(nx.all_neighbors(g, x)))


def neighbors(g, x):
    """Returns all neighbors of x in Graph g.
    A neighbor is an adjacent node that"""
    potentialNeighbors = nx.all_neighbors(g, x)
    resulting_neighbors = set({})
    for pNode in potentialNeighbors:
        if undir_edge_neighbors(g, x, pNode):
            resulting_neighbors.add(pNode)

    return resulting_neighbors


def get_na_y_x(g, x, y):
    """Gets possible conditioning sets of nodes to consider for P(Y|X)"""
    na_y_x = []
    all_y_neighbors = set(nx.all_neighbors(g, y))

    for z in all_y_neighbors:
        if has_undir_edge(g, z, y):
            if adjacent(g, z, x):
                na_y_x.append(z)

    return set(na_y_x)


def is_clique(g, node_set):
    """Checks to see if the nodes in node_set form a clique"""
    for node in node_set:
        for other_node in node_set:
            if node != other_node and not adjacent(g, node, other_node):
                return False
    return True


def get_t_neighbors(g, x, y):
    """Returns possible members of conditioning set for x---y edge consideration"""
    t = set([])
    all_y_neighbors = set(nx.all_neighbors(g, y))

    for z in all_y_neighbors:
        if has_undir_edge(g, z, y):
            if adjacent(g, z, x):
                continue
            t.add(z)

    return t


def is_kite(g, a, d, b, c):
    """
    Returns if nodes a,b,c, and d form a kite
    D --- B, D --- A, D --- C, B --> A <-- C
    See Meek (1995) Rule 3
    """
    return (
        has_undir_edge(g, d, a)
        and has_undir_edge(g, d, b)
        and has_undir_edge(g, d, c)
        and has_dir_edge(g, b, a)
        and has_dir_edge(g, c, a)
    )


def get_common_adjacents(g, x, y):
    """Get the nodes that are adjacent to both x and y"""
    return set(adjacent_nodes(g, x)).intersection(set(adjacent_nodes(g, y)))


def remove_dir_edge(g, x, y):
    """Removes the directed edge x --> y"""
    if g.has_edge(x, y):
        g.remove_edge(x, y)


def exists_unblocked_semi_directed_path(g, origin, dest, cond_set, bound):
    """Checks if there exists a unblocked semi directed path (that is, there could be a possible path) from
    origin to dest, while conditioning on cond_set"""
    if bound == -1:
        bound = 1000

    q = queue.Queue()
    v = set()
    q.put(origin)
    v.add(origin)

    e = None
    distance = 0

    while not q.empty():
        t = q.get()
        if t == dest:
            return True

        if e == t:
            e = None
            distance += 1
            if distance > bound:
                return False

        for u in set(nx.all_neighbors(g, t)):
            c = traverseSemiDirected(g, t, u)
            if c is None:
                continue

            if c in cond_set:
                continue

            if c == dest:
                return True

            if not c in v:
                v.add(c)
                q.put(c)

                if e == None:
                    e = c
    return False


def detect_cycle_at_node(graph, node):
    """Detect a cycle involving a specific node"""
    q = queue.Queue()
    visited = set()
    q.put(node)

    while not q.empty():
        t = q.get()
        for child in graph.neighbors(t):
            if has_dir_edge(graph, t, child):
                if child == node:
                    return True
                elif child not in visited:
                    q.put(child)
        visited.add(t)

    return False


def detect_cycle(graph):
    """Detect a cycle by finding directed loop."""
    for n in graph.nodes():
        if detect_cycle_at_node(graph, n):
            return True
    return False
