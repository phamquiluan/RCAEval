from abc import ABC
from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd

from RCAEval.classes.data import CaseData
from RCAEval.classes.graph import Graph, MemoryGraph, Node


def _times(num: int, multiplier: int = 10) -> int:
    return num * multiplier


class Score:
    """
    The more suspicious a node, the higher the score.
    """

    def __init__(self, score: float, info: dict = None, key: tuple = None):
        self._score = score
        self._key = key
        self._info = {} if info is None else info

    def __eq__(self, obj) -> bool:
        if isinstance(obj, Score):
            return self.score == obj.score
        return False

    def __getitem__(self, key: str):
        return self._info[key]

    def __setitem__(self, key: str, value):
        self._info[key] = value

    def get(self, key: str, default=None):
        """
        Return the value for key if key has been set, else default
        """
        return self._info.get(key, default)

    @property
    def score(self) -> float:
        """
        The overall score
        """
        return self._score

    @score.setter
    def score(self, value: float):
        """
        Update score
        """
        self._score = value

    @property
    def key(self) -> float:
        """
        key for sorting
        """
        return self.score if self._key is None else self._key

    @key.setter
    def key(self, value: tuple):
        """
        Update key
        """
        self._key = value

    @property
    def info(self) -> dict:
        """
        Extra information
        """
        return self._info

    def update(self, score: "Score") -> "Score":
        """
        Update score and info
        """
        self._info.update(score.info)
        self.score = score.score
        self.key = score.key
        return self

    def asdict(self) -> Dict[str, Union[float, dict, tuple]]:
        """
        Serialized as a dict
        """
        data = {"score": self._score, "info": {**self._info}}
        if self._key is not None:
            data["key"] = self._key
        return data

    def __repr__(self) -> str:
        return str(self.asdict())


class Scorer(ABC):
    """
    The abstract interface to score nodes
    """

    def __init__(
        self,
        aggregator: Callable[[Sequence[float]], float] = max,
        max_workers: int = 1,
        seed: int = 0,
        cuda: bool = False,
    ):
        self._aggregator = aggregator
        self._max_workers = max_workers
        self._seed = seed
        self._cuda = cuda

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        """
        Estimate suspicious nodes
        """
        raise NotImplementedError


class RandomWalkScorer(Scorer):
    """
    Scorer based on random walk
    """

    def __init__(
        self,
        rho: float = 0.5,
        remove_sli: bool = False,
        num_loop: Union[int, Callable[[int], int]] = None,
        **kwargs,
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)
        self._rho = rho
        self._remove_sli = remove_sli
        self._num_loop = num_loop if num_loop is not None else _times
        self._rng = np.random.default_rng(self._seed)

    def generate_transition_matrix(
        self, graph: Graph, data: CaseData, scores: Dict[Node, Score]
    ) -> pd.DataFrame:
        """
        Generate the transition matrix
        """
        nodes = list(scores.keys())
        size = len(nodes)
        matrix = pd.DataFrame(np.zeros([size, size]), index=nodes, columns=nodes)
        for node in scores:
            neighbours = 0
            for child in graph.children(node):
                if child in scores:
                    matrix.loc[child, node] = self._rho * abs(scores[child].score)
                    neighbours += 1

            parents = graph.parents(node)
            if self._remove_sli:
                parents -= {data.sli}
            for parent in parents:
                if parent in scores:
                    matrix.loc[parent, node] = abs(scores[parent].score)
                    neighbours += 1

            if neighbours == 0:
                # nowhere to step: jump to any node, or a walk that starts
                # here would stay on its own self-weight for good
                matrix[node] = 1 / size
                continue

            matrix.loc[node, node] = max(abs(scores[node].score) - matrix[node].max(), 0)

            total_weight = matrix[node].sum()
            if total_weight > 0:
                matrix[node] = matrix[node] / total_weight
            else:
                matrix[node] = 1 / size
        return matrix

    def _walk(self, start: Node, num_loop: int, matrix: pd.DataFrame) -> Dict[Node, int]:
        node: Node = start
        counter = defaultdict(int)
        for _ in range(num_loop):
            node = self._rng.choice(matrix.index, p=matrix[node])
            counter[node] += 1
        return counter

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        if not scores:
            return scores

        matrix = self.generate_transition_matrix(graph=graph, data=data, scores=scores)
        if isinstance(self._num_loop, int):
            num_loop = self._num_loop
        else:
            num_loop = self._num_loop(len(scores))

        # print("==== matrix ===")
        # print(matrix)
        # print("==== end matrix ===")
        counter = self._walk(start=data.sli, num_loop=num_loop, matrix=matrix)
        # print(f"{num_loop=}")
        # print(dict(counter))

        for node, score in scores.items():
            score["pagerank"] = score.score = counter[node] / num_loop
        return scores


class SecondOrderRandomWalkScorer(RandomWalkScorer):
    """
    Scorer based on second-order random walk
    """

    def __init__(self, beta: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._beta = beta

    def _walk(self, start: Node, num_loop: int, matrix: pd.DataFrame) -> Dict[Node, int]:
        node: Node = start
        node_pre: Node = start
        counter = defaultdict(int)
        for _ in range(num_loop):
            prob_pre = matrix[node_pre][node]
            node_pre = node

            candidates: List[Node] = []
            weights: List[float] = []
            # for key, value in matrix[node].iteritems():
            for key, value in matrix[node].items():
                if value > 0:
                    candidates.append(key)
                    weights.append((1 - self._beta) * prob_pre + self._beta * value)
            total_weight = sum(weights)
            if total_weight == 0:
                node = self._rng.choice(candidates)
            else:
                node = self._rng.choice(candidates, p=[weight / total_weight for weight in weights])
            counter[node] += 1
        return counter


def _pag_marks_to_cpdag(adj):
    """Map FCI's circle marks (2) onto the marks the walk understands.

    adj[x, y] is the mark at x on the edge x-y (-1 tail, 1 arrow, 2 circle):
    o-> becomes -->, o-o becomes <->, and o-- becomes --.
    """
    adj = np.array(adj, copy=True)
    for a in range(len(adj)):
        for b in range(a + 1, len(adj)):
            u, v = adj[a, b], adj[b, a]
            if u == 2 and v == 2:
                adj[a, b] = adj[b, a] = 1
            elif u == 2 and v == 1:
                adj[a, b] = -1
            elif u == 1 and v == 2:
                adj[b, a] = -1
            elif u == 2 or v == 2:
                adj[a, b] = adj[b, a] = -1
    return adj


def _causal_graph(adj: np.ndarray, nodes: List[Node]) -> MemoryGraph:
    """Build the graph the walk runs on, with every edge pointing cause -> effect.

    adj holds either causal-learn's endpoint marks, as PC and FCI return them
    (adj[a, b] = -1 and adj[b, a] = 1 for a -> b, -1/-1 for a -- b, 1/1 for
    a <-> b, 2 for an FCI circle), or 0/1 entries where adj[b, a] = 1 means
    a -> b, as granger() and LiNGAM return them.
    """
    adj = _pag_marks_to_cpdag(adj)
    graph = nx.DiGraph()
    # a node without an edge is still a candidate
    graph.add_nodes_from(nodes)

    node_num = len(adj)
    for a in range(node_num):
        for b in range(node_num):
            # case 1 no edge: a b
            if adj[a, b] == adj[b, a] == 0:
                pass
            # case 2 undirected a -- b: the loop reaches (b, a) too, so the
            # edge ends up in both directions
            elif adj[a, b] == adj[b, a] == -1:
                graph.add_edge(nodes[b], nodes[a])
            # case 3 directed b -> a
            elif adj[a, b] == 1 and adj[b, a] == -1:
                graph.add_edge(nodes[b], nodes[a])
            # case 4 directed a -> b
            elif adj[a, b] == -1 and adj[b, a] == 1:
                graph.add_edge(nodes[a], nodes[b])
            elif adj[a, b] == 0 and adj[b, a] == 1:
                graph.add_edge(nodes[a], nodes[b])
            elif adj[a, b] == 1 and adj[b, a] == 0:
                graph.add_edge(nodes[b], nodes[a])
            elif adj[a, b] == 1 and adj[b, a] == 1:
                graph.add_edge(nodes[a], nodes[b])
                graph.add_edge(nodes[b], nodes[a])
            else:
                raise ValueError(f"Unexpected value: {adj[a, b]}, {adj[b, a]}")
    return MemoryGraph(graph)


def _rank_by_walk(scorer_cls, adj, node_names, num_loop, previous_scores, seed):
    if node_names is None:
        node_names = [f"X{i}" for i in range(len(adj))]
    nodes = [Node(name, None) for name in node_names]
    graph = _causal_graph(adj, nodes)

    if previous_scores is None:
        # no score to prefer one neighbour over another: weigh them equally,
        # so the graph alone shapes the walk. A score of 0 would zero every
        # transition weight and leave a walk that ignores the edges (#96)
        scores = {node: Score(1.0) for node in nodes}
    else:
        # transfer previous_scores (maybe pearsonr)
        scores = {node: Score(previous_scores[node.entity]) for node in nodes}

    rw = scorer_cls(num_loop=num_loop, seed=seed)
    # the walk has to start somewhere; drawn from the seeded generator so
    # that the ranking is reproducible
    start = nodes[rw._rng.integers(len(nodes))]
    data = CaseData(data_loader=None, sli=start, detect_time=0)

    scores = rw.score(graph, data, current=0, scores=scores)
    scores = sorted(scores.items(), key=lambda item: item[1].key, reverse=True)
    return [(node.entity, score.score) for node, score in scores]


def random_walk(
    adj: np.ndarray,
    node_names: List[str] = None,
    sli: Node = None,
    num_loop=None,
    previous_scores=None,
    seed: int = 0,
):
    """Rank the nodes of a causal graph by how often a random walk visits them.

    At each step the walk moves to a cause of the current node (weight: the
    cause's score), back to an effect (weight: rho times the effect's score) or
    stays, so it drifts up the graph towards root causes. A node without
    neighbours jumps to any node.

    adj: the graph, in either mark convention `_causal_graph` describes
    node_names: one name per row of adj
    num_loop: number of steps (default: 10 per node)
    previous_scores: {name: score}, how suspicious each node is; all equal
        when omitted, so that only the graph shapes the walk
    seed: seeds the start node and the steps

    Returns [(name, visit frequency)], most visited first.
    """
    return _rank_by_walk(RandomWalkScorer, adj, node_names, num_loop, previous_scores, seed)


def second_order_random_walk(
    adj: np.ndarray,
    node_names: List[str] = None,
    sli: Node = None,
    num_loop=None,
    previous_scores=None,
    seed: int = 0,
):
    """`random_walk` with a second-order walk: each step also depends on the
    transition probability of the previous one."""
    return _rank_by_walk(SecondOrderRandomWalkScorer, adj, node_names, num_loop, previous_scores, seed)
