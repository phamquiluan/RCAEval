from typing import List, Set, Sequence
from RCAEval.classes.graph import Node
from RCAEval.utility import dump_json, load_json


class Evaluator:
    """"""

    def __init__(self):
        self._accuracy = {k: 0.0 for k in range(1, 6)}
        self._accuracy_service = {k: 0.0 for k in range(1, 6)}
        self._ranks: List[List[Node]] = []
        self._n_candidates: List[int] = []

    def add_case(self, ranks: Sequence[Node], answer: Node):
        self._ranks.append(ranks[: 5])
        self._n_candidates.append(len(ranks))

        service_ranks = [n.entity for n in ranks]
        service_answer = answer.entity

        for k in range(1, 6):
            # fine-grained accuracy
            self._accuracy[k] += int(answer in ranks[:k])

            # coarse-grained
            self._accuracy_service[k] += int(service_answer in service_ranks[:k])

    @property
    def num(self) -> int:
        """
        Number of cases
        """
        return len(self._ranks)

    @property
    def n_candidates(self) -> List[int]:
        """
        Length of the full ranking submitted for each case, in the order the cases were added.
        """
        return list(self._n_candidates)

    def accuracy(self, k: int) -> float:
        """
        AC@k is the average of accuracy@k among cases

        For each case, accuracy@k = |ranks[:k] \\cap answers| / |answers|
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return self._accuracy[k] / self.num

    def accuracy_service(self, k: int) -> float:
        """
        AC@k is the average of accuracy@k among cases

        For each case, accuracy@k = |ranks[:k] \\cap answers| / |answers|
        """
        if k not in self._accuracy_service or not self._ranks:
            return None
        return self._accuracy_service[k] / self.num

    def average(self, k: int) -> float:
        """
        Avg@k = \\sum_{j=1}^{k} AC@j / k
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return sum(self.accuracy(i) for i in range(1, k + 1)) / k

    def average_service(self, k: int) -> float:
        """
        Avg@k = \\sum_{j=1}^{k} AC@j / k
        """
        if k not in self._accuracy_service or not self._ranks:
            return None
        return sum(self.accuracy_service(i) for i in range(1, k + 1)) / k

    def chance_accuracy(self, k: int) -> float:
        """
        AC@k that a uniformly random ranking over the same candidates would reach.

        For a case with n candidates the chance of the answer landing in the top k is min(k, n) / n.
        The value is the mean of that chance over all cases. A case with no candidates counts as 0.
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return sum(min(k, n) / n if n else 0.0 for n in self._n_candidates) / self.num

    def chance_average(self, k: int) -> float:
        """
        Avg@k of a uniformly random ranking, computed the same way as average(k) from chance_accuracy.
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return sum(self.chance_accuracy(i) for i in range(1, k + 1)) / k

    def lift(self, k: int) -> float:
        """
        average(k) minus chance_average(k). Zero means the method did no better than a random ranking.
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return self.average(k) - self.chance_average(k)
