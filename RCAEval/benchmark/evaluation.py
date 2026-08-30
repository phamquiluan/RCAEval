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
        self._answer_in_candidates: List[bool] = []

    def add_case(self, ranks: Sequence[Node], answer: Node, n_candidates: int = None):
        """
        Record one case. n_candidates is the size of the candidate set the method
        ranked over. It defaults to the length of the submitted ranking, which is
        only correct for methods that rank every candidate.
        """
        self._ranks.append(ranks[: 5])
        n = n_candidates if n_candidates is not None else len(ranks)
        self._n_candidates.append(n)
        self._answer_in_candidates.append(answer in ranks or len(ranks) < n)

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
        Candidate set size recorded for each case, in the order the cases were added.
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
        A case whose answer is outside the candidate set counts 0 instead of k / n, since no
        ranking over those candidates can place it. The answer is treated as outside the set
        when the submitted ranking already covers the whole candidate set and does not contain
        it. The value is the mean over all cases, with an empty candidate set counting 0.
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return sum(
            min(k, n) / n if n and found else 0.0
            for n, found in zip(self._n_candidates, self._answer_in_candidates)
        ) / self.num

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
