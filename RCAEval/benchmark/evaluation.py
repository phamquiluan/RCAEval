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

    def add_case(self, ranks: Sequence[Node], answer: Node, n_candidates: int = None,
                 answer_in_candidates: bool = True):
        """
        Record one case. n_candidates is the size of the candidate set the method
        ranked over, as recorded by the caller. A case without it still counts for
        accuracy, but the chance baseline and lift for the whole run become None,
        because guessing a denominator from the ranking length understates the
        floor for any method that truncates its output.

        answer_in_candidates says whether the ground-truth answer is in the
        candidate set. It defaults to True, which holds by construction for
        benchmarks whose answer is an injected member of the ranked system, so a
        method that never places the answer lands below chance rather than at it.
        A caller that knows the answer is absent passes False, and the case then
        contributes a chance of 0, since no ranking over those candidates can
        place it.
        """
        self._ranks.append(ranks[: 5])
        self._n_candidates.append(n_candidates)
        self._answer_in_candidates.append(bool(answer_in_candidates))

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
        Candidate set size recorded for each case, in the order the cases were
        added. None for cases added without one.
        """
        return list(self._n_candidates)

    @property
    def n_missing_candidates(self) -> int:
        """
        Number of cases added without a recorded candidate set size.
        """
        return sum(1 for n in self._n_candidates if n is None)

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

        For a case with n candidates and the answer among them, the chance of the
        answer landing in the top k is min(k, n) / n. A case recorded with
        answer_in_candidates=False counts 0, since no ranking over those candidates
        can place it. The value is the mean over all cases, with an empty candidate
        set counting 0. Returns None when any case has no recorded candidate count,
        so a partially recorded run reports no floor at all instead of a wrong one.
        """
        if k not in self._accuracy or not self._ranks:
            return None
        if any(n is None for n in self._n_candidates):
            return None
        return sum(
            min(k, n) / n if n and found else 0.0
            for n, found in zip(self._n_candidates, self._answer_in_candidates)
        ) / self.num

    def chance_average(self, k: int) -> float:
        """
        Avg@k of a uniformly random ranking, computed the same way as average(k)
        from chance_accuracy. None when any case has no recorded candidate count.
        """
        if k not in self._accuracy or not self._ranks:
            return None
        parts = [self.chance_accuracy(i) for i in range(1, k + 1)]
        if any(p is None for p in parts):
            return None
        return sum(parts) / k

    def lift(self, k: int) -> float:
        """
        average(k) minus chance_average(k). Zero means the method did no better than
        a random ranking. None when the chance baseline is unavailable.
        """
        if k not in self._accuracy or not self._ranks:
            return None
        chance = self.chance_average(k)
        if chance is None:
            return None
        return self.average(k) - chance
