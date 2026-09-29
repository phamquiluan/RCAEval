"""Tests for Evaluator on controlled rankings, including the chance baseline."""
import random

import pytest

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.classes.graph import Node


N_NODES = 12
N_CASES = 50
SEEDS = range(20)


def _nodes():
    return [Node(f"svc-{i}", "unknown") for i in range(N_NODES)]


def _answers(rng):
    nodes = _nodes()
    return [rng.choice(nodes) for _ in range(N_CASES)]


def test_random_ranker_lands_at_chance():
    nodes = _nodes()
    observed = []
    expected = []
    for seed in SEEDS:
        rng = random.Random(seed)
        evaluator = Evaluator()
        for answer in _answers(rng):
            ranks = nodes[:]
            rng.shuffle(ranks)
            evaluator.add_case(ranks=ranks, answer=answer, n_candidates=N_NODES)
        observed.append(evaluator.average(5))
        expected.append(evaluator.chance_average(5))

    mean_observed = sum(observed) / len(observed)
    mean_expected = sum(expected) / len(expected)
    # With 12 candidates, chance Avg@5 is the mean of 1/12 .. 5/12, which is 0.25.
    assert mean_expected == pytest.approx(3 / N_NODES)
    assert abs(mean_observed - mean_expected) < 0.05


def test_perfect_ranker_scores_one_with_lift_of_one_minus_chance():
    nodes = _nodes()
    rng = random.Random(0)
    evaluator = Evaluator()
    for answer in _answers(rng):
        ranks = [answer] + [n for n in nodes if n != answer]
        evaluator.add_case(ranks=ranks, answer=answer, n_candidates=N_NODES)

    assert evaluator.average(5) == pytest.approx(1.0)
    assert evaluator.chance_average(5) == pytest.approx(3 / N_NODES)
    assert evaluator.lift(5) == pytest.approx(1.0 - evaluator.chance_average(5))
    assert evaluator.n_candidates == [N_NODES] * N_CASES


def test_constant_ranker_scores_at_or_below_answer_frequency():
    n_candidates = 10
    nodes = _nodes()[:n_candidates]
    constant = nodes[0]
    rng = random.Random(1)
    answers = [rng.choice(nodes) for _ in range(N_CASES)]
    frequency = sum(a == constant for a in answers) / len(answers)

    evaluator = Evaluator()
    for answer in answers:
        evaluator.add_case(ranks=[constant], answer=answer, n_candidates=n_candidates)

    assert evaluator.accuracy(1) == pytest.approx(frequency)
    assert evaluator.average(5) <= frequency + 1e-9
    # The floor comes from the candidate set of 10, not from the one-item ranking.
    assert evaluator.chance_average(5) == pytest.approx(3 / n_candidates)
    assert evaluator.lift(5) == pytest.approx(evaluator.average(5) - 3 / n_candidates)


def test_short_precise_ranking_keeps_the_candidate_set_floor():
    n_candidates = 30
    nodes = [Node(f"svc-{i}", "unknown") for i in range(n_candidates)]
    rng = random.Random(2)
    evaluator = Evaluator()
    for _ in range(N_CASES):
        answer = rng.choice(nodes)
        others = [n for n in nodes if n != answer]
        rng.shuffle(others)
        # A method that returns only its top three, with the answer first.
        evaluator.add_case(ranks=[answer] + others[:2], answer=answer, n_candidates=n_candidates)

    assert evaluator.average(5) == pytest.approx(1.0)
    # The floor stays at the 30-candidate chance level, not near 1.0 for a 3-item list.
    assert evaluator.chance_average(5) == pytest.approx(3 / n_candidates)
    assert evaluator.lift(5) == pytest.approx(1.0 - 3 / n_candidates)


def test_missing_candidate_count_makes_chance_and_lift_unavailable():
    nodes = _nodes()
    evaluator = Evaluator()
    evaluator.add_case(ranks=nodes[:], answer=nodes[0], n_candidates=N_NODES)
    evaluator.add_case(ranks=nodes[:], answer=nodes[1])

    # Accuracy still works, but one unrecorded count disables the whole floor,
    # rather than silently falling back to the ranking length for that case.
    assert evaluator.average(5) is not None
    assert evaluator.n_missing_candidates == 1
    assert evaluator.chance_average(5) is None
    assert evaluator.lift(5) is None


def test_never_placing_the_answer_lands_below_chance():
    n_candidates = 14
    nodes = [Node(f"svc-{i}", "unknown") for i in range(n_candidates)]
    # The injected answer is a candidate by construction, but its name never
    # matches what the method ranks, so the method scores zero on every case.
    answer = Node("svc-0-misspelled", "unknown")
    evaluator = Evaluator()
    for _ in range(10):
        evaluator.add_case(ranks=nodes[:], answer=answer, n_candidates=n_candidates)

    assert evaluator.average(5) == pytest.approx(0.0)
    assert evaluator.chance_average(5) == pytest.approx(3 / n_candidates)
    # Below chance, not at it. The floor does not vanish just because the
    # ranking covers the whole candidate set without containing the answer.
    assert evaluator.lift(5) == pytest.approx(-3 / n_candidates)


def test_answer_declared_absent_counts_zero_chance():
    n_candidates = 14
    nodes = [Node(f"svc-{i}", "unknown") for i in range(n_candidates)]
    absent = Node("not-deployed", "unknown")
    evaluator = Evaluator()
    evaluator.add_case(ranks=nodes[:], answer=absent, n_candidates=n_candidates,
                       answer_in_candidates=False)

    # When the caller states the answer is outside the candidate set, no ranking
    # over those candidates can place it, so the case contributes zero chance
    # and zero accuracy, and the lift for it is zero.
    assert evaluator.average(5) == pytest.approx(0.0)
    assert evaluator.chance_average(5) == pytest.approx(0.0)
    assert evaluator.lift(5) == pytest.approx(0.0)


def test_default_max_k_keeps_cutoffs_one_to_five():
    evaluator = Evaluator()
    nodes = _nodes()
    evaluator.add_case(ranks=nodes[:], answer=nodes[0], n_candidates=N_NODES)

    assert evaluator.accuracy(5) == pytest.approx(1.0)
    assert evaluator.accuracy(6) is None
    assert evaluator.retrieval(15) is None
    assert evaluator.rerank(1, 15) is None


def test_invalid_max_k_raises():
    with pytest.raises(ValueError):
        Evaluator(max_k=0)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k,budget", [(1, 5), (3, 10), (5, 15), (15, 15)])
def test_accuracy_equals_retrieval_times_rerank(seed, k, budget):
    n_nodes = 30
    nodes = [Node(f"svc-{i}", "unknown") for i in range(n_nodes)]
    rng = random.Random(seed)
    evaluator = Evaluator(max_k=15)
    for _ in range(N_CASES):
        ranks = nodes[:]
        rng.shuffle(ranks)
        evaluator.add_case(ranks=ranks, answer=rng.choice(nodes), n_candidates=n_nodes)

    retrieval = evaluator.retrieval(budget)
    rerank = evaluator.rerank(k, budget)
    assert 0.0 < retrieval
    assert retrieval == evaluator.accuracy(budget)
    assert evaluator.accuracy(k) == pytest.approx(retrieval * rerank)
    assert evaluator.retrieval_service(budget) == evaluator.accuracy_service(budget)
    assert evaluator.accuracy_service(k) == pytest.approx(
        evaluator.retrieval_service(budget) * evaluator.rerank_service(k, budget))


def test_decomposition_matches_readme_example_of_the_paper():
    # README example of DecompRCA: decompose(y_true, y_pred, cutoff=1, n_candidates=15)
    # gives top@1 0.5, Retrieval@15 1.0, Rerank@1 0.5.
    evaluator = Evaluator(max_k=15)
    evaluator.add_case(ranks=[Node("LIT101", "unknown"), Node("P101", "unknown"), Node("MV101", "unknown")],
                       answer=Node("P101", "unknown"))
    evaluator.add_case(ranks=[Node("FIT201", "unknown"), Node("AIT202", "unknown"), Node("P201", "unknown")],
                       answer=Node("FIT201", "unknown"))

    assert evaluator.accuracy(1) == pytest.approx(0.5)
    assert evaluator.retrieval(15) == pytest.approx(1.0)
    assert evaluator.rerank(1, 15) == pytest.approx(0.5)


def test_rerank_is_none_when_nothing_is_retrieved():
    nodes = _nodes()
    evaluator = Evaluator(max_k=10)
    evaluator.add_case(ranks=nodes[1:], answer=nodes[0], n_candidates=N_NODES)

    assert evaluator.retrieval(10) == pytest.approx(0.0)
    assert evaluator.rerank(1, 10) is None


def test_rerank_cutoff_above_budget_raises():
    evaluator = Evaluator(max_k=10)
    with pytest.raises(ValueError):
        evaluator.rerank(5, 3)


def test_retrieval_over_the_whole_ranking():
    # A method that returns only its own candidates, like RCD: the answer is
    # retrieved when it appears anywhere in the returned ranking.
    nodes = _nodes()
    evaluator = Evaluator()
    evaluator.add_case(ranks=nodes[:2], answer=nodes[0])        # first
    evaluator.add_case(ranks=nodes[:8], answer=nodes[7])        # retrieved, 8th
    evaluator.add_case(ranks=nodes[:3], answer=nodes[9])        # never returned

    assert evaluator.retrieval() == pytest.approx(2 / 3)
    assert evaluator.rerank(1) == pytest.approx(1 / 2)
    assert evaluator.accuracy(1) == pytest.approx(evaluator.retrieval() * evaluator.rerank(1))
    assert evaluator.retrieval_service() == pytest.approx(2 / 3)
    assert evaluator.rerank_service(1) == pytest.approx(1 / 2)
