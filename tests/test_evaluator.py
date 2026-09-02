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
