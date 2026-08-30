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
            evaluator.add_case(ranks=ranks, answer=answer)
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
        evaluator.add_case(ranks=ranks, answer=answer)

    assert evaluator.average(5) == pytest.approx(1.0)
    assert evaluator.chance_average(5) == pytest.approx(3 / N_NODES)
    assert evaluator.lift(5) == pytest.approx(1.0 - evaluator.chance_average(5))
    assert evaluator.n_candidates == [N_NODES] * N_CASES


def test_constant_ranker_scores_at_or_below_answer_frequency():
    nodes = _nodes()
    constant = nodes[0]
    rng = random.Random(1)
    answers = _answers(rng)
    frequency = sum(a == constant for a in answers) / len(answers)

    evaluator = Evaluator()
    for answer in answers:
        evaluator.add_case(ranks=[constant], answer=answer)

    assert evaluator.accuracy(1) == pytest.approx(frequency)
    assert evaluator.average(5) <= frequency + 1e-9
    # A one-item ranking is at chance by construction, so lift is zero when the
    # constant node is the answer every time and negative otherwise.
    assert evaluator.chance_average(5) == pytest.approx(1.0)
    assert evaluator.lift(5) <= 1e-9
