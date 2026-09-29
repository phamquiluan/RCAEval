"""Regression tests for silent fallbacks in the metric-based methods."""
import io

import numpy as np
import pandas as pd
import pytest

from RCAEval.io.time_series import drop_time


def test_drop_time_removes_repeated_time_header():
    # some RE1-OB cases repeat the "time" header; pandas reads it as "time.1"
    data = pd.read_csv(io.StringIO("time,a_cpu,time,b_cpu\n1,0.1,1,0.2\n2,0.3,2,0.4\n"))
    assert list(data.columns) == ["time", "a_cpu", "time.1", "b_cpu"]
    assert list(drop_time(data).columns) == ["a_cpu", "b_cpu"]


def test_page_rank_head_scores_every_node():
    pytest.importorskip("sknetwork")
    from RCAEval.graph_heads.page_rank import page_rank

    adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    ranks = page_rank(adj, node_names=["a", "b", "c"])
    assert sorted(name for name, _ in ranks) == ["a", "b", "c"]
    assert all(np.isfinite(score) for _, score in ranks)


def test_pc_pagerank_ranks_every_node_without_falling_back(capsys):
    pytest.importorskip("causallearn")
    pytest.importorskip("sknetwork")
    from RCAEval.e2e import pc_pagerank

    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    b = a + 0.1 * rng.normal(size=500)
    c = b + 0.1 * rng.normal(size=500)
    # an independent column PC leaves isolated in the graph
    data = pd.DataFrame({"a_cpu": a, "b_cpu": b, "c_cpu": c, "d_cpu": rng.normal(size=500)})

    out = pc_pagerank(data)

    # the @rca wrapper reports any exception it swallows
    assert "failed" not in capsys.readouterr().err
    # every node keeps its own score, isolated ones included; dropping d_cpu
    # from the graph used to shift scores onto the wrong names
    assert np.asarray(out["adj"]).shape == (4, 4)
    assert sorted(out["ranks"]) == sorted(data.columns)
