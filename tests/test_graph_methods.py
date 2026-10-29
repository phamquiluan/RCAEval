"""Regression tests for silent fallbacks in the metric-based methods."""
import importlib
import io

import networkx as nx
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


def test_causalrca_ranks_upstream_cause_first(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sknetwork")
    m = importlib.import_module("RCAEval.e2e.causalrca")

    known = np.zeros((3, 3))
    known[0, 1] = known[1, 2] = 1.0
    init = m.MLPEncoder.__init__

    def frozen(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.adj_A = torch.nn.Parameter(torch.from_numpy(known).double(), requires_grad=False)

    monkeypatch.setattr(m.MLPEncoder, "__init__", frozen)
    monkeypatch.setattr(m.CONFIG, "epochs", 2)
    monkeypatch.setattr(m.CONFIG, "k_max_iter", 1)

    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(100, 3)) + 5, columns=["cause", "mid", "symptom"])
    assert m.causalrca(data)["ranks"] == ["cause", "mid", "symptom"]


def test_importing_methods_leaves_torch_anomaly_mode_off():
    torch = pytest.importorskip("torch")
    importlib.import_module("RCAEval.e2e.run")

    # run.py used to switch anomaly mode on globally, slowing every torch method
    assert not torch.is_anomaly_enabled()


def test_microcause_ranks_without_crashing(monkeypatch):
    pytest.importorskip("tigramite")
    pytest.importorskip("pingouin")
    m = importlib.import_module("RCAEval.e2e.microcause")

    # the fixed 1000 x 1000-step walk is what makes the method slow, not what breaks
    walk = m.randomwalk
    monkeypatch.setattr(m, "randomwalk", lambda Q, num_loop, start, **kw: walk(Q, 20, start, **{**kw, "walk_step": 50}))

    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = np.roll(a, 1) + 0.1 * rng.normal(size=200)
    c = np.roll(b, 1) + 0.1 * rng.normal(size=200)
    data = pd.DataFrame({"a_cpu": a, "b_cpu": b, "c_cpu": c, "d_cpu": rng.normal(size=200)})

    # tigramite 5 removed return_significant_links, pandas 3 refuses
    # float(Series), and networkx 3 removed adj_matrix
    out = m.microcause(data, sli="c_cpu")
    assert sorted(out["ranks"]) == sorted(data.columns)


def test_e_diagnosis_runs_without_pyrca():
    from RCAEval.e2e import e_diagnosis

    rng = np.random.default_rng(0)
    data = pd.DataFrame({
        "time": np.arange(200),
        "a_cpu": rng.normal(size=200),
        "b_cpu": rng.normal(size=200),
        "c_mem": rng.normal(size=200),
    })
    data.loc[100:, "b_cpu"] += np.linspace(0, 20, 100)

    out = e_diagnosis(data, inject_time=100, dataset="online-boutique")
    assert out["node_names"] == ["a_cpu", "b_cpu", "c_mem"]
    assert set(out["ranks"]) <= set(out["node_names"])


def test_random_walk_accepts_fci_circle_marks():
    from RCAEval.graph_heads.random_walk import random_walk

    # a o-> b, b o-o c: FCI marks circles with 2
    adj = np.array([[0, 2, 0], [1, 0, 2], [0, 2, 0]])
    ranks = random_walk(adj, node_names=["a", "b", "c"], num_loop=50)
    assert sorted(name for name, _ in ranks) == ["a", "b", "c"]


def test_random_walk_transition_matrix_follows_scores():
    import warnings

    from RCAEval.classes.graph import MemoryGraph, Node
    from RCAEval.graph_heads.random_walk import RandomWalkScorer, Score

    a, b, c = (Node(name, None) for name in "abc")
    graph = nx.DiGraph([(a, b), (b, c)])
    scores = {a: Score(0.9), b: Score(0.5), c: Score(0.1)}

    # writing through matrix[col][row] is a silent no-op under pandas 3
    # copy-on-write, which left every column uniform
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        matrix = RandomWalkScorer().generate_transition_matrix(MemoryGraph(graph), data=None, scores=scores)
    assert not np.allclose(matrix[b], 1 / 3)
    assert np.allclose(matrix.sum(), 1)


def test_ges_pagerank_returns_ranked_names(capsys):
    pytest.importorskip("causallearn")
    pytest.importorskip("sknetwork")
    from RCAEval.e2e import ges_pagerank

    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    b = a + 0.1 * rng.normal(size=300)
    data = pd.DataFrame({"a_cpu": a, "b_cpu": b, "c_cpu": b + 0.1 * rng.normal(size=300)})

    out = ges_pagerank(data)
    assert "failed" not in capsys.readouterr().err
    # main.py splits each rank on "_"; (name, score) tuples crashed it
    assert sorted(out["ranks"]) == sorted(data.columns)


def test_granger_is_quiet_and_warning_free(capsys):
    import warnings

    pytest.importorskip("statsmodels")
    from RCAEval.graph_construction.granger import granger

    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    data = pd.DataFrame({"a": a, "b": np.roll(a, 1) + 0.1 * rng.normal(size=300), "c": rng.normal(size=300)})

    # statsmodels 0.15 removed verbose=, which made granger_pagerank fall back
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        adj = granger(data)
    assert adj[1, 0] == 1  # a Granger-causes b
    assert capsys.readouterr().out == ""
