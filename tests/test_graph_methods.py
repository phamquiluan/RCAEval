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


def _six_node_graphs():
    names = [f"svc{i}_cpu" for i in range(6)]
    chain = np.zeros((6, 6), int)  # svc0 -> svc1 -> ... -> svc5, causal-learn marks
    star = np.zeros((6, 6), int)  # svc5 -> every other node
    chain01 = np.zeros((6, 6), int)  # the chain as granger() and LiNGAM write it
    for a in range(5):
        chain[a, a + 1], chain[a + 1, a] = -1, 1
        star[5, a], star[a, 5] = -1, 1
        chain01[a + 1, a] = 1
    return names, chain, star, chain01


def test_random_walk_ranking_follows_the_graph():
    from RCAEval.graph_heads.random_walk import random_walk

    names, chain, star, chain01 = _six_node_graphs()

    def rank(adj):
        return [name for name, _ in random_walk(adj, names, num_loop=600)]

    # every node used to start with score 0, which zeroed all transition
    # weights: the walk was uniform and any graph gave the same ranking (#96)
    assert rank(chain) != rank(star)
    # the walk climbs from effects to causes, so it gathers at the root; it
    # used to run on the reversed graph, which sends it to the last effect
    assert rank(chain)[0] == "svc0_cpu"
    assert rank(star)[0] == "svc5_cpu"
    assert rank(chain01)[0] == "svc0_cpu"


def test_random_walk_does_not_read_the_global_rng():
    from RCAEval.graph_heads.random_walk import random_walk

    names, chain, _, _ = _six_node_graphs()
    # the start node used to come from np.random, which nothing seeds; it now
    # matters, so the ranking would change from run to run
    np.random.seed(1)
    first = random_walk(chain, names, num_loop=6)
    np.random.seed(2)
    assert random_walk(chain, names, num_loop=6) == first


def test_random_walk_leaves_a_node_without_neighbours():
    from RCAEval.classes.graph import MemoryGraph, Node
    from RCAEval.graph_heads.random_walk import RandomWalkScorer, Score

    a, b, c = (Node(name, None) for name in "abc")
    graph = nx.DiGraph([(a, b)])
    graph.add_node(c)
    scores = {node: Score(1.0) for node in (a, b, c)}

    matrix = RandomWalkScorer().generate_transition_matrix(MemoryGraph(graph), data=None, scores=scores)
    # c's own score was its only weight, so a walk that started there never left
    assert np.allclose(matrix[c], 1 / 3)
    assert np.allclose(matrix.sum(), 1)


def test_randomwalk_methods_rank_by_the_graph(monkeypatch, capsys):
    pytest.importorskip("causallearn")
    pytest.importorskip("statsmodels")
    m = importlib.import_module("RCAEval.e2e.pc_randomwalk")

    names, chain, star, _ = _six_node_graphs()
    data = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 6)), columns=names)

    def ranks(adj):
        monkeypatch.setattr(m, "pc_default", lambda data: adj)
        return m.pc_randomwalk(data, n_iter=600)["ranks"]

    # the same metrics on two different graphs used to get the same ranking (#96)
    assert ranks(chain) != ranks(star)
    assert ranks(chain)[0] == "svc0_cpu"
    assert "failed" not in capsys.readouterr().err


def test_ntlr_randomwalk_reads_notears_marks(monkeypatch, capsys):
    import sys
    import types

    pytest.importorskip("causallearn")
    pytest.importorskip("statsmodels")
    m = importlib.import_module("RCAEval.e2e.pc_randomwalk")

    names, _, _, chain01 = _six_node_graphs()
    # NOTEARS writes svc0 -> svc1 as adj[0, 1], the transpose of granger()'s
    # matrix; stand in for the gcastle-backed builder, which is an optional extra
    fake = types.ModuleType("RCAEval.graph_construction.dag_gnn")
    fake.notears_low_rank = lambda data: chain01.T
    monkeypatch.setitem(sys.modules, "RCAEval.graph_construction.dag_gnn", fake)

    data = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 6)), columns=names)
    out = m.ntlr_randomwalk(data, n_iter=600)

    assert out["ranks"][0] == "svc0_cpu"
    assert "failed" not in capsys.readouterr().err


def _two_causes_of_the_sli():
    # a -> sli <- b in causal-learn marks; c has no edge
    adj = np.zeros((4, 4), int)
    adj[1, 0] = adj[2, 0] = -1
    adj[0, 1] = adj[0, 2] = 1
    return adj


def test_random_walk_starts_at_the_sli_and_follows_scores():
    from RCAEval.graph_heads.random_walk import random_walk

    names = ["sli", "a", "b", "c"]
    scores = {"sli": 1.0, "a": 0.9, "b": 0.1, "c": 0.5}
    visits = dict(random_walk(_two_causes_of_the_sli(), names, sli="sli", num_loop=400, previous_scores=scores))

    # of the SLI's two causes, the walk prefers the one with the higher score
    assert visits["a"] > visits["b"] > 0
    # it starts at the SLI, so it never reaches the unconnected node; the sli
    # argument used to be overwritten with a random node
    assert visits["c"] == 0

    with pytest.raises(ValueError):
        random_walk(_two_causes_of_the_sli(), names, sli="frontend_latency", num_loop=10)


def test_sli_correlation_scores():
    from RCAEval.graph_heads.random_walk import sli_correlation

    t = np.arange(50.0)
    data = pd.DataFrame({"sli": t, "inverse": -t, "flat": np.ones(50), "wave": np.sin(t)})
    scores = sli_correlation(data, "sli")

    assert scores["sli"] == pytest.approx(1) and scores["inverse"] == pytest.approx(1)
    # Pearson's r is undefined for a constant column
    assert scores["flat"] == 0
    assert 0 <= scores["wave"] < 0.5


def test_randomwalk_methods_use_the_sli_when_given(monkeypatch, capsys):
    pytest.importorskip("causallearn")
    pytest.importorskip("statsmodels")
    m = importlib.import_module("RCAEval.e2e.pc_randomwalk")

    rng = np.random.default_rng(0)
    sli = rng.normal(size=200)
    data = pd.DataFrame({
        "front_latency": sli,
        "a_cpu": sli + 0.1 * rng.normal(size=200),
        "b_cpu": rng.normal(size=200),
        "c_cpu": rng.normal(size=200),
    })
    monkeypatch.setattr(m, "pc_default", lambda data: _two_causes_of_the_sli())

    ranks = m.pc_randomwalk(data, n_iter=400, sli="front_latency")["ranks"]
    # both are causes of the SLI in the graph; only a_cpu moves with it
    assert ranks.index("a_cpu") < ranks.index("b_cpu")
    assert "failed" not in capsys.readouterr().err

    # an SLI the data does not have is reported, not replaced by another start
    m.pc_randomwalk(data, n_iter=400, sli="frontend_latency")
    assert "frontend_latency" in capsys.readouterr().err


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


def test_run_pearson_correlation_handles_large_int_metrics():
    pytest.importorskip("torch")
    from RCAEval.e2e.run import pearson_correlation

    # memory metrics are int64 near 1e8; summing their squares overflowed
    rng = np.random.default_rng(0)
    x = pd.Series(rng.integers(5 * 10**7, 10**8, size=4000))
    y = 3 * x + pd.Series(rng.integers(0, 10**7, size=4000))
    assert np.isclose(pearson_correlation(x, y), np.corrcoef(x, y)[0, 1])
    assert pearson_correlation(x, pd.Series(np.zeros(4000))) == 0


@pytest.mark.parametrize("sli_col", [0, 2])
def test_cloudranger_walk_starts_at_the_sli(monkeypatch, sli_col):
    pytest.importorskip("causallearn")
    pytest.importorskip("pingouin")
    m = importlib.import_module("RCAEval.e2e.cloudranger")

    # cloudranger passed the 0-based column index where relaToRank and the walk
    # expect a 1-based node number, so the walk started one node before the
    # SLI, or at the last node when the SLI was the first column (issue #89)
    starts, walk = [], m.secondorder_randomwalk

    def recording_walk(M, epochs, start_node, label=[], **kw):
        starts.append(start_node)
        return walk(M, 1, start_node, label, walk_step=20)

    monkeypatch.setattr(m, "secondorder_randomwalk", recording_walk)
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    data = pd.DataFrame({
        "time": np.arange(200), "a_latency": a, "b_cpu": a + 0.1 * rng.normal(size=200),
        "c_mem": rng.normal(size=200), "d_cpu": rng.normal(size=200),
    })
    sli = ["a_latency", "b_cpu", "c_mem", "d_cpu"][sli_col]
    out = m.cloudranger(data, dataset="re2-ob", sli=sli)
    assert out["node_names"][starts[0] - 1] == sli
