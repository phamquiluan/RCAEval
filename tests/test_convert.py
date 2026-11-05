"""Converters from exported telemetry to an RCAEval case (RCAEval.convert)."""
import json

import pandas as pd
import pytest

from RCAEval.convert import prometheus


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class FakePrometheus:
    """Answers query_range with one series per service, value = timestamp."""

    def __init__(self, services=("cart", "front-end"), label="container", status="success"):
        self.services, self.label, self.status, self.calls = services, label, status, []

    def get(self, url, params, timeout):
        self.calls.append(params)
        if self.status != "success":
            return FakeResponse({"status": "error", "error": "bad query"})
        start, end, step = params["start"], params["end"], params["step"]
        result = [
            {"metric": {self.label: s}, "values": [[t, str(t % 7)] for t in range(start, end + 1, step)]}
            for s in self.services
        ]
        return FakeResponse({"status": "success", "data": {"resultType": "matrix", "result": result}})


def test_pull_metrics_names_columns_service_then_metric():
    queries = {"cpu": {"query": "q1", "label": None}, "mem": {"query": "q2", "label": "container"}}
    frame = prometheus.pull_metrics("http://p", queries, 100, 104, session=FakePrometheus())
    assert list(frame.columns) == ["time", "cart_cpu", "front-end_cpu", "cart_mem", "front-end_mem"]
    assert frame["time"].tolist() == [100, 101, 102, 103, 104]
    assert frame["cart_cpu"].tolist() == [100 % 7, 101 % 7, 102 % 7, 103 % 7, 104 % 7]


def test_long_ranges_are_split_without_gaps_or_duplicates(monkeypatch):
    monkeypatch.setattr(prometheus, "PROMETHEUS_MAX_POINTS", 4)
    fake = FakePrometheus(services=("cart",))
    result = prometheus.query_range("http://p", "q", 0, 20, step=2, session=fake)
    assert len(fake.calls) > 1
    assert [int(t) for t, _ in result[0]["values"]] == list(range(0, 21, 2))


def test_series_without_a_service_label_are_rejected():
    fake = FakePrometheus(services=("cart",), label="pod")
    with pytest.raises(ValueError, match="no 'container' label"):
        prometheus.pull_metrics("http://p", {"cpu": {"query": "q", "label": "container"}}, 0, 2, session=fake)


def test_ambiguous_or_colliding_series_are_rejected():
    two_labels = [{"metric": {"container": "cart", "pod": "cart-1"}, "values": [[0, "1"]]}]
    with pytest.raises(ValueError, match="which label"):
        prometheus.metrics_frame({"cpu": (two_labels, None)})
    pods = [{"metric": {"container": "cart", "pod": p}, "values": [[0, "1"]]} for p in ("a", "b")]
    with pytest.raises(ValueError, match="map to 'cart_cpu'"):
        prometheus.metrics_frame({"cpu": (pods, "container")})


def test_service_names_with_underscores_are_rejected():
    # methods split columns on the first "_" to recover the service
    with pytest.raises(ValueError, match="contain no '_'"):
        prometheus.metrics_frame({"cpu": ([{"metric": {"container": "my_svc"}, "values": [[0, "1"]]}], None)})


def test_failed_query_raises():
    with pytest.raises(RuntimeError, match="bad query"):
        prometheus.query_range("http://p", "q", 0, 1, session=FakePrometheus(status="error"))


def test_load_queries_accepts_strings_and_objects(tmp_path):
    path = tmp_path / "queries.json"
    path.write_text(json.dumps({"cpu": "sum(x) by (container)", "latency-90": {"query": "y", "label": "destination_workload"}}))
    assert prometheus.load_queries(path) == {
        "cpu": {"query": "sum(x) by (container)", "label": None},
        "latency-90": {"query": "y", "label": "destination_workload"},
    }
    path.write_text(json.dumps({"cpu": 3}))
    with pytest.raises(ValueError, match="PromQL"):
        prometheus.load_queries(path)
