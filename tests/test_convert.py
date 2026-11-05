"""Converters from exported telemetry to an RCAEval case (RCAEval.convert)."""
import json

import pandas as pd
import pytest

from RCAEval.convert import logs, otlp, prometheus
from RCAEval.convert.common import to_unix_ns


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


def _attr(key, **value):
    return {"key": key, "value": value}


def _span(span_id, name, start_s, ms, parent="", attributes=(), status=None):
    span = {
        "traceId": "5B8EFFF798038103D269B633813FC60C", "spanId": span_id, "parentSpanId": parent,
        "name": name, "kind": 2, "attributes": list(attributes),
        "startTimeUnixNano": str(int(start_s * 1e9)), "endTimeUnixNano": str(int(start_s * 1e9 + ms * 1e6)),
    }
    if status is not None:
        span["status"] = status
    return span


def _request(service, spans, key="scopeSpans"):
    return {"resourceSpans": [{
        "resource": {"attributes": [_attr("service.name", stringValue=service)]},
        key: [{"scope": {"name": "lib"}, "spans": spans}],
    }]}


def test_otlp_spans_become_trace_rows(tmp_path):
    grpc = _span("A1", "hipstershop.CartService/GetCart", 1700000001.5, 12, parent="B2",
                 attributes=[_attr("rpc.method", stringValue="GetCart"), _attr("rpc.grpc.status_code", intValue="14")])
    http = _span("B2", "GET /cart", 1700000001, 30, status={"code": 2},
                 attributes=[_attr("http.response.status_code", intValue="503")])
    plain = _span("C3", "render", 1700000002, 1, status={"code": "STATUS_CODE_ERROR"})
    path = tmp_path / "traces.jsonl"
    # the collector's file exporter writes one request per line
    path.write_text(json.dumps(_request("cart", [grpc])) + "\n" + json.dumps(_request("front-end", [http, plain])) + "\n")

    frame = otlp.read_otlp_traces([str(path)])
    assert list(frame.columns) == otlp.TRACE_COLUMNS
    assert frame["spanID"].tolist() == ["b2", "a1", "c3"]  # sorted by start
    row = frame.set_index("spanID").loc["a1"]
    assert (row["serviceName"], row["methodName"], row["parentSpanID"]) == ("cart", "GetCart", "b2")
    assert (row["time"], row["startTimeMillis"], row["startTime"], row["duration"]) == (
        1700000001, 1700000001500, 1700000001500000, 12000)
    # gRPC code; HTTP code of an ERROR span; 2 for an ERROR span without one
    assert frame.set_index("spanID")["statusCode"].to_dict() == {"b2": 503, "a1": 14, "c3": 2}
    assert pd.isna(frame.set_index("spanID").loc["b2", "parentSpanID"])


def test_otlp_ok_spans_have_status_zero_and_old_key_is_read(tmp_path):
    path = tmp_path / "t.json"
    ok = _span("D4", "op", 1700000000, 5, status={"code": 1},
               attributes=[_attr("http.response.status_code", intValue="404")])
    path.write_text(json.dumps(_request("cart", [ok], key="instrumentationLibrarySpans")))
    assert otlp.read_otlp_traces(str(path))["statusCode"].tolist() == [0]


def test_otlp_without_service_name_is_rejected(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": []}]}))
    with pytest.raises(ValueError, match="service.name"):
        otlp.read_otlp_traces(str(path))
    path.write_text(json.dumps({"resourceLogs": []}))
    with pytest.raises(ValueError, match="not an OTLP trace export"):
        otlp.read_otlp_traces(str(path))


def test_timestamps_in_any_unit_or_as_dates_become_nanoseconds():
    ns = 1705353846067349596
    assert to_unix_ns([1705353846, 1705353846067, 1705353846067349, ns]).tolist() == [
        1705353846_000_000_000, 1705353846_067_000_000, 1705353846_067_349_000, ns]
    assert to_unix_ns(["2024-01-15T21:24:06.067349596Z", "2024-01-15 21:24:06"]).tolist() == [
        ns, 1705353846_000_000_000]
    with pytest.raises(ValueError, match="mix"):
        to_unix_ns([1705353846, "2024-01-15"])


def test_loki_export_maps_nested_fields(tmp_path):
    path = tmp_path / "loki.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"labels": {"container": "cart"}, "line": "GetCart took 5ms", "timestamp": "2024-01-15T21:24:07Z"},
        {"labels": {"container": "front-end"}, "line": "request started", "timestamp": "2024-01-15T21:24:06Z"},
    ]))
    frame = logs.read_logs([str(path)], {"container": "labels.container", "message": "line"})
    assert list(frame.columns) == logs.LOG_COLUMNS
    assert frame["container_name"].tolist() == ["front-end", "cart"]  # sorted by time
    assert frame["time"].tolist() == [1705353846, 1705353847]
    assert frame["level"].isna().all()


def test_elasticsearch_keys_with_dots_are_matched_whole(tmp_path):
    path = tmp_path / "es.json"
    path.write_text(json.dumps([{"_source": {
        "@timestamp": 1705353846067, "kubernetes.container.name": "cart", "log": {"level": "error", "msg": "x"},
    }}]))
    frame = logs.read_logs(str(path), {
        "timestamp": "_source.@timestamp", "container": "_source.kubernetes.container.name",
        "message": "_source.log", "level": "_source.log.level",
    })
    row = frame.iloc[0]
    assert (row["container_name"], row["level"], row["timestamp"]) == ("cart", "error", 1705353846067000000)
    assert json.loads(row["message"]) == {"level": "error", "msg": "x"}  # an object is kept as JSON


def test_csv_with_rcaeval_columns_needs_no_mapping(tmp_path):
    path = tmp_path / "logs.csv"
    path.write_text("timestamp,container_name,message,level\n1705353846067349596,cart,ok,info\n")
    frame = logs.read_logs(str(path))
    assert frame.iloc[0][["timestamp", "container_name", "message", "level"]].tolist() == [
        1705353846067349596, "cart", "ok", "info"]


def test_incomplete_log_records_are_dropped_with_a_warning(tmp_path):
    path = tmp_path / "l.jsonl"
    path.write_text(json.dumps({"timestamp": 1, "container_name": "cart", "message": "a"}) + "\n"
                    + json.dumps({"timestamp": 2, "message": "no container"}) + "\n")
    with pytest.warns(UserWarning, match="dropped 1 of 2"):
        frame = logs.read_logs(str(path))
    assert frame["message"].tolist() == ["a"]
    with pytest.raises(ValueError, match="unknown log fields"):
        logs.read_logs(str(path), {"service": "x"})
