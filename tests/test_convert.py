"""Converters from exported telemetry to an RCAEval case (RCAEval.convert)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from RCAEval.convert import derive, logs, otlp, prometheus
from RCAEval.convert.__main__ import main as convert_main
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


INJECT = 1_700_000_310  # failure start, on a 15 s bin boundary (bins start at ...995)


def _traces(seed=0):
    """cart/GetCart every second for 600 s: ~10 ms before INJECT, 100 ms and
    some errors after; front-end/GET / (no method) never changes."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, t in enumerate(range(1_700_000_000, 1_700_000_600)):
        after = t >= INJECT
        rows.append(("cart", "GetCart", "hipstershop.CartService/GetCart", t,
                     int(100_000 if after else 10_000 + rng.integers(-500, 500)), 14 if after and i % 3 == 0 else 0))
        rows.append(("front-end", None, "GET /", t, int(20_000 + rng.integers(-500, 500)), 0))
    frame = pd.DataFrame(rows, columns=["serviceName", "methodName", "operationName", "time", "duration", "statusCode"])
    frame["startTime"] = frame["time"] * 1_000_000
    frame["startTimeMillis"] = frame["time"] * 1_000
    frame["traceID"] = frame["spanID"] = frame["parentSpanID"] = "x"
    return frame[otlp.TRACE_COLUMNS]


def test_trace_series_counts_slow_and_error_spans_per_15s():
    err, lat = derive.trace_series(_traces(), INJECT)
    assert list(err.columns) == ["time", "cart_GetCart", "front-end_GET /"]
    assert list(lat.columns) == list(err.columns)
    assert err["time"].tolist() == list(range(1_700_000_000 - 1_700_000_000 % 15, 1_700_000_600, 15))
    before, after = err["time"] < INJECT, err["time"] >= INJECT
    # every cart span after the failure is slow; none before; front-end never
    assert (lat.loc[before, "cart_GetCart"] == 0).all()
    assert (lat.loc[after, "cart_GetCart"].iloc[:-1] == 15).all()  # the last bin is partial
    assert lat.loc[after, "cart_GetCart"].sum() == 600 - 310
    assert (lat["front-end_GET /"] == 0).all()
    # errors: every third cart span after the failure, counted (not summed codes)
    assert err.loc[before, "cart_GetCart"].sum() == 0
    assert err.loc[after, "cart_GetCart"].sum() == len([i for i in range(310, 600) if i % 3 == 0])


def test_trace_baseline_uses_only_spans_that_ended_before_the_failure():
    # with the failure at the very end, the slow spans are part of the baseline
    # and inflate it, so fewer count as slow
    _, lat = derive.trace_series(_traces(), 1_700_000_600)
    assert lat["cart_GetCart"].sum() < 300


def _logs():
    rows = []
    for t in range(1_700_000_000, 1_700_000_600):
        rows.append((t * 10**9, "cart", f"GetCart for user {t % 97} took {t % 13}ms"))
        rows.append((t * 10**9 + 1, "front-end", "request started"))
    rows.append((1_700_000_030 * 10**9, "front-end", f"failed to call 10.0.0.{7}:8080"))
    frame = pd.DataFrame(rows, columns=["timestamp", "container_name", "message"])
    frame["time"] = frame["timestamp"] // 10**9
    for c in ("level", "req_path", "error"):
        frame[c] = None
    return frame[logs.LOG_COLUMNS]


def test_log_series_counts_lines_per_template_and_container():
    pytest.importorskip("drain3")
    logts, info, out = derive.log_series(_logs())
    assert len(info) == 3  # numbers and the IP are masked, so 3 templates
    by_template = {v["template"]: (k, v["container"]) for k, v in info.items()}
    cart_id, containers = by_template["cart_GetCart for user <NUM> took <*>"]
    assert containers == ["cart"]
    assert logts["time"].tolist() == [1_700_000_000 - 1_700_000_000 % 15 + 15 * i for i in range(len(logts))]
    assert logts[f"cart_{cart_id}"].sum() == 600
    assert out["cluster_id"].astype(str).isin(info).all()
    assert out["log_template"].notna().all()


def test_derive_case_writes_the_multi_source_files(tmp_path):
    pytest.importorskip("drain3")
    _traces().to_csv(tmp_path / "traces.csv", index=False)
    _logs().to_csv(tmp_path / "logs.csv", index=False)
    with pytest.raises(FileNotFoundError, match="inject_time.txt"):
        derive.derive_case(str(tmp_path))
    (tmp_path / "inject_time.txt").write_text(str(INJECT))
    written = derive.derive_case(str(tmp_path))
    assert sorted(written) == ["cluster_info.json", "logs.csv", "logts.csv", "tracets_err.csv", "tracets_lat.csv"]
    assert {"cluster_id", "log_template"} <= set(pd.read_csv(tmp_path / "logs.csv").columns)


def test_mmbaro_runs_on_a_derived_case(tmp_path):
    pytest.importorskip("drain3")
    from RCAEval.e2e import mmbaro

    err, lat = derive.trace_series(_traces(), INJECT)
    logts, _, _ = derive.log_series(_logs())
    times = np.arange(1_700_000_000, 1_700_000_600)
    noise = np.random.default_rng(0).normal(1, 0.1, (2, len(times)))
    metrics = pd.DataFrame({"time": times, "cart_cpu": noise[0] + (times >= INJECT) * 5, "front-end_cpu": noise[1]})
    data = {"metric": metrics, "logts": logts, "tracets_err": err, "tracets_lat": lat}
    # multi-source BARO reads traces only for the datasets named in _mm_has_traces
    ranks = mmbaro(data, INJECT, dataset="re2-ob")["ranks"]
    assert ranks[0].startswith("cart")
    assert ranks.count("cart_GetCart") == 2  # its error and latency series


def test_cli_builds_a_case_from_exports(tmp_path, monkeypatch, capsys):
    pytest.importorskip("drain3")
    import requests

    case = tmp_path / "case"
    spans = [_span(f"{i:02x}", "GetCart", 1_700_000_000 + i, 10 + 90 * (i >= 300),
                   attributes=[_attr("rpc.method", stringValue="GetCart")]) for i in range(600)]
    (tmp_path / "otel.jsonl").write_text(json.dumps(_request("cart", spans)) + "\n")
    (tmp_path / "loki.jsonl").write_text("\n".join(
        json.dumps({"labels": {"container": "cart"}, "line": f"GetCart took {i}ms", "timestamp": 1_700_000_000 + i})
        for i in range(600)))

    class Session(FakePrometheus):
        headers = {}

    monkeypatch.setattr(requests, "Session", lambda: Session(services=("cart",)))
    (tmp_path / "q.json").write_text(json.dumps({"cpu": "sum(x) by (container)"}))

    assert convert_main(["metrics", "--prometheus-url", "http://p", "--queries", str(tmp_path / "q.json"),
                         "--start", "1700000000", "--end", "2023-11-14T22:23:19Z", "--header", "X-Org: 1",
                         "--out", str(case)]) == 0
    assert convert_main(["traces", str(tmp_path / "otel.jsonl"), "--out", str(case)]) == 0
    assert convert_main(["logs", str(tmp_path / "loki.jsonl"), "--field", "container=labels.container",
                         "--field", "message=line", "--out", str(case)]) == 0
    assert convert_main(["derive", str(case), "--inject-time", "1700000300"]) == 0
    assert Session.headers == {"X-Org": "1"}
    assert sorted(os.listdir(case)) == ["cluster_info.json", "inject_time.txt", "logs.csv", "logts.csv",
                                        "metrics.csv", "traces.csv", "tracets_err.csv", "tracets_lat.csv"]
    metrics = pd.read_csv(case / "metrics.csv")
    assert list(metrics.columns) == ["time", "cart_cpu"] and len(metrics) == 600
    lat = pd.read_csv(case / "tracets_lat.csv")
    assert lat.loc[lat["time"] >= 1_700_000_310, "cart_GetCart"].sum() == 290
    assert "wrote" in capsys.readouterr().out


def test_cli_refuses_to_overwrite_and_reports_bad_input(tmp_path, capsys):
    (tmp_path / "otel.json").write_text(json.dumps(_request("cart", [_span("a", "op", 1_700_000_000, 1)])))
    args = ["traces", str(tmp_path / "otel.json"), "--out", str(tmp_path / "case")]
    assert convert_main(args) == 0
    assert convert_main(args) == 1
    assert "pass --force" in capsys.readouterr().err
    assert convert_main(args + ["--force"]) == 0

    assert convert_main(["logs", str(tmp_path / "otel.json"), "--field", "container", "--out", str(tmp_path / "c")]) == 1
    assert "expected FIELD=PATH" in capsys.readouterr().err
    (tmp_path / "case" / "inject_time.txt").write_text("1700000000")
    assert convert_main(["derive", str(tmp_path / "case"), "--inject-time", "1700000001"]) == 1
    assert "says 1700000000" in capsys.readouterr().err
