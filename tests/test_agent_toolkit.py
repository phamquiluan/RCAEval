"""Tests for the agent toolkit (RCAEval.agent.toolkit)."""
import inspect
import json
import subprocess
import sys
from os.path import join

import pytest

from agent_synthetic import INJECT, START, exported_case
from RCAEval.agent.export import export
from RCAEval.agent.toolkit import TOOLS, CaseToolkit, cap, mask_message


@pytest.fixture
def toolkit(tmp_path):
    case_dir, _ = exported_case(tmp_path)
    return CaseToolkit(case_dir)


def test_registry_matches_the_tool_methods():
    for tool in TOOLS:
        params = inspect.signature(getattr(CaseToolkit, tool["name"])).parameters
        declared = set(tool["parameters"]["properties"])
        assert declared == set(params) - {"self"}, tool["name"]
        for required in tool["parameters"].get("required", []):
            assert params[required].default is inspect.Parameter.empty


def test_describe_case_lists_telemetry_and_services(toolkit):
    r = toolkit.call("describe_case")
    assert r["inject_time"] == INJECT
    assert set(r["telemetry"]) == {"metrics", "logs", "logts", "traces", "tracets_lat", "tracets_err"}
    assert r["candidate_services"] == ["cartservice", "checkoutservice", "frontend"]


def test_list_metrics(toolkit):
    assert toolkit.call("list_metrics")["metrics_by_service"]["checkoutservice"] == ["cpu", "latency"]
    assert toolkit.call("list_metrics", {"service": "cartservice"})["metrics"] == ["cartservice_cpu", "cartservice_mem"]
    assert "error" in toolkit.call("list_metrics", {"service": "nope"})


def test_get_metric_buckets_and_compares_before_and_after(toolkit):
    r = toolkit.call("get_metric", {"name": "cartservice_cpu", "points": 10})
    assert len(r["series"]) == 10
    assert r["series"][0][0] == START
    assert r["after_inject"]["mean"] > r["before_inject"]["mean"] + 8
    r = toolkit.call("get_metric", {"name": "cartservice_cpu", "start": START + 150, "end": START + 160})
    assert len(r["series"]) == 10 and r["before_inject"] is None
    assert "did you mean cartservice_cpu" in toolkit.call("get_metric", {"name": "cartservce_cpu"})["error"]
    assert "error" in toolkit.call("get_metric", {"name": "cartservice_cpu", "start": START + 100, "end": START + 50})


def test_rank_anomalies_finds_the_planted_fault_in_every_modality(toolkit):
    assert toolkit.call("rank_anomalies")["ranked"][0]["name"] == "cartservice_cpu"
    logs = toolkit.call("rank_anomalies", {"modality": "logs"})["ranked"][0]
    assert logs["name"] == "checkoutservice_1"
    assert logs["template"].startswith("ERROR connection refused")
    for modality in ("trace_latency", "trace_errors"):
        assert toolkit.call("rank_anomalies", {"modality": modality})["ranked"][0]["name"] == "cartservice_GetCart"
    assert "error" in toolkit.call("rank_anomalies", {"modality": "nope"})
    assert len(toolkit.call("rank_anomalies", {"top_k": 2})["ranked"]) == 2


def test_search_logs_groups_by_pattern(toolkit):
    r = toolkit.call("search_logs", {"pattern": "error"})
    assert r["n_groups"] == 1
    group = r["groups"][0]
    assert group["service"] == "checkoutservice"
    assert group["pattern"] == "ERROR connection refused to cart id=<n>"
    assert (group["before"], group["after"]) == (1, 20)
    assert toolkit.call("search_logs", {"service": "frontend"})["groups"][0]["pattern"] == "INFO served request <n>"
    assert "invalid regex" in toolkit.call("search_logs", {"pattern": "("})["error"]


def test_trace_summary_and_call_graph(toolkit):
    top = toolkit.call("trace_summary")["rows"][0]
    assert (top["service"], top["operation"]) == ("cartservice", "GetCart")
    assert top["after"]["errors"] == top["after"]["requests"] and top["before"]["errors"] == 0
    assert top["after"]["p90_ms"] == 90.0
    edges = {(e["caller"], e["callee"]) for e in toolkit.call("call_graph")["edges"]}
    assert edges == {("frontend", "checkoutservice"), ("checkoutservice", "cartservice")}


def test_grpc_and_http_status_codes_count_as_errors_correctly(toolkit):
    import pandas as pd

    traces = toolkit.traces()
    traces["status"] = traces["status"].map({200: 0, 500: 14})  # gRPC: OK, UNAVAILABLE
    top = toolkit.call("trace_summary")["rows"][0]
    assert top["before"]["errors"] == 0 and top["after"]["errors"] == top["after"]["requests"]
    traces["status"] = pd.Series(302, index=traces.index)
    assert all(r["after"]["errors"] == 0 for r in toolkit.call("trace_summary")["rows"])


def test_submit_validates_against_the_candidates(toolkit):
    assert "not in candidate" in toolkit.call("submit", {"ranks": ["paymentservice"]})["error"]
    assert "empty" in toolkit.call("submit", {"ranks": [" "]})["error"]
    assert toolkit.submission is None
    assert toolkit.call("submit", {"ranks": [" CartService_CPU ", "frontend"], "reasoning": "cpu"})["accepted"]
    assert toolkit.submission == {"ranks": ["cartservice_cpu", "frontend"], "reasoning": "cpu"}


def test_every_call_is_logged(toolkit):
    toolkit.call("describe_case")
    toolkit.call("nope")
    toolkit.call("list_metrics", {"bogus": 1})
    assert [(c["seq"], c["tool"], c["ok"]) for c in toolkit.calls] == [
        (1, "describe_case", True), (2, "nope", False), (3, "list_metrics", False)]
    assert "unknown argument" in toolkit.calls[2]["error"]


def test_missing_telemetry_is_an_error_not_a_crash(tmp_path):
    toolkit = CaseToolkit(exported_case(tmp_path, with_logs=False, with_traces=False)[0])
    assert set(toolkit.call("describe_case")["telemetry"]) == {"metrics"}
    for name, args in (("search_logs", {}), ("trace_summary", {}), ("call_graph", {}),
                       ("rank_anomalies", {"modality": "logs"}), ("rank_anomalies", {"modality": "trace_errors"})):
        assert "this case has no" in toolkit.call(name, args)["error"], name


def test_results_are_capped():
    result = cap({"rows": [{"text": "x" * 100}] * 500, "n": 500}, limit=2000)
    assert result["truncated"] and len(json.dumps(result)) <= 2000 and result["n"] == 500


def test_mask_message():
    assert mask_message("took 12.5 ms for 3f2a9c81d4e5b6a7 user=42") == "took <n> ms for <id> user=<n>"


def test_metric_ranking_matches_the_baro_reference_agent(tmp_path):
    """rank_anomalies(metrics) is BARO on the same rows as docs/examples/baro_agent.py."""
    task_dir, _ = export("re1-ob", out_dir=str(tmp_path / "out"))
    case_ids = json.load(open(join(task_dir, "manifest.json")))["case_ids"][:3]
    preds = tmp_path / "preds.jsonl"
    subprocess.run([sys.executable, join("docs", "examples", "baro_agent.py"), task_dir, str(preds),
                    "--cases", *case_ids], check=True)
    for line in open(preds):
        p = json.loads(line)
        ranked = CaseToolkit(join(task_dir, "cases", p["case_id"])).call("rank_anomalies", {"top_k": 100})["ranked"]
        assert [r["name"] for r in ranked] == p["ranks"]
