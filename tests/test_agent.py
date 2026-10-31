"""Tests for the agent onboarding workflow: blind export and predictions evaluation."""
import glob
import json
import os
import subprocess
import sys
import time
from os.path import basename, join

import pytest

from RCAEval.agent.__main__ import main as cli
from RCAEval.agent.export import export
from RCAEval.agent.predictions import PredictionsError, load_predictions, run
from RCAEval.benchmark.scoring import fault_to_indicator, indicator_ranks, service_ranks
from RCAEval.classes.graph import Node

SERVICES = ["cartservice", "checkoutservice", "emailservice"]
FAULTS = ["cpu", "mem", "delay"]
HEADER = ["time", "cartservice_cpu", "cartservice_mem", "cartservice_latency-50", "cartservice_latency-90",
          "checkoutservice_cpu", "checkoutservice_mem", "checkoutservice_latency-90",
          "emailservice_cpu", "emailservice_mem", "emailservice_latency-90", "redis-db_cpu", "time.1"]
DECOYS = ["metrics_postprocess.log", "pod-node-1.csv"]


def make_dataset(root, faults=FAULTS, n_cases=2):
    """A tiny dataset in the raw `<service>_<fault>/<n>/` layout."""
    for service in SERVICES:
        for fault in faults:
            for n in range(1, n_cases + 1):
                d = join(root, f"{service}_{fault}", str(n))
                os.makedirs(d)
                with open(join(d, "data.csv"), "w") as f:
                    f.write(",".join(HEADER) + "\n")
                    for t in range(1000, 1010):
                        f.write(",".join([str(t)] + ["1"] * (len(HEADER) - 2) + [str(t)]) + "\n")
                with open(join(d, "inject_time.txt"), "w") as f:
                    f.write("1005\n")
                with open(join(d, "logs.csv"), "w") as f:
                    f.write("time,container_name,message\n")
                for decoy in DECOYS:
                    with open(join(d, decoy), "w") as f:
                        f.write(f"{service} {fault}\n")
                with open(join(d, "cluster_info.json"), "w") as f:
                    json.dump({"1": {"template": "GET /cart <:NUM:>", "container": ["cartservice"]}}, f)
    return root


@pytest.fixture
def exported(tmp_path):
    data_root = make_dataset(str(tmp_path / "raw"))
    task_dir, answers_path = export("re2-ob", out_dir=str(tmp_path / "out"), data_root=data_root)
    answers = [json.loads(line) for line in open(answers_path)]
    return task_dir, answers_path, answers


def write_preds(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write((line if isinstance(line, str) else json.dumps(line)) + "\n")
    return str(path)


# --- export ---

def test_export_paths_do_not_reveal_the_answer(exported):
    task_dir, answers_path, _ = exported
    assert not answers_path.startswith(task_dir + os.sep)
    for dirpath, dirnames, filenames in os.walk(task_dir):
        for name in dirnames + filenames:
            path = join(dirpath, name)
            assert not os.path.islink(path)
            assert not any(token in name for token in SERVICES + FAULTS), path
            assert name not in DECOYS + ["inject_time.txt", "data.csv", "cluster_info.json"], path


def test_export_writes_task_json_with_main_py_candidates(exported):
    task_dir, _, answers = exported
    task = json.load(open(join(task_dir, "cases", answers[0]["case_id"], "task.json")))
    assert task["inject_time"] == 1005
    assert task["standard_window"] == {"start": 405, "end": 1605}
    assert task["files"] == {"metrics": "metrics.csv", "logs": "logs.csv", "log_templates": "log_templates.json"}
    # no time columns, p50 latency dropped, p90 latency renamed
    assert task["candidate_indicators"] == [
        "cartservice_cpu", "cartservice_mem", "cartservice_latency",
        "checkoutservice_cpu", "checkoutservice_mem", "checkoutservice_latency",
        "emailservice_cpu", "emailservice_mem", "emailservice_latency", "redis-db_cpu"]
    # a database container folds into its service, as in main.py's scoring
    assert task["candidate_services"] == SERVICES + ["redis"]


def test_export_answers_carry_the_labels(exported):
    task_dir, _, answers = exported
    assert len(answers) == len(SERVICES) * len(FAULTS) * 2
    manifest = json.load(open(join(task_dir, "manifest.json")))
    assert manifest["case_ids"] == [a["case_id"] for a in answers]
    for a in answers:
        service, fault = a["source"].split("/")[0].split("_")
        assert (a["service"], a["fault"]) == (service, fault)
        assert a["indicator"] == {"cpu": "cpu", "mem": "mem", "delay": "latency"}[fault]
        assert a["answer_in_candidates"] and a["indicator_in_candidates"]


def test_case_ids_are_stable_and_not_in_source_order(tmp_path):
    data_root = make_dataset(str(tmp_path / "raw"))
    _, a1 = export("re2-ob", out_dir=str(tmp_path / "out1"), data_root=data_root)
    _, a2 = export("re2-ob", out_dir=str(tmp_path / "out2"), data_root=data_root, copy=True)
    assert open(a1).read() == open(a2).read()
    sources = [json.loads(line)["source"] for line in open(a1)]
    assert sources != sorted(sources)


def test_export_refuses_to_overwrite_without_force(tmp_path):
    data_root = make_dataset(str(tmp_path / "raw"))
    export("re2-ob", out_dir=str(tmp_path / "out"), data_root=data_root)
    with pytest.raises(FileExistsError):
        export("re2-ob", out_dir=str(tmp_path / "out"), data_root=data_root)
    export("re2-ob", out_dir=str(tmp_path / "out"), data_root=data_root, force=True)


# --- evaluate ---

def test_perfect_predictions_score_one(exported, tmp_path):
    task_dir, _, answers = exported
    preds = write_preds(tmp_path / "p.jsonl",
                        [{"case_id": a["case_id"], "ranks": [f"{a['service']}_{a['indicator']}"]} for a in answers])
    summary = run(preds, tasks_dir=task_dir)
    assert summary["n_predicted"] == len(answers)
    for level in ("service", "indicator"):
        assert summary[level]["AC@1"] == 1.0
        assert summary[level]["Avg@5"] == 1.0


def test_partial_predictions_score_as_computed_by_hand(exported, tmp_path):
    task_dir, _, answers = exported
    lines = []
    for i, a in enumerate(answers):
        decoy = next(s for s in SERVICES if s != a["service"])
        if i % 2 == 0:  # the right service second, but with the wrong indicator
            lines.append({"case_id": a["case_id"], "ranks": [f"{decoy}_cpu", f"{a['service']}_wrong"]})
        else:  # the right service and indicator first; a duplicate service entry is ignored
            lines.append({"case_id": a["case_id"], "ranks": [f"{a['service']}_{a['indicator']}", a["service"]]})
    summary = run(write_preds(tmp_path / "p.jsonl", lines), tasks_dir=task_dir)
    s, f = summary["service"], summary["indicator"]
    assert s["AC@1"] == 0.5 and s["AC@3"] == 1.0
    assert s["Avg@5"] == pytest.approx((0.5 + 4 * 1.0) / 5)
    assert f["AC@1"] == 0.5 and f["AC@5"] == 0.5


def test_missing_cases_count_as_misses(exported, tmp_path, capsys):
    task_dir, _, answers = exported
    a = answers[0]
    preds = write_preds(tmp_path / "p.jsonl", [{"case_id": a["case_id"], "ranks": [a["service"]]}])
    summary = run(preds, tasks_dir=task_dir)
    assert summary["n_predicted"] == 1
    assert summary["service"]["AC@1"] == pytest.approx(1 / len(answers))
    assert f"({len(answers) - 1} missing, scored as misses)" in capsys.readouterr().out


def test_invalid_predictions_list_every_problem(exported, tmp_path, capsys):
    task_dir, _, answers = exported
    ok = {"case_id": answers[0]["case_id"], "ranks": ["cartservice"]}
    preds = write_preds(tmp_path / "p.jsonl", [
        ok,
        ok,                                                   # duplicate
        "{not json",                                          # malformed
        {"case_id": "re2-ob-9999", "ranks": []},              # unknown
        {"case_id": answers[1]["case_id"], "ranks": "cartservice"},  # ranks not a list
        {"ranks": ["cartservice"]},                           # no case_id
    ])
    with pytest.raises(PredictionsError) as e:
        load_predictions(preds, [a["case_id"] for a in answers])
    assert len(e.value.errors) == 5
    assert cli(["evaluate", "--tasks", task_dir, "--predictions", preds]) == 1
    err = capsys.readouterr().err
    for fragment in ("line 2: duplicate", "line 3: not valid JSON", "line 4: unknown case_id",
                     "line 5: ranks must be a list", "line 6: missing or non-string case_id"):
        assert fragment in err


def test_names_are_normalised_before_matching(exported, tmp_path):
    task_dir, _, answers = exported
    preds = write_preds(tmp_path / "p.jsonl", [
        {"case_id": a["case_id"], "ranks": [f"  {a['service'].upper()}_{a['indicator'].upper()} "]} for a in answers])
    assert run(preds, tasks_dir=task_dir)["indicator"]["AC@1"] == 1.0


def test_code_level_faults_have_no_indicator_score(tmp_path):
    data_root = make_dataset(str(tmp_path / "raw"), faults=["f1", "f2"])
    task_dir, _ = export("re3-ob", out_dir=str(tmp_path / "out"), data_root=data_root)
    answers = [json.loads(line) for line in open(task_dir + ".answers.jsonl")]
    assert all(a["indicator"] is None for a in answers)
    preds = write_preds(tmp_path / "p.jsonl", [{"case_id": a["case_id"], "ranks": [a["service"]]} for a in answers])
    summary = run(preds, tasks_dir=task_dir)
    assert summary["service"]["AC@1"] == 1.0
    assert summary["indicator"] is None
    assert list(summary["service_by_fault"]) == ["f1", "f2"]


def test_usage_is_aggregated(exported, tmp_path):
    task_dir, _, answers = exported
    preds = write_preds(tmp_path / "p.jsonl", [
        {"case_id": a["case_id"], "ranks": [], "usage": {"cost_usd": 0.5, "tool_calls": 3}} for a in answers[:4]])
    usage = run(preds, tasks_dir=task_dir)["usage"]
    assert usage["cost_usd"] == {"total": 2.0, "mean": 0.5, "n": 4}
    assert usage["tool_calls"]["total"] == 12


def test_answers_from_another_export_are_rejected(tmp_path):
    t1, _ = export("re2-ob", out_dir=str(tmp_path / "a"), data_root=make_dataset(str(tmp_path / "raw1")))
    _, a2 = export("re2-ss", out_dir=str(tmp_path / "b"), data_root=make_dataset(str(tmp_path / "raw2")))
    preds = write_preds(tmp_path / "p.jsonl", [])
    with pytest.raises(ValueError, match="does not belong"):
        run(preds, tasks_dir=t1, answers_path=a2)


# --- scoring helpers shared with main.py ---

def test_service_ranks_fold_databases_and_drop_duplicates():
    ranks = ["carts-db_cpu", "carts_mem", "orders_latency", "carts", "orders_cpu"]
    assert service_ranks(ranks) == [Node("carts", "unknown"), Node("orders", "unknown")]
    assert service_ranks([]) == []


def test_indicator_ranks_keep_services_without_an_indicator():
    assert indicator_ranks(["carts_cpu", "orders"]) == [Node("carts", "cpu"), Node("orders", "unknown")]


def test_disk_faults_score_against_latency_on_re1_only():
    assert fault_to_indicator("disk", "re1-ob") == "latency"
    assert fault_to_indicator("disk", "online-boutique") == "latency"
    assert fault_to_indicator("disk", "re2-ob") == "diskio"
    assert fault_to_indicator("loss", "re2-tt") == "latency"
    assert fault_to_indicator("socket", "re2-ss") == "socket"
    assert fault_to_indicator("f3", "re3-tt") is None


# --- parity with main.py on real data ---

def test_baro_agent_matches_main_py(tmp_path):
    """The reference agent in docs/examples, run on the blind view, ranks a case
    exactly as main.py ranks the same source case."""
    started = time.time()
    result = subprocess.run([sys.executable, "main.py", "--method", "baro", "--dataset", "re1-ob", "--test"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    fresh = [p for p in glob.glob(join("output", "results", "*.json")) if os.path.getmtime(p) >= started]
    assert len(fresh) == 2

    task_dir, answers_path = export("re1-ob", out_dir=str(tmp_path / "out"))
    by_source = {a["source"].replace("/", "_"): a["case_id"] for a in map(json.loads, open(answers_path))}
    expected = {by_source[basename(p)[: -len(".json")]]: json.load(open(p))["0"] for p in fresh}

    preds = tmp_path / "preds.jsonl"
    result = subprocess.run([sys.executable, join("docs", "examples", "baro_agent.py"), task_dir, str(preds),
                             "--cases", *expected], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    got = {p["case_id"]: p["ranks"] for p in map(json.loads, open(preds))}
    assert got == expected
