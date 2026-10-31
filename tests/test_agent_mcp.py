"""Tests for the MCP server and the agent CLI runner (needs the `agent` extra)."""
import json
import sys
from os.path import abspath, exists, join

import pytest

pytest.importorskip("mcp")

from agent_synthetic import exported_tasks
from RCAEval.agent.predictions import run as evaluate
from RCAEval.agent.runner import build_command, claude_usage, run_tasks

FAKE_AGENT = abspath(join("tests", "data", "fake_mcp_agent.py"))  # agents run in an empty temp dir


def test_runner_drives_an_mcp_agent_end_to_end(tmp_path):
    task_dir, answers_path = exported_tasks(tmp_path)
    out = str(tmp_path / "preds.jsonl")
    command = f"{sys.executable} {FAKE_AGENT} {{mcp_config}}"
    results = run_tasks(task_dir, out, agent="custom", command=command, timeout=120)
    assert [p["stop_reason"] for p in results] == ["submit", "submit"]

    lines = [json.loads(line) for line in open(out)]
    for p in lines:
        run_dir = join(out + ".runs", p["case_id"])
        assert json.load(open(join(run_dir, "submission.json")))["ranks"] == p["ranks"]
        calls = [json.loads(line)["tool"] for line in open(join(run_dir, "calls.jsonl"))]
        assert calls == ["describe_case", "rank_anomalies", "get_metric", "submit"]
        assert p["usage"]["tool_calls"] == 4 and p["model"] == "custom"
        assert exists(join(tmp_path, p["trajectory"]))

    # the fake agent submits rank_anomalies' order, which leads with the planted fault
    summary = evaluate(out, tasks_dir=task_dir)
    assert summary["service"]["AC@1"] == 1.0
    assert summary["evidence"] == {"n": 2, "inspected_root": 1.0, "AC@1_inspected": 1.0, "AC@1_not_inspected": None}

    # a rerun skips both cases
    assert run_tasks(task_dir, out, agent="custom", command=command) == []


def test_timeout_and_no_submit_are_misses(tmp_path):
    task_dir, _ = exported_tasks(tmp_path)
    out = str(tmp_path / "preds.jsonl")
    results = run_tasks(task_dir, out, agent="custom", timeout=2,
                        command=f"{sys.executable} -c \"import time; time.sleep(10)\"")
    assert {p["stop_reason"] for p in results} == {"timeout"}
    assert all(p["ranks"] == [] for p in results)
    out2 = str(tmp_path / "preds2.jsonl")
    results = run_tasks(task_dir, out2, agent="custom", command=f"{sys.executable} -c pass")
    assert {p["stop_reason"] for p in results} == {"no_submit"}


def test_claude_code_preset_reaches_data_only_through_mcp():
    cmd = build_command("claude-code", None, "the prompt", "/tmp/cfg.json", "/tmp/w", model="sonnet")
    assert cmd[:3] == ["claude", "-p", "the prompt"]
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--bare" in cmd and "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__rcaeval"
    assert cmd[cmd.index("--mcp-config") + 1] == "/tmp/cfg.json"
    assert cmd[-2:] == ["--model", "sonnet"]


def test_custom_command_template():
    cmd = build_command("custom", "agent --mcp {mcp_config} --cwd {workdir} -p {prompt}", "two words", "c.json", "/w")
    assert cmd == ["agent", "--mcp", "c.json", "--cwd", "/w", "-p", "two words"]
    with pytest.raises(ValueError):
        build_command("custom", None, "p", "c", "w")


def test_claude_usage_parsing():
    out = json.dumps({"total_cost_usd": 0.12, "num_turns": 7, "usage": {
        "input_tokens": 10, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 1000, "output_tokens": 50}})
    assert claude_usage(out) == {"cost_usd": 0.12, "turns": 7, "input_tokens": 1110, "output_tokens": 50}
    assert claude_usage("not json") == {}
