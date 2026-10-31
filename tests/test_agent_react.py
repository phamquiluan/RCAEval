"""Tests for the reference LLM agent loop (RCAEval.agent.react), with a scripted LLM."""
import json
from os.path import exists, join
from types import SimpleNamespace

import pytest

from agent_synthetic import exported_case, exported_tasks
from RCAEval.agent.predictions import run as evaluate
from RCAEval.agent.react import run_case, run_tasks, with_cache_marks


def tool_call(name, args, i=0):
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=f"call_{name}_{i}", function=SimpleNamespace(name=name, arguments=raw))


def reply(*calls, text=""):
    message = SimpleNamespace(content=text, tool_calls=list(calls) or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


class ScriptedLLM:
    """Replies from a script; repeats the last reply when the script runs out."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, **kwargs):
        # a snapshot: the agent keeps appending to the same message list
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def run(case_dir, llm, cost=lambda r: 0.01, **kwargs):
    return run_case(case_dir, "test-model", completion=llm, cost=cost, **kwargs)


@pytest.fixture
def case_dir(tmp_path):
    return exported_case(tmp_path)[0]


def test_tools_run_until_submit(case_dir):
    llm = ScriptedLLM(
        reply(tool_call("describe_case", {})),
        reply(tool_call("rank_anomalies", {"top_k": 3}), tool_call("get_metric", {"name": "cartservice_cpu"}, 1)),
        reply(tool_call("submit", {"ranks": ["cartservice_cpu", "checkoutservice"], "reasoning": "cpu jump"})),
    )
    prediction, records = run(case_dir, llm)
    assert prediction["ranks"] == ["cartservice_cpu", "checkoutservice"]
    assert prediction["reasoning"] == "cpu jump"
    assert prediction["stop_reason"] == "submit"
    assert prediction["usage"]["steps"] == 3
    assert prediction["usage"]["tool_calls"] == 4
    assert prediction["usage"]["input_tokens"] == 300
    assert prediction["usage"]["cost_usd"] == pytest.approx(0.03)
    assert [r["tool"] for r in records if r["type"] == "tool_call"] == [
        "describe_case", "rank_anomalies", "get_metric", "submit"]
    # each tool result goes back to the model under its call id
    last = llm.requests[-1]["messages"]
    tool_messages = [m for m in last if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == [
        "call_describe_case_0", "call_rank_anomalies_0", "call_get_metric_1"]
    assert json.loads(tool_messages[1]["content"])["ranked"][0]["name"] == "cartservice_cpu"
    assert "tool_choice" not in llm.requests[-1]


def test_rejected_submit_continues_the_investigation(case_dir):
    llm = ScriptedLLM(
        reply(tool_call("submit", {"ranks": ["paymentservice"]})),
        reply(tool_call("submit", {"ranks": ["cartservice"]})),
    )
    prediction, _ = run(case_dir, llm)
    assert prediction["ranks"] == ["cartservice"] and prediction["usage"]["steps"] == 2
    assert "not in candidate" in llm.requests[1]["messages"][-1]["content"]


def test_step_limit_requires_submit_then_gives_up(case_dir):
    llm = ScriptedLLM(reply(tool_call("describe_case", {})))
    prediction, _ = run(case_dir, llm, max_steps=2)
    assert len(llm.requests) == 3
    assert llm.requests[-1]["tool_choice"] == {"type": "function", "function": {"name": "submit"}}
    assert "Limit reached" in llm.requests[-1]["messages"][-1]["content"]
    assert prediction["ranks"] == [] and prediction["stop_reason"] == "no_submit"


def test_cost_limit_requires_submit(case_dir):
    llm = ScriptedLLM(reply(tool_call("describe_case", {})), reply(tool_call("describe_case", {})),
                      reply(tool_call("submit", {"ranks": ["frontend"]})))
    prediction, _ = run(case_dir, llm, cost=lambda r: 1.0, max_cost_usd=1.5)
    assert "tool_choice" in llm.requests[2] and prediction["ranks"] == ["frontend"]


def test_text_replies_and_bad_arguments_are_handled(case_dir):
    llm = ScriptedLLM(
        reply(text="Let me think."),
        reply(tool_call("get_metric", "{not json")),
        reply(tool_call("submit", {"ranks": ["frontend"]})),
    )
    prediction, records = run(case_dir, llm)
    assert llm.requests[1]["messages"][-1] == {"role": "user", "content": "Use the tools, and finish by calling submit."}
    assert "could not parse arguments" in llm.requests[2]["messages"][-1]["content"]
    assert prediction["ranks"] == ["frontend"]
    assert any(r["type"] == "tool_call" and r["tool"] == "get_metric" and not r["ok"] for r in records)


def test_unknown_price_is_recorded_as_null(case_dir):
    prediction, _ = run(case_dir, ScriptedLLM(reply(tool_call("submit", {"ranks": ["frontend"]}))), cost=lambda r: None)
    assert prediction["usage"]["cost_usd"] is None


def test_cache_marks_go_on_the_system_prompt_and_latest_message():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "", "tool_calls": []}, {"role": "tool", "content": "r"}]
    marked = with_cache_marks(messages)
    assert marked[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert marked[3]["content"][0] == {"type": "text", "text": "r", "cache_control": {"type": "ephemeral"}}
    assert marked[1]["content"] == "u" and messages[3]["content"] == "r"


def test_run_tasks_writes_predictions_resumes_and_feeds_evaluate(tmp_path):
    task_dir, answers_path = exported_tasks(tmp_path)
    out = str(tmp_path / "preds.jsonl")

    def llm(**kwargs):
        case_id = kwargs["messages"][1]["content"].split(":")[0].split()[-1]
        service = "cartservice" if case_id.endswith("1") else "checkoutservice"
        calls = [m for m in kwargs["messages"] if m["role"] == "tool"]
        if not calls:
            return reply(tool_call("get_metric", {"name": f"{service}_cpu"}))
        return reply(tool_call("submit", {"ranks": [f"{service}_cpu"]}))

    first = run_tasks(task_dir, "test-model", out, completion=llm, cost=lambda r: 0.0)
    assert len(first) == 2
    assert run_tasks(task_dir, "test-model", out, completion=llm, cost=lambda r: 0.0) == []
    lines = [json.loads(line) for line in open(out)]
    assert len(lines) == 2
    assert all(exists(join(tmp_path, p["trajectory"])) for p in lines)

    summary = evaluate(out, tasks_dir=task_dir)
    assert summary["evidence"]["n"] == 2
    answers = {json.loads(line)["case_id"]: json.loads(line)["service"] for line in open(answers_path)}
    expected = sum(("cartservice" if c.endswith("1") else "checkoutservice") == s for c, s in answers.items()) / 2
    assert summary["service"]["AC@1"] == expected
