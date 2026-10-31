"""The reference LLM agent: a tool-calling loop over CaseToolkit via LiteLLM.

    rcaeval-agent run --tasks agent_tasks/re2-ob --model anthropic/claude-sonnet-5-5 --out preds.jsonl

Each case runs until the model's submit is accepted, or until the step or cost
limit, after which one last turn requires submit. Predictions are appended as
cases finish and a rerun skips cases already in the output, so an interrupted
run never pays twice.
"""
import json
import time

from RCAEval.agent.batch import run_batch
from RCAEval.agent.prompt import PROMPT_VERSION, case_prompt, system_prompt
from RCAEval.agent.toolkit import TOOLS, CaseToolkit

TOOL_SPECS = [{"type": "function", "function": t} for t in TOOLS]


def litellm_completion():
    try:
        import litellm
    except ImportError:
        raise SystemExit("the reference agent needs LiteLLM: pip install RCAEval[agent]")
    litellm.drop_params = True  # e.g. temperature on models that reject it
    return litellm.completion


def litellm_cost(response):
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response))
    except Exception:
        return None  # model without a known price


def uses_prompt_cache(model):
    return "claude" in model or model.startswith("anthropic/")


def with_cache_marks(messages):
    """Copy of messages with Anthropic cache breakpoints on the system prompt and
    the latest message, so each turn re-reads the conversation from cache."""
    marked = []
    for i, m in enumerate(messages):
        m = dict(m)
        if (i == 0 or i == len(messages) - 1) and isinstance(m.get("content"), str) and m["content"]:
            m["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
        marked.append(m)
    return marked


def run_case(case_dir, model, max_steps=30, max_cost_usd=None, temperature=0.0,
             completion=None, cost=None, prompt_version=PROMPT_VERSION):
    """Investigate one case; returns (prediction dict, trajectory records)."""
    completion = completion or litellm_completion()
    cost = cost or litellm_cost
    toolkit = CaseToolkit(case_dir)
    messages = [{"role": "system", "content": system_prompt(prompt_version)},
                {"role": "user", "content": case_prompt(toolkit.task)}]
    records = [{"type": "meta", "case_id": toolkit.task["case_id"], "model": model,
                "prompt_version": prompt_version}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "steps": 0}
    unpriced = False
    stop_reason = None
    started = time.time()
    cache = uses_prompt_cache(model)

    while stop_reason is None:
        over_budget = max_cost_usd is not None and usage["cost_usd"] >= max_cost_usd
        final_turn = usage["steps"] >= max_steps or over_budget
        kwargs = {"model": model, "messages": with_cache_marks(messages) if cache else messages,
                  "tools": TOOL_SPECS, "temperature": temperature}
        if final_turn:
            messages.append({"role": "user", "content": "Limit reached: call submit now with your best ranking."})
            kwargs["messages"] = with_cache_marks(messages) if cache else messages
            kwargs["tool_choice"] = {"type": "function", "function": {"name": "submit"}}

        response = completion(**kwargs)
        usage["steps"] += 1
        u = getattr(response, "usage", None)
        usage["input_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        usage["output_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
        c = cost(response)
        if c is None:
            unpriced = True
        else:
            usage["cost_usd"] += c

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        assistant = {"role": "assistant", "content": message.content or ""}
        if tool_calls:
            assistant["tool_calls"] = [{"id": tc.id, "type": "function",
                                        "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                       for tc in tool_calls]
        messages.append(assistant)
        records.append({"type": "message", **assistant})

        if not tool_calls:
            if final_turn:
                stop_reason = "no_submit"
            else:
                messages.append({"role": "user", "content": "Use the tools, and finish by calling submit."})
            continue

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
            except ValueError as e:
                result = {"error": f"could not parse arguments: {e}"}
                records.append({"type": "tool_call", "seq": None, "tool": tc.function.name, "args": None,
                                "ok": False, "error": result["error"]})
            else:
                result = toolkit.call(tc.function.name, args)
                records.append({"type": "tool_call", **toolkit.calls[-1]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)})
            if toolkit.submission is not None:
                stop_reason = "submit"
                break
        if stop_reason is None and final_turn:
            stop_reason = "no_submit"

    usage["cost_usd"] = None if unpriced else round(usage["cost_usd"], 6)
    usage["wall_time_s"] = round(time.time() - started, 2)
    usage["tool_calls"] = len(toolkit.calls)
    submission = toolkit.submission or {"ranks": [], "reasoning": None}
    prediction = {
        "case_id": toolkit.task["case_id"],
        "ranks": submission["ranks"],
        "reasoning": submission["reasoning"],
        "model": model,
        "prompt_version": prompt_version,
        "stop_reason": stop_reason,
        "usage": usage,
    }
    records.append({"type": "result", **prediction})
    return prediction, records


def run_tasks(tasks_dir, model, out, cases=None, trajectories=None, workers=1, **case_kwargs):
    """Run the agent over a task view, appending to `out`; returns the new predictions."""
    completion = case_kwargs.pop("completion", None) or litellm_completion()

    def solve(case_id, case_dir):
        return run_case(case_dir, model, completion=completion, **case_kwargs)

    return run_batch(tasks_dir, out, solve, cases=cases, trajectories=trajectories, workers=workers)
