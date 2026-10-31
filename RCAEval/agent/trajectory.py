"""Per-case trajectories (JSONL) and the evidence metric computed from them.

A trajectory line is a dict with a "type": "meta", "message" (reference agent
only), "tool_call" (a CaseToolkit.calls entry) or "result".
"""
import json
from os.path import dirname, exists, isabs, join

from RCAEval.benchmark.scoring import to_service


def write(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def tool_calls(path):
    with open(path) as f:
        return [r for r in map(json.loads, f) if r.get("type") == "tool_call"]


def resolve(predictions_path, trajectory):
    """A prediction's trajectory path, which is relative to the predictions file."""
    return trajectory if isabs(trajectory) else join(dirname(predictions_path), trajectory)


def inspected(calls, service):
    """Whether any call before submit targeted `service`, through a service
    argument or a metric name belonging to it."""
    for c in calls:
        if c["tool"] == "submit":
            break
        args = c.get("args") or {}
        for key in ("service", "name"):
            value = args.get(key)
            if isinstance(value, str) and to_service(value.strip().lower()) == service:
                return True
    return False


def load_calls(predictions_path, prediction):
    """The tool calls of a prediction's trajectory, or None without one."""
    trajectory = (prediction or {}).get("trajectory")
    if not isinstance(trajectory, str):
        return None
    path = resolve(predictions_path, trajectory)
    return tool_calls(path) if exists(path) else None
