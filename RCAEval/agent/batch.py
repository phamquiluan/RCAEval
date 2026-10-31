"""Run a per-case solver over a task view, appending one prediction per case.

Shared by the reference agent (`run`) and the CLI runner (`run-cli`): case
selection, resume (cases already in the output are skipped), per-case
trajectory files and progress lines.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from os.path import abspath, dirname, exists, join, relpath

from RCAEval.agent import trajectory


def done_case_ids(out):
    if not exists(out):
        return set()
    with open(out) as f:
        return {json.loads(line)["case_id"] for line in f if line.strip()}


def select_cases(tasks_dir, cases=None):
    with open(join(tasks_dir, "manifest.json")) as f:
        case_ids = json.load(f)["case_ids"]
    if not cases:
        return case_ids
    unknown = sorted(set(cases) - set(case_ids))
    if unknown:
        raise ValueError(f"unknown case ids: {', '.join(unknown)}")
    return [c for c in case_ids if c in set(cases)]


def run_batch(tasks_dir, out, solve, cases=None, trajectories=None, workers=1):
    """solve(case_id, case_dir) -> (prediction, trajectory records). Returns the new predictions."""
    case_ids = select_cases(tasks_dir, cases)
    done = done_case_ids(out)
    todo = [c for c in case_ids if c not in done]
    if len(todo) < len(case_ids):
        print(f"skipping {len(case_ids) - len(todo)} case(s) already in {out}")
    out_dir = dirname(abspath(out))
    trajectories = trajectories or out + ".trajectories"
    os.makedirs(trajectories, exist_ok=True)
    lock = threading.Lock()
    results = []

    def one(case_id):
        prediction, records = solve(case_id, join(tasks_dir, "cases", case_id))
        path = join(trajectories, f"{case_id}.jsonl")
        trajectory.write(path, records)
        prediction["trajectory"] = relpath(abspath(path), out_dir)
        with lock:
            with open(out, "a") as f:
                f.write(json.dumps(prediction) + "\n")
            results.append(prediction)
            u = prediction.get("usage") or {}
            cost = "n/a" if u.get("cost_usd") is None else f"${u['cost_usd']:.3f}"
            print(f"{case_id}: {prediction.get('stop_reason')}, {u.get('tool_calls')} tool calls, {cost}, "
                  f"top 3 {prediction['ranks'][:3]}", flush=True)
        return prediction

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, todo))
    return results
