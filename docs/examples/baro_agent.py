"""A reference "agent" for the predictions-file workflow (docs/AGENTS.md).

It runs BARO on each case of a blind task view and writes predictions.jsonl.
An LLM agent plugs in at the same place: read task.json and the telemetry,
produce a ranking, write one line per case.

    python docs/examples/baro_agent.py agent_tasks/re2-ob preds.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from RCAEval.e2e import baro

parser = argparse.ArgumentParser()
parser.add_argument("tasks", type=Path, help="exported task directory, e.g. agent_tasks/re2-ob")
parser.add_argument("out", help="predictions file to write")
parser.add_argument("--cases", nargs="*", help="only these case ids (default: all)")
args = parser.parse_args()

manifest = json.loads((args.tasks / "manifest.json").read_text())
with open(args.out, "w") as f:
    for case_id in args.cases or manifest["case_ids"]:
        case = args.tasks / "cases" / case_id
        task = json.loads((case / "task.json").read_text())
        start = time.time()

        # read the metrics the way main.py prepares them for every method
        data = pd.read_csv(case / task["files"]["metrics"])
        data = data.drop(columns=["time.1"], errors="ignore")
        data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
        data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        inject_time = task["inject_time"]
        half = inject_time - task["standard_window"]["start"]  # seconds = rows at 1 Hz
        data = pd.concat([data[data["time"] < inject_time].tail(half),
                          data[data["time"] >= inject_time].head(half)], ignore_index=True)
        data = data.rename(columns=lambda c: c.replace("_latency-90", "_latency"))

        ranks = baro(data, inject_time, dataset=task["dataset"])["ranks"]
        f.write(json.dumps({
            "case_id": case_id,
            "ranks": ranks,
            "model": "baro",
            "usage": {"wall_time_s": round(time.time() - start, 3)},
        }) + "\n")
