"""Derive the time series the multi-source methods read from ``logs.csv`` and
``traces.csv``, in 15-second bins aligned to the epoch:

- ``logts.csv``: log lines per template, one ``{container}_{cluster_id}``
  column each; the templates are mined with Drain3 and listed in
  ``cluster_info.json``, and ``logs.csv`` gains ``cluster_id`` and
  ``log_template`` columns
- ``tracets_err.csv``: error spans (``statusCode`` not 0) per operation
- ``tracets_lat.csv``: slow spans per operation, i.e. longer than the mean
  plus 3 standard deviations of that operation's spans that ended before the
  failure started

An operation is ``{serviceName}_{methodName}``, or ``{serviceName}_{operationName}``
for spans without a method.

These follow the scripts behind the released datasets, except that the
released ``tracets_err`` summed status codes instead of counting error spans,
the released latency baseline mixed milliseconds and microseconds when
selecting the spans before the failure, the released operation names and
Drain3 settings were specific to each system, and each released template
column belonged to the first container that logged it.
"""
import json
import os

import pandas as pd

BIN_SECONDS = 15

# Drain3 masks used when no .ini is given: values that vary between otherwise
# identical lines. A system-specific .ini (masks, sim_th) usually merges more.
_B, _E = r"((?<=[^A-Za-z0-9])|^)", r"((?=[^A-Za-z0-9])|$)"
DEFAULT_MASKS = [
    (_B + r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})" + _E, "UUID"),
    (_B + r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(:\d+)?)" + _E, "IP"),
    (_B + r"(0x[0-9a-fA-F]+)" + _E, "HEX"),
    (_B + r"([0-9a-fA-F]{16,})" + _E, "HEX"),
    (_B + r"([\-\+]?\d+(\.\d+)?)" + _E, "NUM"),
]


def trace_series(traces, inject_time, bin_seconds=BIN_SECONDS):
    """Return ``(tracets_err, tracets_lat)`` for a ``traces.csv`` table and the
    failure start ``inject_time`` (unix seconds)."""
    traces = traces.copy()
    op = traces["methodName"].where(traces["methodName"].notna(), traces["operationName"])
    traces["operation"] = traces["serviceName"].astype(str) + "_" + op.astype(str)

    # the baseline: spans that ended before the failure started (microseconds)
    ended = traces["startTime"] + traces["duration"]
    normal = traces[ended < inject_time * 1_000_000]
    stats = normal.groupby("operation")["duration"].agg(["mean", "std"])
    threshold = traces["operation"].map(stats["mean"] + 3 * stats["std"])
    # operations never seen before the failure, or seen once, have no baseline
    traces["slow"] = (traces["duration"] > threshold).astype(int)
    traces["error"] = (traces["statusCode"].fillna(0) != 0).astype(int)

    start_s = traces["startTime"] // 1_000_000
    traces["bin"] = start_s - start_s % bin_seconds
    bins = range(int(traces["bin"].min()), int(traces["bin"].max()) + 1, bin_seconds)
    operations = list(dict.fromkeys(traces["operation"]))  # in order of first span
    out = []
    for column in ("error", "slow"):
        table = traces.pivot_table(index="bin", columns="operation", values=column, aggfunc="sum")
        table = table.reindex(index=bins, columns=operations, fill_value=0).fillna(0).astype(int)
        table.index.name = "time"
        table.columns.name = None
        out.append(table.reset_index())
    return out[0], out[1]


def log_series(logs, drain_config=None, bin_seconds=BIN_SECONDS):
    """Mine templates from a ``logs.csv`` table and return ``(logts,
    cluster_info, logs)``, where ``logs`` gains ``cluster_id`` and
    ``log_template``. ``drain_config`` is a Drain3 ``.ini`` file (optional)."""
    try:
        from drain3 import TemplateMiner
        from drain3.masking import MaskingInstruction
        from drain3.template_miner_config import TemplateMinerConfig
    except ImportError as e:
        raise ImportError(
            "building logts.csv needs drain3: pip install RCAEval[convert]"
        ) from e

    # always pass a config: without one Drain3 looks for ./drain3.ini and
    # prints a warning when it is missing
    config = TemplateMinerConfig()
    if drain_config is not None:
        if not os.path.exists(drain_config):
            raise FileNotFoundError(drain_config)
        config.load(drain_config)
    else:
        config.masking_instructions = [MaskingInstruction(p, m) for p, m in DEFAULT_MASKS]
    miner = TemplateMiner(config=config)

    logs = logs.copy()
    ids = []
    for container, message, error in zip(logs["container_name"], logs["message"], logs.get("error", [None] * len(logs))):
        # prefix the container, as for the released datasets, so templates
        # stay within one service
        text = f"{container}_{message}"
        if isinstance(error, str) and error:
            text += f"_{error}"
        ids.append(miner.add_log_message(text)["cluster_id"])
    # a cluster's template generalises as lines arrive, so take the final one
    templates = {c.cluster_id: c.get_template() for c in miner.drain.clusters}
    logs["cluster_id"] = ids
    logs["log_template"] = logs["cluster_id"].map(templates)

    start_s = logs["timestamp"] // 1_000_000_000
    logs_bin = start_s - start_s % bin_seconds
    column = logs["container_name"].astype(str) + "_" + logs["cluster_id"].astype(str)
    counts = pd.crosstab(logs_bin, column)
    bins = range(int(logs_bin.min()), int(logs_bin.max()) + 1, bin_seconds)
    columns = list(dict.fromkeys(column))
    logts = counts.reindex(index=bins, columns=columns, fill_value=0).astype(int)
    logts.index.name = "time"
    logts.columns.name = None

    cluster_info = {}
    for cid, container in dict.fromkeys(zip(logs["cluster_id"], logs["container_name"])):
        entry = cluster_info.setdefault(str(cid), {"template": templates[cid], "container": []})
        entry["container"].append(container)
    return logts.reset_index(), cluster_info, logs


def derive_case(case_dir, inject_time=None, drain_config=None):
    """Write the derived files for the case in ``case_dir`` from its
    ``logs.csv`` and ``traces.csv`` (each optional). ``inject_time`` defaults
    to the case's ``inject_time.txt``. Returns the names of the files written."""
    written = []
    traces_path = os.path.join(case_dir, "traces.csv")
    logs_path = os.path.join(case_dir, "logs.csv")
    if not os.path.exists(traces_path) and not os.path.exists(logs_path):
        raise FileNotFoundError(f"{case_dir} has neither logs.csv nor traces.csv")

    if os.path.exists(traces_path):
        if inject_time is None:
            inject_time = read_inject_time(case_dir)
        err, lat = trace_series(pd.read_csv(traces_path), inject_time)
        err.to_csv(os.path.join(case_dir, "tracets_err.csv"), index=False)
        lat.to_csv(os.path.join(case_dir, "tracets_lat.csv"), index=False)
        written += ["tracets_err.csv", "tracets_lat.csv"]

    if os.path.exists(logs_path):
        logts, cluster_info, logs = log_series(pd.read_csv(logs_path), drain_config=drain_config)
        logts.to_csv(os.path.join(case_dir, "logts.csv"), index=False)
        with open(os.path.join(case_dir, "cluster_info.json"), "w") as f:
            json.dump(cluster_info, f, indent=4)
        logs.to_csv(logs_path, index=False)
        written += ["logts.csv", "cluster_info.json", "logs.csv"]
    return written


def read_inject_time(case_dir):
    path = os.path.join(case_dir, "inject_time.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found: the trace latency baseline needs the failure start time"
        )
    with open(path) as f:
        return int(f.readline().strip())
