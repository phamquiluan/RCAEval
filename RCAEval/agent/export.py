"""Export a dataset as a blind task view for LLM agents.

A case directory in the raw datasets is named after its answer
(`<service>_<fault>/<n>`), so an agent that can list files sees the root cause.
The export writes each case under an anonymised id with only its telemetry and
a task.json, and puts the labels in a separate answers file next to (never
inside) the task directory.
"""
import csv
import glob
import hashlib
import json
import os
import shutil
from os.path import basename, dirname, exists, join, relpath

from RCAEval.benchmark.datasets import DATASET_MAP, download_dataset
from RCAEval.benchmark.scoring import fault_to_indicator, indicator_ranks, to_service
from RCAEval.classes.graph import Node

SCHEMA_VERSION = 1

# datasets whose cases follow the `<service>_<fault>/<n>/` metrics layout;
# EventADL (event logs) and TORAI (local-only data) are not exported yet
EXPORTABLE = [
    "re1-ob", "re1-ss", "re1-tt", "re2-ob", "re2-ss", "re2-tt", "re3-ob", "re3-ss", "re3-tt",
    "online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket",
]

# the length main.py uses by default (minutes): methods see half of it on each
# side of the inject time, and metrics are sampled once per second
STANDARD_LENGTH = 20

# exported name -> source name, in addition to the primary metrics file; files
# not listed here (metrics_postprocess.log, pod-node-*.csv, inject_time.txt)
# are not exported. cluster_info.json holds the log templates behind logts.csv.
EXTRA_FILES = {
    "metrics_raw.csv": "metrics.csv",
    "logs.csv": "logs.csv",
    "logts.csv": "logts.csv",
    "traces.csv": "traces.csv",
    "tracets_err.csv": "tracets_err.csv",
    "tracets_lat.csv": "tracets_lat.csv",
    "log_templates.json": "cluster_info.json",
}

ANSWER_FORMAT = (
    "Write one JSON line per case to a predictions file: "
    '{"case_id": "<case_id>", "ranks": ["<candidate>", ...]}, most likely root cause first. '
    "A candidate is a service from candidate_services (scored at service level) or a "
    "<service>_<indicator> column from candidate_indicators (scored at service and indicator "
    "level). Optional fields: reasoning, model, and usage "
    '{"input_tokens", "output_tokens", "cost_usd", "wall_time_s", "tool_calls"}.'
)


def system_of(dataset):
    if dataset.endswith("-ob") or dataset == "online-boutique":
        return "Online Boutique"
    if dataset.endswith("-ss") or dataset.startswith("sock-shop"):
        return "Sock Shop"
    return "Train Ticket"


def find_metric_files(dataset_dir):
    """The per-case primary metrics files, chosen as main.py chooses them."""
    paths = glob.glob(join(dataset_dir, "**/data.csv"), recursive=True)
    if not paths:
        paths = glob.glob(join(dataset_dir, "**/simple_metrics.csv"), recursive=True)
    return sorted(paths)


def candidate_indicators(metrics_path):
    """The metrics header as main.py presents it to methods: no time columns,
    p50 latency dropped, p90 latency renamed to plain latency."""
    with open(metrics_path, newline="") as f:
        header = next(csv.reader(f))
    columns = []
    for c in header:
        if c in ("time", "time.1") or c.endswith("_latency-50"):
            continue
        if c.endswith("_latency-90"):
            c = c[: -len("-90")]
        if c not in columns:
            columns.append(c)
    return columns


def case_id_order(dataset, sources):
    """Sources in a fixed order that does not follow the answer-bearing names."""
    return sorted(sources, key=lambda s: hashlib.sha256(f"{dataset}:{s}".encode()).hexdigest())


def place(src, dst, copy):
    """Hardlink src to dst (copy when asked, or across filesystems). Never a
    symlink: its target would reveal the answer-bearing source path."""
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copyfile(src, dst)


def export(dataset, out_dir="agent_tasks", copy=False, force=False, download=True, data_root=None):
    """Write <out_dir>/<dataset>/ (the task view) and <out_dir>/<dataset>.answers.jsonl.

    data_root overrides the dataset's directory (default: DATASET_MAP, relative
    to the working directory). Returns (task_dir, answers_path).
    """
    if dataset not in EXPORTABLE:
        raise ValueError(f"cannot export {dataset!r}; choose one of {', '.join(EXPORTABLE)}")
    if data_root is None:
        if download:
            download_dataset(dataset)
        data_root = DATASET_MAP[dataset]

    metric_files = find_metric_files(data_root)
    if not metric_files:
        raise FileNotFoundError(f"no cases (data.csv or simple_metrics.csv) under {data_root}")
    by_source = {relpath(dirname(p), data_root).replace(os.sep, "/"): p for p in metric_files}

    task_dir = join(out_dir, dataset)
    answers_path = join(out_dir, f"{dataset}.answers.jsonl")
    if exists(task_dir) or exists(answers_path):
        if not force:
            raise FileExistsError(f"{task_dir} or {answers_path} already exists (use --force to replace)")
        shutil.rmtree(task_dir, ignore_errors=True)
        if exists(answers_path):
            os.remove(answers_path)
    os.makedirs(join(task_dir, "cases"))

    case_ids = []
    answers = []
    for i, source in enumerate(case_id_order(dataset, by_source), start=1):
        metrics_path = by_source[source]
        src_dir = dirname(metrics_path)
        case_id = f"{dataset}-{i:04d}"
        case_dir = join(task_dir, "cases", case_id)
        os.makedirs(case_dir)

        service, fault = basename(dirname(src_dir)).split("_")
        with open(join(src_dir, "inject_time.txt")) as f:
            inject_time = int(f.readlines()[0].strip())

        place(metrics_path, join(case_dir, "metrics.csv"), copy)
        files = {"metrics": "metrics.csv"}
        for name, src_name in EXTRA_FILES.items():
            if exists(join(src_dir, src_name)):
                place(join(src_dir, src_name), join(case_dir, name), copy)
                files[os.path.splitext(name)[0]] = name

        indicators = candidate_indicators(metrics_path)
        services = sorted({to_service(c) for c in indicators})
        indicator = fault_to_indicator(fault, dataset)
        in_candidates = service in services
        indicator_in_candidates = indicator is not None and Node(service, indicator) in indicator_ranks(indicators)
        if not in_candidates:
            print(f"warning: {case_id}: the root-cause service is not among the candidate services")

        half = STANDARD_LENGTH * 60 // 2
        task = {
            "schema_version": SCHEMA_VERSION,
            "case_id": case_id,
            "dataset": dataset,
            "system": system_of(dataset),
            "inject_time": inject_time,
            "standard_window": {"start": inject_time - half, "end": inject_time + half},
            "files": files,
            "candidate_services": services,
            "candidate_indicators": indicators,
            "answer_format": ANSWER_FORMAT,
        }
        with open(join(case_dir, "task.json"), "w") as f:
            json.dump(task, f, indent=2)

        case_ids.append(case_id)
        answers.append({
            "case_id": case_id,
            "dataset": dataset,
            "source": source,
            "service": service,
            "fault": fault,
            "indicator": indicator,
            "answer_in_candidates": in_candidates,
            "indicator_in_candidates": indicator_in_candidates,
            "n_candidate_services": len(services),
            "n_candidate_indicators": len(indicators),
        })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "system": system_of(dataset),
        "n_cases": len(case_ids),
        "case_ids": case_ids,
        "answer_format": ANSWER_FORMAT,
    }
    with open(join(task_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(answers_path, "w") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")
    return task_dir, answers_path
