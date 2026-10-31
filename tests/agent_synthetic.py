"""A small synthetic dataset with every telemetry type, for the agent tests.

The planted fault: cartservice CPU rises at the inject time, its spans slow
down and fail, and checkoutservice (its caller) logs connection errors.
"""
import json
import os
from os.path import join

from RCAEval.agent.export import export

START = 1_700_000_000  # realistic epoch seconds, so ms/us/ns timestamps are told apart
INJECT, END = START + 100, START + 200


def _write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def write_case(d, root_service="cartservice", with_logs=True, with_traces=True):
    os.makedirs(d)
    with open(join(d, "inject_time.txt"), "w") as f:
        f.write(f"{INJECT}\n")

    header = ["time", "cartservice_cpu", "cartservice_mem", "checkoutservice_cpu",
              "checkoutservice_latency-50", "checkoutservice_latency-90", "frontend_cpu"]
    rows = []
    for t in range(START, END):
        noise = (t % 3) * 0.1
        jump = 9 if t >= INJECT else 0
        rows.append([t, 1 + noise + (jump if root_service == "cartservice" else 0), 5 + noise,
                     1 + noise + (jump if root_service == "checkoutservice" else 0), 10, 20 + noise, 1 + noise])
    _write_csv(join(d, "data.csv"), header, rows)

    if with_logs:
        rows = []
        for t in range(START, END, 5):
            rows.append(["00:00", t * 10**9, "frontend", "INFO served request 42"])
            if t >= INJECT:
                rows.append(["00:00", t * 10**9, "checkoutservice", f"ERROR connection refused to cart id={t}"])
        rows.append(["00:00", START * 10**9, "checkoutservice", "ERROR connection refused to cart id=7"])
        _write_csv(join(d, "logs.csv"), ["time", "timestamp", "container_name", "message"], rows)
        _write_csv(join(d, "logts.csv"), ["time", "checkoutservice_1", "frontend_2"],
                   [[t, (12 if t >= INJECT else 0) + t % 2, 3 + t % 2] for t in range(START, END, 15)])
        with open(join(d, "cluster_info.json"), "w") as f:
            json.dump({"1": {"template": "ERROR connection refused to cart id=<:NUM:>", "container": ["checkoutservice"]},
                       "2": {"template": "INFO served request <:NUM:>", "container": ["frontend"]}}, f)

    if with_traces:
        rows = []
        for i, t in enumerate(range(START, END, 2)):
            after = t >= INJECT
            trace = f"t{i}"
            rows.append(["00:00", trace, f"a{i}", "frontend", "", "GET /", t * 1000, t * 10**6, 5000, 200, ""])
            rows.append(["00:00", trace, f"b{i}", "checkoutservice", "", "PlaceOrder", t * 1000, t * 10**6,
                         3000, 200, f"a{i}"])
            rows.append(["00:00", trace, f"c{i}", "cartservice", "", "GetCart", t * 1000, t * 10**6,
                         90000 if after else 1000 + (i % 3) * 10, 500 if after else 200, f"b{i}"])
        _write_csv(join(d, "traces.csv"), ["time", "traceID", "spanID", "serviceName", "methodName", "operationName",
                                           "startTimeMillis", "startTime", "duration", "statusCode", "parentSpanID"],
                   rows)
        for name, base, high in (("tracets_lat.csv", 1, 90), ("tracets_err.csv", 0, 30)):
            _write_csv(join(d, name), ["time", "cartservice_GetCart", "frontend_GET"],
                       [[t, (high if t >= INJECT else base) + t % 2, 1 + t % 2] for t in range(START, END, 15)])


def make_dataset(root, faults=(("cartservice", "cpu"), ("checkoutservice", "cpu")), n_cases=1, **kwargs):
    for service, fault in faults:
        for n in range(1, n_cases + 1):
            write_case(join(root, f"{service}_{fault}", str(n)), root_service=service, **kwargs)
    return root


def exported_case(tmp_path, dataset="re2-ob", **kwargs):
    """Export a one-case synthetic dataset; returns (case_dir, answer)."""
    data_root = make_dataset(str(tmp_path / "raw"), faults=(("cartservice", "cpu"),), **kwargs)
    task_dir, answers_path = export(dataset, out_dir=str(tmp_path / "out"), data_root=data_root)
    answer = json.loads(open(answers_path).readline())
    return join(task_dir, "cases", answer["case_id"]), answer


def exported_tasks(tmp_path, dataset="re2-ob", **kwargs):
    """Export a two-case synthetic dataset; returns (task_dir, answers_path)."""
    data_root = make_dataset(str(tmp_path / "raw"), **kwargs)
    return export(dataset, out_dir=str(tmp_path / "out"), data_root=data_root)
