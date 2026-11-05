"""Pull metrics from Prometheus into ``metrics.csv``.

Each query is named, e.g. ``"cpu": 'sum(rate(container_cpu_usage_seconds_total[1m])) by (container)'``,
and every series it returns becomes the column ``{service}_{name}``, where the
service is the value of one label of the series (``container`` above). This is
how the RE2/RE3 metrics were collected.
"""
import json

import pandas as pd

# Prometheus rejects a range query that would return more points per series
PROMETHEUS_MAX_POINTS = 11_000


def load_queries(path):
    """Read a queries file: a JSON object mapping each metric name to a PromQL
    query, or to ``{"query": ..., "label": ...}`` naming the label that holds
    the service."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a JSON object of name -> query")
    queries = {}
    for name, spec in raw.items():
        if isinstance(spec, str):
            spec = {"query": spec}
        if not isinstance(spec, dict) or not isinstance(spec.get("query"), str):
            raise ValueError(f"{path}: {name!r} needs a PromQL string or {{\"query\": ...}}")
        if not name:
            raise ValueError(f"{path}: empty metric name")
        queries[name] = {"query": spec["query"], "label": spec.get("label")}
    return queries


def query_range(url, query, start, end, step=1, session=None, timeout=60):
    """Run one range query over [start, end] (unix seconds) and return its
    ``data.result`` list, split into several requests when the range is longer
    than Prometheus allows."""
    if session is None:
        import requests

        session = requests.Session()
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")

    series = {}
    span = step * (PROMETHEUS_MAX_POINTS - 1)
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + span, end)
        response = session.get(
            url.rstrip("/") + "/api/v1/query_range",
            params={"query": query, "start": chunk_start, "end": chunk_end, "step": step},
            timeout=timeout,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {body.get('error', body)} ({query})")
        if body["data"]["resultType"] != "matrix":
            raise ValueError(f"expected a range vector (matrix), got {body['data']['resultType']}: {query}")
        for s in body["data"]["result"]:
            key = tuple(sorted(s["metric"].items()))
            series.setdefault(key, {"metric": s["metric"], "values": []})["values"].extend(s["values"])
        chunk_start = chunk_end + step
    return list(series.values())


def metrics_frame(results):
    """Build the ``metrics.csv`` table from ``{name: (result, label)}``, where
    ``result`` is a query's ``data.result`` list and ``label`` the label that
    names the service (None: the series' only label)."""
    columns = {}
    for name, (result, label) in results.items():
        for s in result:
            service = _service(s["metric"], label, name)
            column = f"{service}_{name}"
            if column in columns:
                raise ValueError(
                    f"two series of {name!r} map to {column!r}; aggregate the query "
                    f"by the service label (e.g. `sum(...) by ({label or 'container'})`)"
                )
            columns[column] = pd.Series(
                {int(float(t)): float(v) for t, v in s["values"]}, dtype=float
            )
    if not columns:
        raise ValueError("the queries returned no series")
    frame = pd.DataFrame(columns).sort_index()
    frame.index.name = "time"
    return frame.reset_index()


def pull_metrics(url, queries, start, end, step=1, session=None):
    """Run every query in ``queries`` (as returned by `load_queries`) and return
    the ``metrics.csv`` table."""
    results = {
        name: (query_range(url, q["query"], start, end, step, session=session), q["label"])
        for name, q in queries.items()
    }
    return metrics_frame(results)


def _service(metric, label, name):
    labels = {k: v for k, v in metric.items() if k != "__name__"}
    if label is not None:
        if label not in labels:
            raise ValueError(f"a series of {name!r} has no {label!r} label: {metric}")
        service = labels[label]
    elif len(labels) == 1:
        service = next(iter(labels.values()))
    else:
        raise ValueError(
            f"cannot tell which label of {name!r} names the service ({sorted(labels) or 'none'}); "
            f"aggregate the query by one label or set \"label\" in the queries file"
        )
    check_service(service)
    return service


def check_service(service):
    """Methods recover the service from a column as the text before the first
    "_", so a service name must be non-empty and contain none."""
    if not service or "_" in service:
        raise ValueError(f"service name {service!r} must be non-empty and contain no '_'")
