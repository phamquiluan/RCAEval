"""Convert a log export (JSON Lines, JSON or CSV) into ``logs.csv``.

Log stacks store the same few fields under different names, so the fields are
mapped to RCAEval's columns by path. A path is a key, or keys joined by "."
for nested JSON; a key that itself contains dots, as Elasticsearch documents
often have, is matched whole first. For example:

- Loki (``logcli query --output=jsonl``): ``timestamp=timestamp``,
  ``container=labels.container``, ``message=line``
- Elasticsearch documents: ``timestamp=_source.@timestamp``,
  ``container=_source.kubernetes.container.name``, ``message=_source.message``
- Fluentd file output: ``timestamp=time``, ``container=kubernetes.container_name``,
  ``message=log``
"""
import json
import warnings

import pandas as pd

from RCAEval.convert.common import check_service, to_unix_ns

LOG_COLUMNS = ["time", "timestamp", "container_name", "message", "level", "req_path", "error"]

# RCAEval field -> default path, i.e. the column name in a released logs.csv
DEFAULT_FIELDS = {
    "timestamp": "timestamp",
    "container": "container_name",
    "message": "message",
    "level": "level",
    "req_path": "req_path",
    "error": "error",
}
REQUIRED_FIELDS = ("timestamp", "container", "message")


def read_logs(paths, fields=None):
    """Read log exports and return the ``logs.csv`` table, sorted by time.

    ``fields`` maps RCAEval fields (`DEFAULT_FIELDS`) to paths in the input;
    unmapped fields keep their default path. Records missing a timestamp,
    container or message are dropped with a warning.
    """
    unknown = set(fields or {}) - set(DEFAULT_FIELDS)
    if unknown:
        raise ValueError(f"unknown log fields {sorted(unknown)}; expected {sorted(DEFAULT_FIELDS)}")
    fields = {**DEFAULT_FIELDS, **(fields or {})}

    records = []
    for path in [paths] if isinstance(paths, str) else paths:
        records.extend(_records(path))
    if not records:
        raise ValueError("no log records found")

    rows = {field: [_lookup(r, fields[field]) for r in records] for field in DEFAULT_FIELDS}
    frame = pd.DataFrame(rows)
    complete = frame[list(REQUIRED_FIELDS)].notna().all(axis=1)
    if not complete.all():
        missing = [f"{f} ({fields[f]!r})" for f in REQUIRED_FIELDS if frame[f].isna().any()]
        warnings.warn(f"dropped {(~complete).sum()} of {len(frame)} log records without {', '.join(missing)}")
        frame = frame[complete]
    if frame.empty:
        raise ValueError(
            "no log record has all of " + ", ".join(f"{f} ({fields[f]!r})" for f in REQUIRED_FIELDS)
        )

    for service in frame["container"].unique():
        check_service(service)
    frame["timestamp"] = to_unix_ns(frame["timestamp"]).values
    frame["time"] = frame["timestamp"] // 1_000_000_000
    frame = frame.rename(columns={"container": "container_name"})
    # a structured message (an object) is kept as its JSON text
    frame["message"] = frame["message"].map(lambda m: m if isinstance(m, str) else json.dumps(m))
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return frame[LOG_COLUMNS]


def _records(path):
    if str(path).endswith(".csv"):
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        return [{k: (v if v != "" else None) for k, v in row.items()} for row in frame.to_dict("records")]
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = [data]
    if not all(isinstance(r, dict) for r in data):
        raise ValueError(f"{path}: expected JSON objects, one per log record")
    return data


def _lookup(record, path):
    if not isinstance(record, dict):
        return None
    if path in record:
        return record[path]
    # try the longest dotted prefix that is a key, then descend
    parts = path.split(".")
    for i in range(len(parts) - 1, 0, -1):
        key = ".".join(parts[:i])
        if key in record:
            return _lookup(record[key], ".".join(parts[i:]))
    return None
