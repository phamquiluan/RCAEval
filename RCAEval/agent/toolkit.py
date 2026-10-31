"""Tools an agent uses to investigate one case of a blind task view.

CaseToolkit reads a case directory written by `rcaeval-agent export` and
answers small questions about it: which metrics moved, what the logs and
traces say, how services call each other. Every answer is a JSON-serialisable
dict capped at MAX_CHARS, so it fits in an LLM context, and every call is
logged in `calls` for the trajectory.

The same TOOLS registry backs the reference LLM agent (RCAEval.agent.react)
and the MCP server (RCAEval.agent.mcp_server).
"""
import difflib
import json
import re
import time
from os.path import exists, join

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from RCAEval.benchmark.scoring import to_service

MAX_CHARS = 8000

TOOLS = [
    {
        "name": "describe_case",
        "description": "Overview of the case: system, inject time, standard window, available telemetry and candidate services. Call this first.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "list_metrics",
        "description": "Metric names. Without a service, the indicator suffixes per service; with a service, that service's full metric names.",
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string", "description": "Only this service's metrics."},
        }},
    },
    {
        "name": "get_metric",
        "description": "One metric over time, averaged into at most `points` buckets, with its mean and max before and after the inject time.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "Metric name, e.g. cartservice_cpu."},
            "start": {"type": "integer", "description": "Unix seconds (default: window start)."},
            "end": {"type": "integer", "description": "Unix seconds (default: window end)."},
            "points": {"type": "integer", "description": "Number of buckets (default 40, max 120)."},
        }, "required": ["name"]},
    },
    {
        "name": "rank_anomalies",
        "description": (
            "Series ranked by how far they move after the inject time (BARO-style robust z-score). "
            "modality: metrics, logs (log template counts), trace_latency or trace_errors."
        ),
        "parameters": {"type": "object", "properties": {
            "modality": {"type": "string", "enum": ["metrics", "logs", "trace_latency", "trace_errors"]},
            "top_k": {"type": "integer", "description": "Number of series to return (default 20, max 100)."},
        }},
    },
    {
        "name": "search_logs",
        "description": (
            "Log lines matching a case-insensitive regex, optionally for one service, grouped by message "
            "pattern with counts before and after the inject time, largest increase first."
        ),
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"},
            "pattern": {"type": "string", "description": "Regex, e.g. 'error|exception|timeout'."},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "limit": {"type": "integer", "description": "Number of groups (default 20, max 50)."},
        }},
    },
    {
        "name": "trace_summary",
        "description": (
            "Per service and operation: requests, errors and p50/p90 latency (ms) before and after the "
            "inject time, largest change first."
        ),
        "parameters": {"type": "object", "properties": {
            "service": {"type": "string"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "top_k": {"type": "integer", "description": "Number of rows (default 20, max 100)."},
        }},
    },
    {
        "name": "call_graph",
        "description": "Which service calls which, with call counts, from the traces.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "submit",
        "description": (
            "Submit the final answer: candidates ranked most likely root cause first, each a service from "
            "candidate_services or a <service>_<indicator> metric name. Ends the investigation."
        ),
        "parameters": {"type": "object", "properties": {
            "ranks": {"type": "array", "items": {"type": "string"}},
            "reasoning": {"type": "string", "description": "One or two sentences on the evidence."},
        }, "required": ["ranks"]},
    },
]
TOOL_NAMES = [t["name"] for t in TOOLS]

_MASKS = [
    (re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I), "<id>"),
    (re.compile(r"\b[0-9a-f]{12,}\b", re.I), "<id>"),
    (re.compile(r"\d+(?:\.\d+)?"), "<n>"),
    (re.compile(r"\s+"), " "),
]


def mask_message(message):
    """A log line with ids and numbers masked, so lines group by pattern."""
    for pattern, repl in _MASKS:
        message = pattern.sub(repl, message)
    return message.strip()[:200]


def to_seconds(values):
    """Unix timestamps in s, ms, us or ns, as float seconds."""
    values = pd.to_numeric(values, errors="coerce")
    sample = values.dropna()
    if sample.empty:
        return values
    magnitude = sample.iloc[0]
    for threshold, scale in ((1e17, 1e9), (1e14, 1e6), (1e11, 1e3)):
        if magnitude > threshold:
            return values / scale
    return values


def first_column(columns, names):
    return next((c for c in names if c in columns), None)


def cap(result, limit=MAX_CHARS):
    """Shorten the longest list in result until it serialises within limit."""
    while len(json.dumps(result, default=str)) > limit:
        lists = [(k, v) for k, v in result.items() if isinstance(v, list) and len(v) > 1]
        if not lists:
            break
        key, value = max(lists, key=lambda kv: len(json.dumps(kv[1], default=str)))
        result[key] = value[: len(value) // 2]
        result["truncated"] = True
    return result


def clamp(value, default, lo, hi):
    if value is None:
        return default
    return max(lo, min(hi, int(value)))


class ToolError(Exception):
    """A problem with the agent's request, returned to it as {"error": ...}."""


class CaseToolkit:
    """The tools over one exported case directory."""

    def __init__(self, case_dir):
        self.case_dir = case_dir
        with open(join(case_dir, "task.json")) as f:
            self.task = json.load(f)
        self.inject_time = self.task["inject_time"]
        self.window = self.task["standard_window"]
        self.calls = []
        self.submission = None
        self._cache = {}

    # --- dispatch ---

    def call(self, name, args=None):
        """Run tool `name` with `args`; returns its result dict, or {"error": ...}."""
        args = dict(args or {})
        started = time.time()
        try:
            if name not in TOOL_NAMES:
                raise ToolError(f"unknown tool {name!r}; tools: {', '.join(TOOL_NAMES)}")
            allowed = next(t for t in TOOLS if t["name"] == name)["parameters"]["properties"]
            unknown = sorted(set(args) - set(allowed))
            if unknown:
                raise ToolError(f"unknown argument(s) for {name}: {', '.join(unknown)}")
            result = cap(getattr(self, name)(**args))
            error = None
        except ToolError as e:
            result, error = {"error": str(e)}, str(e)
        except Exception as e:  # a bug or unreadable data, not the agent's fault
            result, error = {"error": f"{type(e).__name__}: {e}"}, f"{type(e).__name__}: {e}"
        self.calls.append({
            "seq": len(self.calls) + 1,
            "tool": name,
            "args": args,
            "ok": error is None,
            "error": error,
            "result_chars": len(json.dumps(result, default=str)),
            "ms": round((time.time() - started) * 1000),
        })
        return result

    # --- data ---

    def _path(self, modality):
        name = self.task["files"].get(modality)
        return join(self.case_dir, name) if name else None

    def _has(self, modality):
        path = self._path(modality)
        return path is not None and exists(path)

    def _require(self, modality, what):
        if not self._has(modality):
            raise ToolError(f"this case has no {what}")

    def _window(self, start, end):
        start = self.window["start"] if start is None else int(float(start))
        end = self.window["end"] if end is None else int(float(end))
        if end <= start:
            raise ToolError("end must be after start")
        return start, end

    def metrics(self):
        """metrics.csv prepared as main.py prepares it for methods."""
        if "metrics" not in self._cache:
            data = pd.read_csv(self._path("metrics"))
            data = data.drop(columns=["time.1"], errors="ignore")
            data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
            data = data.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            data = data.rename(columns=lambda c: c.replace("_latency-90", "_latency"))
            self._cache["metrics"] = data
        return self._cache["metrics"]

    def standard_metrics(self):
        """The rows main.py gives methods: the standard window around the inject time."""
        data = self.metrics()
        half = self.inject_time - self.window["start"]
        return pd.concat([data[data["time"] < self.inject_time].tail(half),
                          data[data["time"] >= self.inject_time].head(half)], ignore_index=True)

    def series(self, modality):
        """logts / tracets_lat / tracets_err, with NaN filled as multi-source BARO does."""
        if modality not in self._cache:
            data = pd.read_csv(self._path(modality))
            self._cache[modality] = data.ffill().fillna(0)
        return self._cache[modality]

    def logs(self):
        if "logs" not in self._cache:
            header = list(pd.read_csv(self._path("logs"), nrows=0).columns)
            ts = first_column(header, ["timestamp", "time"])
            svc = first_column(header, ["container_name", "service", "container"])
            msg = first_column(header, ["message", "log", "msg"])
            if ts is None or msg is None:
                raise ToolError("the logs have no timestamp or message column")
            usecols = [c for c in (ts, svc, msg) if c]
            data = pd.read_csv(self._path("logs"), usecols=usecols, dtype=str, keep_default_na=False)
            out = pd.DataFrame({
                "t": to_seconds(data[ts]),
                "service": data[svc] if svc else "",
                "message": data[msg],
            })
            self._cache["logs"] = out.dropna(subset=["t"])
        return self._cache["logs"]

    def traces(self):
        if "traces" not in self._cache:
            data = pd.read_csv(self._path("traces"), dtype=str, keep_default_na=False)
            start = first_column(data.columns, ["startTime", "startTimeMillis", "timestamp"])
            out = pd.DataFrame({
                "t": to_seconds(data[start]),
                "service": data.get("serviceName", ""),
                "operation": data.get("operationName", ""),
                "span": data.get("spanID", ""),
                "parent": data.get("parentSpanID", ""),
                # Jaeger durations are microseconds
                "ms": pd.to_numeric(data.get("duration"), errors="coerce") / 1000.0,
                "status": pd.to_numeric(data.get("statusCode"), errors="coerce"),
            })
            self._cache["traces"] = out.dropna(subset=["t"])
        return self._cache["traces"]

    def log_template(self, column):
        """The template text behind a logts column such as "cartservice_12"."""
        if "log_templates" not in self._cache:
            path = self._path("log_templates")
            self._cache["log_templates"] = json.load(open(path)) if path and exists(path) else {}
        key = column.rsplit("_", 1)[-1]
        entry = self._cache["log_templates"].get(key, {})
        return entry.get("template", "")[:200] if isinstance(entry, dict) else ""

    # --- tools ---

    def describe_case(self):
        modalities = {
            "metrics": "1 s samples; use list_metrics, get_metric, rank_anomalies(metrics)",
            "logs": "raw log lines; use search_logs",
            "logts": "log template counts per 15 s; use rank_anomalies(logs)",
            "traces": "spans; use trace_summary, call_graph",
            "tracets_lat": "latency per operation per 15 s; use rank_anomalies(trace_latency)",
            "tracets_err": "errors per operation per 15 s; use rank_anomalies(trace_errors)",
        }
        return {
            "case_id": self.task["case_id"],
            "system": self.task["system"],
            "inject_time": self.inject_time,
            "standard_window": self.window,
            "telemetry": {k: v for k, v in modalities.items() if self._has(k)},
            "candidate_services": self.task["candidate_services"],
            "n_candidate_metrics": len(self.task["candidate_indicators"]),
        }

    def list_metrics(self, service=None):
        names = [c for c in self.metrics().columns if c != "time"]
        if service:
            matched = [c for c in names if to_service(c) == service.strip().lower()]
            if not matched:
                raise ToolError(f"no metrics for service {service!r}; see describe_case for services")
            return {"service": service, "metrics": matched}
        grouped = {}
        for c in names:
            s, _, indicator = c.partition("_")
            grouped.setdefault(s, []).append(indicator)
        return {"metrics_by_service": grouped}

    def get_metric(self, name, start=None, end=None, points=None):
        data = self.metrics()
        if name not in data.columns or name == "time":
            close = difflib.get_close_matches(name, [c for c in data.columns if c != "time"], n=5)
            raise ToolError(f"unknown metric {name!r}" + (f"; did you mean {', '.join(close)}?" if close else ""))
        start, end = self._window(start, end)
        points = clamp(points, 40, 1, 120)
        rows = data[(data["time"] >= start) & (data["time"] < end)][["time", name]]
        if rows.empty:
            raise ToolError("no samples in that time range")
        buckets = [rows.iloc[idx] for idx in np.array_split(np.arange(len(rows)), min(points, len(rows))) if len(idx)]
        before = rows[rows["time"] < self.inject_time][name]
        after = rows[rows["time"] >= self.inject_time][name]

        def stats(s):
            return {"mean": round(float(s.mean()), 4), "max": round(float(s.max()), 4)} if len(s) else None

        return {
            "name": name,
            "series": [[int(b["time"].iloc[0]), round(float(b[name].mean()), 4)] for b in buckets],
            "before_inject": stats(before),
            "after_inject": stats(after),
        }

    def rank_anomalies(self, modality="metrics", top_k=None):
        top_k = clamp(top_k, 20, 1, 100)
        if modality == "metrics":
            from RCAEval.e2e.baro import baro

            scores = baro(self.standard_metrics(), self.inject_time, dataset=self.task["dataset"])["scores"]
            ranked = [{"name": n, "score": round(s, 3)} for n, s in scores.items()]
        elif modality in ("logs", "trace_latency", "trace_errors"):
            source = {"logs": "logts", "trace_latency": "tracets_lat", "trace_errors": "tracets_err"}[modality]
            self._require(source, f"{modality.replace('_', ' ')} time series")
            data = self.series(source)
            data = data[(data["time"] >= self.window["start"]) & (data["time"] < self.window["end"])]
            normal, anomal = data[data["time"] < self.inject_time], data[data["time"] >= self.inject_time]
            if normal.empty or anomal.empty:
                raise ToolError("not enough samples on both sides of the inject time")
            ranked = []
            for col in data.columns:
                if col == "time" or data[col].nunique() <= 1:
                    continue
                scaler = RobustScaler().fit(normal[[col]].to_numpy())
                score = float(np.max(scaler.transform(anomal[[col]].to_numpy())))
                entry = {"name": col, "score": round(score, 3)}
                if modality == "logs":
                    entry["template"] = self.log_template(col)
                ranked.append(entry)
            ranked.sort(key=lambda r: r["score"], reverse=True)
        else:
            raise ToolError("modality must be metrics, logs, trace_latency or trace_errors")
        return {"modality": modality, "ranked": ranked[:top_k], "n_series": len(ranked)}

    def search_logs(self, service=None, pattern=None, start=None, end=None, limit=None):
        self._require("logs", "logs")
        start, end = self._window(start, end)
        limit = clamp(limit, 20, 1, 50)
        logs = self.logs()
        rows = logs[(logs["t"] >= start) & (logs["t"] < end)]
        if service:
            rows = rows[rows["service"].str.lower() == service.strip().lower()]
        if pattern:
            try:
                rows = rows[rows["message"].str.contains(pattern, case=False, regex=True)]
            except re.error as e:
                raise ToolError(f"invalid regex: {e}")
        if rows.empty:
            return {"matches": 0, "groups": []}
        rows = rows.assign(pattern=rows["message"].map(mask_message), after=rows["t"] >= self.inject_time)
        groups = []
        for (svc, pat), g in rows.groupby(["service", "pattern"], sort=False):
            n_after = int(g["after"].sum())
            groups.append({
                "service": svc,
                "pattern": pat,
                "before": len(g) - n_after,
                "after": n_after,
                "example": g["message"].iloc[-1][:300],
            })
        groups.sort(key=lambda r: (r["after"] - r["before"], r["after"]), reverse=True)
        return {"matches": len(rows), "n_groups": len(groups), "groups": groups[:limit]}

    def trace_summary(self, service=None, start=None, end=None, top_k=None):
        self._require("traces", "traces")
        start, end = self._window(start, end)
        top_k = clamp(top_k, 20, 1, 100)
        tr = self.traces()
        rows = tr[(tr["t"] >= start) & (tr["t"] < end)]
        if service:
            rows = rows[rows["service"].str.lower() == service.strip().lower()]
        if rows.empty:
            return {"rows": []}
        has_status = rows["status"].notna().any()

        def side(g):
            if g.empty:
                return None
            out = {"requests": len(g),
                   "p50_ms": round(float(g["ms"].quantile(0.5)), 2),
                   "p90_ms": round(float(g["ms"].quantile(0.9)), 2)}
            if has_status:
                out["errors"] = int((g["status"] >= 400).sum())
            return out

        table = []
        for (svc, op), g in rows.groupby(["service", "operation"], sort=False):
            before, after = side(g[g["t"] < self.inject_time]), side(g[g["t"] >= self.inject_time])
            change = 0.0
            if before and after:
                change = after["p90_ms"] / max(before["p90_ms"], 1e-3)
                if has_status:
                    change += (after["errors"] / after["requests"] - before["errors"] / before["requests"]) * 10
            table.append({"service": svc, "operation": op, "before": before, "after": after,
                          "change": round(change, 2)})
        table.sort(key=lambda r: r["change"], reverse=True)
        note = None if has_status else "these traces carry no status codes; see rank_anomalies(trace_errors)"
        result = {"rows": table[:top_k], "n_operations": len(table)}
        if note:
            result["note"] = note
        return result

    def call_graph(self):
        self._require("traces", "traces")
        tr = self.traces()
        tr = tr[(tr["t"] >= self.window["start"]) & (tr["t"] < self.window["end"])]
        parents = tr[["span", "service"]].rename(columns={"span": "parent", "service": "caller"})
        edges = tr.merge(parents, on="parent")
        edges = edges[edges["caller"] != edges["service"]]
        counts = edges.groupby(["caller", "service"]).size().sort_values(ascending=False)
        return {"edges": [{"caller": c, "callee": s, "calls": int(n)} for (c, s), n in counts.items()]}

    def submit(self, ranks, reasoning=None):
        if not isinstance(ranks, list) or not all(isinstance(r, str) for r in ranks):
            raise ToolError("ranks must be a list of strings")
        ranks = [r.strip().lower() for r in ranks if r.strip()]
        services = set(self.task["candidate_services"])
        indicators = {c.lower() for c in self.task["candidate_indicators"]}
        invalid = [r for r in ranks if r not in indicators and r not in services]
        if invalid:
            raise ToolError(f"not in candidate_services or candidate metrics: {', '.join(invalid[:10])}")
        if not ranks:
            raise ToolError("ranks is empty")
        self.submission = {"ranks": ranks, "reasoning": reasoning}
        return {"accepted": True, "n_ranks": len(ranks)}
