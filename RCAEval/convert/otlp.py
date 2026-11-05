"""Convert OpenTelemetry traces (OTLP/JSON) into ``traces.csv``.

Jaeger, Zipkin and SkyWalking all export to, or are bridged by, the
OpenTelemetry Collector, whose file exporter writes OTLP/JSON: one
``ExportTraceServiceRequest`` per line. A single request object, or a JSON
list of them, is accepted too.

``statusCode`` follows the released datasets, which record gRPC status codes
(0 is OK): a span's ``rpc.grpc.status_code`` when it has one; otherwise, for a
span whose OpenTelemetry status is ERROR, its HTTP status code, or 2 (gRPC
UNKNOWN) when it has none; otherwise 0. A span is an error exactly when its
``statusCode`` is not 0.
"""
import json

import pandas as pd

from RCAEval.convert.common import check_service

TRACE_COLUMNS = [
    "time", "traceID", "spanID", "serviceName", "methodName", "operationName",
    "startTimeMillis", "startTime", "duration", "statusCode", "parentSpanID",
]

_STATUS_ERROR = 2
_GRPC_UNKNOWN = 2


def read_otlp_traces(paths):
    """Read OTLP/JSON trace files and return the ``traces.csv`` table, sorted
    by span start."""
    rows = []
    for path in [paths] if isinstance(paths, str) else paths:
        for request in _requests(path):
            rows.extend(_spans(request, path))
    if not rows:
        raise ValueError("no spans found in the OTLP input")
    frame = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    return frame.sort_values(["startTime", "spanID"], kind="stable").reset_index(drop=True)


def _requests(path):
    with open(path) as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # the collector's file exporter: one request per line
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    return data if isinstance(data, list) else [data]


def _spans(request, path):
    if "resourceSpans" not in request:
        raise ValueError(f"{path}: not an OTLP trace export (no resourceSpans)")
    for resource_spans in request["resourceSpans"]:
        resource = _attributes(resource_spans.get("resource", {}).get("attributes", []))
        service = resource.get("service.name")
        if service is None:
            raise ValueError(f"{path}: a resource has no service.name attribute")
        check_service(service)
        # instrumentationLibrarySpans: the name before OTLP 0.15
        scopes = resource_spans.get("scopeSpans", resource_spans.get("instrumentationLibrarySpans", []))
        for scope in scopes:
            for span in scope.get("spans", []):
                yield _row(span, service)


def _row(span, service):
    attributes = _attributes(span.get("attributes", []))
    start = int(span["startTimeUnixNano"])
    end = int(span["endTimeUnixNano"])
    return {
        "time": start // 1_000_000_000,
        # hex IDs, lower-cased like the released datasets'
        "traceID": span["traceId"].lower(),
        "spanID": span["spanId"].lower(),
        "serviceName": service,
        "methodName": attributes.get("rpc.method"),
        "operationName": span.get("name"),
        "startTimeMillis": start // 1_000_000,
        "startTime": start // 1_000,
        "duration": (end - start) // 1_000,
        "statusCode": _status_code(span, attributes),
        "parentSpanID": span.get("parentSpanId", "").lower() or None,
    }


def _status_code(span, attributes):
    if "rpc.grpc.status_code" in attributes:
        return int(attributes["rpc.grpc.status_code"])
    code = span.get("status", {}).get("code", 0)
    if code in (_STATUS_ERROR, "STATUS_CODE_ERROR"):
        http = attributes.get("http.response.status_code", attributes.get("http.status_code"))
        return int(http) if http is not None else _GRPC_UNKNOWN
    return 0


def _attributes(items):
    return {item["key"]: _value(item.get("value", {})) for item in items}


def _value(value):
    # OTLP/JSON wraps each value in its type; 64-bit ints arrive as strings
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        return int(value["intValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    return None
