# Converting your own telemetry

`rcaeval-convert` turns telemetry exported from your own system into an RCAEval case directory, so the RCA methods can run on it. It reads standard export formats rather than talking to each monitoring product:

| Source | Input | Writes |
|---|---|---|
| Prometheus | named PromQL range queries, pulled from the server | `metrics.csv` |
| Jaeger, Zipkin, SkyWalking, any OpenTelemetry SDK | OTLP/JSON, as written by the OpenTelemetry Collector's file exporter | `traces.csv` |
| ELK, Loki, Fluentd, any log store | a JSON Lines, JSON or CSV export, with a field mapping | `logs.csv` |
| the files above | `logs.csv`, `traces.csv` and the failure start time | `logts.csv`, `cluster_info.json`, `tracets_err.csv`, `tracets_lat.csv`, `inject_time.txt` |

Install it with the `convert` extra, which adds [Drain3](https://github.com/logpai/Drain3) for log templates:

```bash
pip install RCAEval[convert]
```

A converted case has no label. It is enough to run methods on a real incident, but to evaluate them you also need the root cause, and the failure start time (`inject_time.txt`) must come from you: when the failure was injected, or when it was detected.

## Building a case

```bash
# metrics: 12 minutes either side of the failure, one sample per second (as RE2/RE3)
rcaeval-convert metrics --prometheus-url http://localhost:9090 \
    --queries docs/examples/prometheus_queries.json \
    --start 2024-01-15T21:12:00Z --end 2024-01-15T21:36:00Z --out case

# traces: the OpenTelemetry Collector's file exporter output
rcaeval-convert traces otel-traces.jsonl --out case

# logs: a Loki export (logcli query --output=jsonl ...)
rcaeval-convert logs loki.jsonl --field container=labels.container --field message=line --out case

# the 15-second series for the multi-source methods, and inject_time.txt
rcaeval-convert derive case --inject-time 2024-01-15T21:24:00Z
```

Times take unix seconds or dates; a date without a timezone is UTC. Every step is optional: metric-based methods need only `metrics.csv` and `inject_time.txt`. Existing files are kept unless you pass `--force`.

Then run a method on the case:

```python
import pandas as pd
from RCAEval.e2e import baro

data = pd.read_csv("case/metrics.csv")
inject_time = int(open("case/inject_time.txt").read())
print(baro(data, inject_time)["ranks"][:5])
```

The same functions are available from Python as `RCAEval.convert.prometheus.pull_metrics`, `RCAEval.convert.otlp.read_otlp_traces`, `RCAEval.convert.logs.read_logs` and `RCAEval.convert.derive.derive_case`.

## Service names

Methods recover a column's service as the text before its first `_`, so service names must not contain `_`; the converters reject them. Use the same name for a service in every source: `cart` in the metrics, the traces' `service.name` and the logs' container field.

## Metrics: Prometheus queries

The queries file is a JSON object from a metric name to a PromQL query. Each series a query returns becomes the column `{service}_{name}`, where the service is the series' only label, or the label named with `"label"`:

```json
{
  "cpu": "sum(rate(container_cpu_usage_seconds_total{namespace=\"default\"}[1m])) by (container) * 100",
  "latency-90": {
    "query": "histogram_quantile(0.90, sum(rate(istio_request_duration_milliseconds_bucket[1m])) by (le, destination_workload)) / 1000",
    "label": "destination_workload"
  }
}
```

Aggregate each query `by` the label that names the service; two series that map to the same column are an error. [`docs/examples/prometheus_queries.json`](examples/prometheus_queries.json) has the cAdvisor and Istio queries the RE2/RE3 datasets used, for the `default` namespace. Long ranges are split into several requests, and `--header 'Authorization: Bearer ...'` authenticates.

Methods that start from an SLI (MicroCause, CloudRanger) look for the entry service's latency, e.g. `frontend_latency`; `main.py` renames `_latency-90` columns to `_latency`. See `RCAEval/benchmark/sli.py`.

## Traces: OTLP/JSON

Each span becomes a row of `traces.csv`: `time` (unix seconds), `traceID`, `spanID`, `serviceName` (the resource's `service.name`), `methodName` (`rpc.method`), `operationName` (the span name), `startTimeMillis`, `startTime` and `duration` (microseconds), `statusCode` and `parentSpanID`.

`statusCode` follows the released datasets, which record gRPC status codes, 0 meaning OK: the span's `rpc.grpc.status_code` if it has one; otherwise, for a span whose OpenTelemetry status is ERROR, its HTTP status code, or 2 when it has none; otherwise 0. A span is an error exactly when its `statusCode` is not 0.

To get OTLP/JSON from Jaeger, Zipkin or SkyWalking, send their spans through an OpenTelemetry Collector (it has receivers for all three) with the `file` exporter.

## Logs: field mapping

`--field FIELD=PATH` says where each field is in the export. The fields are `timestamp`, `container`, `message` (required) and `level`, `req_path`, `error` (optional); unmapped fields keep their default path, the `logs.csv` column name (`timestamp`, `container_name`, `message`, ...). A path descends into nested JSON with `.`, and a key that itself contains dots is matched whole first:

| Source | Mapping |
|---|---|
| Loki (`logcli query --output=jsonl`) | `container=labels.container message=line` |
| Elasticsearch documents | `timestamp=_source.@timestamp container=_source.kubernetes.container.name message=_source.message` |
| Fluentd file output | `timestamp=time container=kubernetes.container_name message=log` |

Timestamps may be in seconds, milliseconds, microseconds or nanoseconds (told apart by size) or date strings. Records without a timestamp, container or message are dropped with a warning that counts them.

## Derived series

`derive` bins logs and spans into 15-second intervals aligned to the epoch:

- `logts.csv`: log lines per template, one `{container}_{cluster_id}` column each. Templates are mined with Drain3 from `{container}_{message}`, and are listed in `cluster_info.json`; `logs.csv` gains `cluster_id` and `log_template` columns. Without `--drain-config`, numbers, UUIDs, IP addresses and long hex IDs are masked; a Drain3 `.ini` with masks for your system's own IDs merges more lines into one template.
- `tracets_err.csv`: error spans per operation, where an operation is `{serviceName}_{methodName}`, or `{serviceName}_{operationName}` for a span without a method.
- `tracets_lat.csv`: slow spans per operation: longer than the mean plus 3 standard deviations of that operation's spans that ended before the failure started. An operation with fewer than two such spans never counts as slow.

These follow the scripts behind the released datasets, so the files have the same layout, but the values differ in a few ways: the released `tracets_err` summed status codes instead of counting error spans; the released latency baseline mixed milliseconds and microseconds when selecting the spans before the failure; and the released operation names and Drain3 settings were specific to each system.

Multi-source BARO (`mmbaro`) reads the trace series only when `dataset` is one of the Online Boutique or Train Ticket names (`re2-ob`, `re2-tt`, ...); for a converted case, pass one of those to use them.
