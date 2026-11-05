"""rcaeval-convert: turn exported telemetry into an RCAEval case directory.

    rcaeval-convert metrics --prometheus-url http://localhost:9090 --queries queries.json \\
        --start 2024-01-15T21:12:00Z --end 2024-01-15T21:36:00Z --out case
    rcaeval-convert traces otel-traces.jsonl --out case
    rcaeval-convert logs loki.jsonl --field container=labels.container --field message=line --out case
    rcaeval-convert derive case --inject-time 2024-01-15T21:24:00Z

See docs/CONVERT.md.
"""
import argparse
import os
import sys

from RCAEval.convert.common import to_unix_ns


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rcaeval-convert", description="Convert exported telemetry into an RCAEval case")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p = sub.add_parser("metrics", help="pull Prometheus range queries into metrics.csv")
    p.add_argument("--prometheus-url", required=True, help="e.g. http://localhost:9090")
    p.add_argument("--queries", required=True, help="JSON file: metric name -> PromQL, or -> {query, label}")
    p.add_argument("--start", required=True, help="unix seconds or a date, e.g. 2024-01-15T21:12:00Z")
    p.add_argument("--end", required=True, help="unix seconds or a date")
    p.add_argument("--step", type=int, default=1, help="seconds between samples (default 1, as in RE2/RE3)")
    p.add_argument("--header", action="append", default=[], metavar="'NAME: VALUE'",
                   help="HTTP header sent with every request, e.g. 'Authorization: Bearer ...' (repeatable)")
    _out(p)

    p = sub.add_parser("traces", help="convert OpenTelemetry OTLP/JSON traces into traces.csv")
    p.add_argument("inputs", nargs="+", help="OTLP/JSON files (the collector's file exporter output)")
    _out(p)

    p = sub.add_parser("logs", help="convert a JSON Lines, JSON or CSV log export into logs.csv")
    p.add_argument("inputs", nargs="+", help="log export files")
    p.add_argument("--field", action="append", default=[], metavar="FIELD=PATH",
                   help="where a field is in the input, e.g. container=labels.container (repeatable); "
                        "fields: timestamp, container, message, level, req_path, error")
    _out(p)

    p = sub.add_parser("derive", help="build logts.csv and tracets_*.csv from a case's logs.csv and traces.csv")
    p.add_argument("case", help="the case directory")
    p.add_argument("--inject-time", default=None,
                   help="when the failure started (unix seconds or a date); written to inject_time.txt "
                        "(default: read it from inject_time.txt)")
    p.add_argument("--drain-config", default=None, help="Drain3 .ini file for log template mining")

    args = parser.parse_args(argv)
    try:
        return {"metrics": _metrics, "traces": _traces, "logs": _logs, "derive": _derive}[args.subcommand](args)
    # OSError covers missing files and requests' connection and HTTP errors
    except (ValueError, OSError, RuntimeError, ImportError) as e:
        print(f"rcaeval-convert {args.subcommand}: error: {e}", file=sys.stderr)
        return 1


def _out(p):
    p.add_argument("--out", required=True, help="the case directory to write into (created if missing)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")


def _seconds(value):
    return int(to_unix_ns([value])[0] // 1_000_000_000)


def _target(args, name):
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, name)
    if os.path.exists(path) and not args.force:
        raise FileExistsError(f"{path} exists; pass --force to overwrite it")
    return path


def _write(frame, path):
    frame.to_csv(path, index=False)
    print(f"wrote {path} ({len(frame)} rows, {frame.shape[1]} columns)")
    return 0


def _metrics(args):
    import requests

    from RCAEval.convert.prometheus import load_queries, pull_metrics

    path = _target(args, "metrics.csv")
    session = requests.Session()
    for header in args.header:
        name, sep, value = header.partition(":")
        if not sep:
            raise ValueError(f"--header {header!r}: expected 'NAME: VALUE'")
        session.headers[name.strip()] = value.strip()
    frame = pull_metrics(args.prometheus_url, load_queries(args.queries),
                         _seconds(args.start), _seconds(args.end), args.step, session=session)
    return _write(frame, path)


def _traces(args):
    from RCAEval.convert.otlp import read_otlp_traces

    path = _target(args, "traces.csv")
    return _write(read_otlp_traces(args.inputs), path)


def _logs(args):
    from RCAEval.convert.logs import read_logs

    fields = {}
    for item in args.field:
        name, sep, value = item.partition("=")
        if not sep or not value:
            raise ValueError(f"--field {item!r}: expected FIELD=PATH")
        fields[name] = value
    path = _target(args, "logs.csv")
    return _write(read_logs(args.inputs, fields), path)


def _derive(args):
    from RCAEval.convert.derive import derive_case

    if not os.path.isdir(args.case):
        raise FileNotFoundError(f"{args.case} is not a directory")
    inject_time = None
    if args.inject_time is not None:
        inject_time = _seconds(args.inject_time)
        path = os.path.join(args.case, "inject_time.txt")
        if os.path.exists(path):
            with open(path) as f:
                existing = int(f.readline().strip())
            if existing != inject_time:
                raise ValueError(f"{path} says {existing}, not {inject_time}")
        else:
            with open(path, "w") as f:
                f.write(str(inject_time))
    written = derive_case(args.case, inject_time=inject_time, drain_config=args.drain_config)
    print(f"wrote {', '.join(written)} in {args.case}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
