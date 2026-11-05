"""Convert exported telemetry into an RCAEval case directory.

A case directory holds the files the RCA methods read:

- ``metrics.csv``: a ``time`` column (unix seconds) and one ``{service}_{metric}``
  column per series (`prometheus`)
- ``traces.csv``: one row per span (`otlp`)
- ``logs.csv``: one row per log line (`logs`)
- ``logts.csv``, ``cluster_info.json``, ``tracets_err.csv``, ``tracets_lat.csv``:
  15-second time series derived from the logs and traces (`derive`)
- ``inject_time.txt``: when the failure started, in unix seconds

See docs/CONVERT.md for the format and the ``rcaeval-convert`` command.
"""
