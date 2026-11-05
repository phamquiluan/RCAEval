"""Which metric main.py hands to the RCA methods as the SLI (`sli=`).

The SLI is the service-level indicator an operator watches when an incident
starts: the latency of the system's entry service (the frontend). Methods such
as MicroCause and CloudRanger start their search from it, so it must not
depend on the case's label. RCAEval 1.11.0 and earlier used the root-cause
service's latency, read from the case directory name, whenever that column
existed (issue #88).
"""

# methods that start their search from the SLI; the others ignore `sli`
SLI_METHODS = ("cloudranger", "microcause")

# (path markers, entry-latency candidates). The first entry whose marker occurs
# in the case path applies; within it, the first candidate column present in
# the data is the SLI. Order matters: "sock-shop" also matches "my-sock-shop".
_ENTRY_SLI = [
    (("my-sock-shop", "fse-ss"), ("front-end_latency",)),
    # RE1-SS (sock-shop-2) records istio latency; sock-shop-1 records no
    # front-end latency, so it has no SLI
    (("sock-shop",), ("front-end_istio-latency-90", "front-end_lat_90")),
    (("train-ticket", "fse-tt", "RE2-TT", "RE3-TT", "torai-TT"), ("ts-ui-dashboard_latency",)),
    # Sock Shop's entry service is "front-end", not "frontend"
    (("RE2-SS", "RE3-SS"), ("front-end_latency",)),
    (("online-boutique", "fse-ob", "RE2-OB", "RE3-OB"), ("frontend_latency", "frontend_1")),
    # unchanged: the TORAI data is not public, so its column names are unverified
    (("torai-OB", "torai-SS"), ("frontend_latency",)),
]


class NoSLIError(ValueError):
    """The case records no entry-service latency to use as the SLI."""


def select_sli(data_path, columns):
    """Return the SLI column for the case at `data_path`: the entry service's
    latency. `columns` are the case's metric columns after main.py's renaming
    (`_latency-90` -> `_latency`). Raise NoSLIError if the case has none."""
    columns = set(columns)
    for markers, candidates in _ENTRY_SLI:
        if any(m in data_path for m in markers):
            for c in candidates:
                if c in columns:
                    return c
            raise NoSLIError(
                f"{data_path} has no SLI: none of {', '.join(candidates)} is recorded"
            )
    raise ValueError(f"SLI not implemented for {data_path}")
