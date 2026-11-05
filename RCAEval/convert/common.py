"""Helpers shared by the converters."""
import pandas as pd


def check_service(service):
    """Methods recover the service from a column as the text before the first
    "_", so a service name must be non-empty and contain none."""
    if not isinstance(service, str) or not service or "_" in service:
        raise ValueError(f"service name {service!r} must be non-empty and contain no '_'")


def to_unix_ns(values):
    """Parse timestamps into int64 nanoseconds since the epoch.

    Numbers are taken as seconds, milliseconds, microseconds or nanoseconds by
    their magnitude (any date after 2001 is unambiguous); strings are parsed as
    dates, and dates without a timezone are taken as UTC.
    """
    values = pd.Series(values)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        scale = numeric.abs().map(_ns_per_unit)
        return (numeric * scale).round().astype("int64")
    if numeric.notna().any():
        raise ValueError("timestamps mix numbers and date strings")
    parsed = pd.to_datetime(values, utc=True, format="mixed")
    # pandas >= 3 picks the resolution from the input, e.g. microseconds
    return parsed.dt.as_unit("ns").astype("int64")


def _ns_per_unit(x):
    if x < 1e11:  # seconds until year 5138
        return 1_000_000_000
    if x < 1e14:
        return 1_000_000  # milliseconds
    if x < 1e17:
        return 1_000  # microseconds
    return 1
