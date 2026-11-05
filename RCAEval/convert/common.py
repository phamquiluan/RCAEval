"""Helpers shared by the converters."""


def check_service(service):
    """Methods recover the service from a column as the text before the first
    "_", so a service name must be non-empty and contain none."""
    if not isinstance(service, str) or not service or "_" in service:
        raise ValueError(f"service name {service!r} must be non-empty and contain no '_'")
