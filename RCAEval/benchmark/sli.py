"""Which metric main.py hands to the RCA methods as the SLI (`sli=`)."""


def select_sli(data_path, columns, service):
    """Return the SLI column for the case at `data_path`, given the metric
    columns after main.py's renaming and the service named in the case path."""
    columns = set(columns)
    if "my-sock-shop" in data_path or "fse-ss" in data_path:
        sli = "front-end_cpu"
        if f"{service}_latency" in columns:
            sli = f"{service}_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_lat_90" in columns:
            sli = f"{service}_lat_90"
    elif "train-ticket" in data_path or "fse-tt" in data_path or "RE2-TT" in data_path or "RE3-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
        if f"{service}_latency" in columns:
            sli = f"{service}_latency"
    elif "online-boutique" in data_path or "fse-ob" in data_path or "RE2-OB" in data_path or "RE2-SS" in data_path or "RE3-OB" in data_path or "RE3-SS" in data_path:
        sli = "frontend_latency"
        if f"{service}_latency" in columns:
            sli = f"{service}_latency"
        elif "frontend_1" in columns:
            sli = "frontend_1"
    elif "torai-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
    elif "torai-OB" in data_path or "torai-SS" in data_path:
        sli = "frontend_latency"
    else:
        raise ValueError("SLI not implemented")
    return sli
