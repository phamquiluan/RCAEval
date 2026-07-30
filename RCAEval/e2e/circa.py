import pandas as pd

from RCAEval.graph_construction.pc import pc_default
from RCAEval.graph_heads.rht import rht
from RCAEval.io.time_series import (
    preprocess,
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    select_useful_cols,
)
from RCAEval.e2e import rca


@rca
def circa(data, inject_time=None, dataset=None, **kwargs):
    time_col = data["time"]

    data = preprocess(
        data=data,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # add time again
    data["time"] = time_col

    # graph construction
    pc_input = data.drop(columns=["time"])
    node_names = pc_input.columns.to_list()

    adj = pc_default(pc_input, dataset="ob")
    ranks = rht(adj, inject_time, data)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": data.columns.to_list(),
        "ranks": ranks,
    }


def mmcirca(data, inject_time=None, dataset=None, **kwargs):
    """Multi-source CIRCA: CIRCA over the combined metric and log time series.

    `data` is a dict with at least the keys "metric" (raw metric DataFrame,
    sampled at 1s) and "logts" (log event time series, sampled at 15s), as
    produced by main.py or docs/multi-source-rca-demo.ipynb.
    """
    metric = data["metric"]
    logts = data["logts"]

    # metrics are sampled at 1s while logts is sampled at 15s;
    # align them by taking one metric point every 15
    metric = metric.iloc[::15, :]
    time_col = metric["time"]

    # == metric ==
    normal_metric = metric[metric["time"] < inject_time]
    anomal_metric = metric[metric["time"] >= inject_time]
    normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
    normal_metric = normal_metric[intersect]
    anomal_metric = anomal_metric[intersect]
    combined = pd.concat([normal_metric, anomal_metric], axis=0, ignore_index=True)

    # == logts ==
    logts = drop_constant(logts)
    normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
    anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])
    log = pd.concat([normal_logts, anomal_logts], axis=0, ignore_index=True)
    combined = pd.concat([combined, log], axis=1)

    combined = combined.loc[:, ~combined.columns.duplicated()]
    combined = combined.fillna(0)

    # remove highly correlated columns to keep the causal graph tractable
    removed_columns = []
    for idx, col1 in enumerate(combined.columns):
        for col2 in combined.columns[idx + 1:]:
            if combined[col1].corr(combined[col2]) > 0.99:
                removed_columns.append(col2)
    combined = combined.drop(columns=list(set(removed_columns)))

    if "time" not in combined.columns:
        combined["time"] = pd.Series(time_col.to_numpy()).reindex(combined.index).ffill()

    data = combined

    # graph construction
    pc_input = data.drop(columns=["time"])
    node_names = pc_input.columns.to_list()

    try:
        adj = pc_default(pc_input, dataset="ob")
    except Exception:
        return {
            "adj": [],
            "node_names": node_names,
            "ranks": node_names,
        }

    ranks = rht(adj, inject_time, data)
    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]
    return {
        "adj": adj,
        "node_names": data.columns.to_list(),
        "ranks": ranks,
    }
