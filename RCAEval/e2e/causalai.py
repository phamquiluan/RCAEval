import numpy as np
import pandas as pd
from causalai.application import RootCauseDetector
from causalai.application.common import rca_preprocess


def drop_constant_column(df):
    for col in df.columns:
        if df[col].nunique() == 1:
            df.drop(col, axis=1, inplace=True)
    return df


def causalai(data, inject_time=None, dataset=None, with_bg=False, args=None, **kwargs):
    data = drop_constant_column(data)
    
    if 'time' not in data.columns:
        data['time'] = np.arange(len(data))


    df_normal = data[data["time"] < inject_time]
    df_abnormal = data[data["time"] >= inject_time]

    lower_level_columns = [c for c in df_normal.columns if c not in ["time"]]
    upper_level_metric = data["time"].tolist()
    df_normal = df_normal[lower_level_columns]
    df_abnormal = df_abnormal[lower_level_columns]

    data_obj, var_names = rca_preprocess(
        data=[df_normal, df_abnormal],
        time_metric=upper_level_metric,
        time_metric_name="time"
    )

    model = RootCauseDetector(
        data_obj = data_obj,
        var_names=var_names,
        time_metric_name="time",
        prior_knowledge=None
        )

    root_causes, graph = model.run(
        pvalue_thres=0.001,
        max_condition_set_size=4,
        return_graph=True
    )

    return {
        "ranks": list(root_causes)
    }
    
