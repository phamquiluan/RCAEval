from cfm.graph_construction.pc import pc_default
from cfm.graph_heads.rht import rht
from cfm.io.time_series import (
    preprocess,
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    select_useful_cols,
)


def circa(data, inject_time=None, dataset=None, **kwargs):
    time_col = data["time"]

    data = preprocess(
        data=data,
        dataset=dataset,
        dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # add time again
    data["time"] = time_col


    # if dataset is not None:
    #     data = drop_constant(data)
    #     data = convert_mem_mb(data)
    #     
    #     if kwargs.get("dk_select_useful", False) is True:
    #         data = drop_near_constant(drop_extra(data))
    #         data = data[select_useful_cols(data)]
    # node_names = data.columns.to_list()

    # graph construction
    pc_input = data.drop(columns=["time"])
    node_names = pc_input.columns.to_list()

    adj = []
    try:
        adj = pc_default(pc_input, dataset="ob")
    except Exception as e:
        print("PC failed, using empty graph")
        print(e)
        return {
            "adj": adj,
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
