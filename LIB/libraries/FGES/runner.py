
import pandas as pd
from libraries.FGES.fges_main import FGES
from sklearn.preprocessing import KBinsDiscretizer


def fges_runner(data, nodes, score, knowledge=None, disc=None, n_bins=5, file_name=None):
    if disc is not None:
        discretizer = KBinsDiscretizer(n_bins, encode="ordinal", strategy=disc)
        nodes_list = []
        for node in nodes:
            if node[1]["type"] == "cont":
                nodes_list.append(node[0])
        data[nodes_list] = discretizer.fit_transform(data[nodes_list])
    variables = list(range(len(data.to_numpy()[0])))

    fges = FGES(
        variables,
        nodes,
        data,
        knowledge=knowledge,
        filename=file_name,
        save_name=file_name,
        score=score,
    )
    result = fges.search()

    return result
