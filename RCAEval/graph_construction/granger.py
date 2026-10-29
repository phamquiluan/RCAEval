import inspect
import warnings

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


def _granger_tests(data, maxlag):
    # statsmodels 0.15 removed verbose= and never prints; 0.14 prints every
    # test unless verbose=False, and warns that verbose is deprecated
    if "verbose" not in inspect.signature(grangercausalitytests).parameters:
        return grangercausalitytests(data, maxlag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return grangercausalitytests(data, maxlag, verbose=False)


def granger(data, maxlag=None, p_val_threshold=0.05, test=None):
    assert test in [None, 'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest']

	# data: pandas dataframe
    if maxlag is None:
        maxlag = 3
    node_names = data.columns.to_list()
    adj = np.zeros((len(node_names), len(node_names)))

    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if i == j:
                continue
            # test j -> i
            output = _granger_tests(data[[node_names[i], node_names[j]]], maxlag)
            caused = False
            for time_lag, out in output.items():
                out = out[0]

                if test is None:
                    avg_p_val = sum([v[1] for k, v in out.items()]) / len(out)
                else:
                    avg_p_val = out[test][1]

                if avg_p_val < p_val_threshold:  # average p-value
                    caused = True
                    break
            if caused:
                adj[i, j] = 1
    return adj
