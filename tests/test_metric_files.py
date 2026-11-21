"""Which file main.py and the agent export read as a case's metrics (issue #97)."""
import os
from os.path import join

from RCAEval.benchmark.datasets import metric_files


def make_case(root, case, names):
    d = join(root, case)
    os.makedirs(d)
    for name in names:
        with open(join(d, name), "w") as f:
            f.write("time\n")
    return d


def test_simple_data_wins_over_the_raw_counters(tmp_path):
    # RE1-SS and RE1-TT: main.py read data.csv from 1.3.0 to 1.12.1
    root = str(tmp_path)
    a = make_case(root, "carts_cpu/1", ["data.csv", "simple_data.csv"])
    b = make_case(root, "carts_cpu/2", ["data.csv", "simple_data.csv"])
    assert metric_files(root) == [join(a, "simple_data.csv"), join(b, "simple_data.csv")]


def test_data_csv_where_there_is_no_simple_data(tmp_path):
    # RE1-OB and sock-shop-1, and a case that lacks the derived file
    root = str(tmp_path)
    a = make_case(root, "carts_cpu/1", ["data.csv"])
    b = make_case(root, "carts_cpu/2", ["data.csv", "simple_data.csv"])
    assert metric_files(root) == [join(a, "data.csv"), join(b, "simple_data.csv")]


def test_simple_metrics_for_re2_onward(tmp_path):
    root = str(tmp_path)
    a = make_case(root, "carts_cpu/1", ["simple_metrics.csv", "logs.csv"])
    assert metric_files(root) == [join(a, "simple_metrics.csv")]


def test_no_cases(tmp_path):
    assert metric_files(str(tmp_path)) == []
