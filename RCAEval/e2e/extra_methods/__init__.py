import pathlib
from RCAEval.e2e.extra_methods.microrca import MicroRCA
from RCAEval.e2e.extra_methods.microscope import MicroScope
from RCAEval.e2e.extra_methods.monitorrank import MonitorRank
from RCAEval.e2e import rca

@rca
def microrca(data, inject_time=None, dataset=None, args=None, **kwargs):
    cls_instance = MicroRCA(pathlib.Path(args.data_path).parent)
    ranks = cls_instance.execute()

    ranks = [r[0] for r in ranks]
    return {
        "ranks": ranks
    }

@rca
def microscope(data, inject_time=None, dataset=None, args=None, **kwargs):
    cls_instance = MicroScope(pathlib.Path(args.data_path).parent)
    ranks = cls_instance.execute()

    ranks = [r[0] for r in ranks]
    return {
        "ranks": ranks
    }

@rca
def monitorrank(data, inject_time=None, dataset=None, args=None, **kwargs):
    cls_instance = MonitorRank(pathlib.Path(args.data_path).parent)
    ranks = cls_instance.execute()

    ranks = [r[0] for r in ranks]
    return {
        "ranks": ranks
    }
