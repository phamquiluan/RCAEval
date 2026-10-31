"""Turn a method's ranking into the Node lists the Evaluator scores.

Shared by main.py and the predictions evaluator (RCAEval.agent), so a ranking
scores the same whichever way it is submitted.
"""
from typing import List, Optional, Sequence

from RCAEval.classes.graph import Node

# RE1 telemetry carries no disk metric (only cpu, mem, workload, latency and error per
# service; blkio was added to the collector after RE1 was recorded), so its disk faults
# are scored against latency, as in the original evaluation. RE2 onward export
# `<service>_diskio`.
RE1_DATASETS = {"online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket", "re1-ob", "re1-ss", "re1-tt"}


def to_service(name: str) -> str:
    """The service a ranked candidate ("<service>" or "<service>_<indicator>")
    belongs to, with a database container folded into its service."""
    return name.split("_")[0].replace("-db", "")


def service_ranks(ranks: Sequence[str]) -> List[Node]:
    """Service-level ranking: each candidate's service, first occurrence kept."""
    s_ranks = [Node(to_service(x), "unknown") for x in ranks]
    # remove duplication
    old_s_ranks = s_ranks.copy()
    s_ranks = (
        [old_s_ranks[0]]
        + [
            old_s_ranks[i]
            for i in range(1, len(old_s_ranks))
            if old_s_ranks[i] not in old_s_ranks[:i]
        ]
        if old_s_ranks
        else []
    )
    return s_ranks


def indicator_ranks(ranks: Sequence[str]) -> List[Node]:
    """Fine-grained ranking: (service, indicator) per candidate, "unknown" when
    the candidate names only a service."""
    return [Node(x.split("_")[0], x.split("_")[1] if "_" in x else "unknown") for x in ranks]


def fault_to_indicator(fault: str, dataset: str) -> Optional[str]:
    """The indicator a fault is scored against, or None for faults with no
    metric indicator (the RE3 code-level faults f1-f5)."""
    if fault in ("cpu", "mem", "socket"):
        return fault
    if fault in ("delay", "loss"):
        return "latency"
    if fault == "disk":
        return "latency" if dataset in RE1_DATASETS else "diskio"
    return None
