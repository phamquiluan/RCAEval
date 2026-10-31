"""Where each benchmark dataset lives locally, and how to fetch it.

Shared by main.py and RCAEval.agent.
"""
from RCAEval.benchmark.scoring import RE1_DATASETS  # noqa: F401  (re-exported)

DATASET_MAP = {
    "online-boutique": "data/online-boutique",
    "sock-shop-1": "data/sock-shop-1",
    "sock-shop-2": "data/sock-shop-2",
    "train-ticket": "data/train-ticket",
    "re1-ob": "data/online-boutique",
    "re1-ss": "data/sock-shop-2",
    "re1-tt": "data/train-ticket",
    "re2-ob": "data/RE2/RE2-OB",
    "re2-ss": "data/RE2/RE2-SS",
    "re2-tt": "data/RE2/RE2-TT",
    "re3-ob": "data/RE3/RE3-OB",
    "re3-ss": "data/RE3/RE3-SS",
    "re3-tt": "data/RE3/RE3-TT",
    "torai-ob": "data/torai-OB",
    "torai-ss": "data/torai-SS",
    "torai-tt": "data/torai-TT",
    "eventadl-falcon": "data/eventadl-falcon",
    "eventadl-flask": "data/eventadl-flask",
    "eventadl-live": "data/eventadl-live",
}


def download_dataset(name):
    """Download `name` into data/ unless it is already there."""
    # imported here so that looking up DATASET_MAP needs no network stack
    from RCAEval.utility import (
        download_online_boutique_dataset,
        download_sock_shop_1_dataset,
        download_sock_shop_2_dataset,
        download_train_ticket_dataset,
        download_re2ob_dataset,
        download_re2ss_dataset,
        download_re2tt_dataset,
        download_re3_dataset,
        download_eventadl_dataset,
    )

    if "online-boutique" in name or "re1-ob" in name:
        download_online_boutique_dataset()
    elif "sock-shop-1" in name:
        download_sock_shop_1_dataset()
    elif "sock-shop-2" in name or "re1-ss" in name:
        download_sock_shop_2_dataset()
    elif "train-ticket" in name or "re1-tt" in name:
        download_train_ticket_dataset()
    elif "re2-ob" in name.lower():
        download_re2ob_dataset()
    elif "re2-ss" in name.lower():
        download_re2ss_dataset()
    elif "re2-tt" in name.lower():
        download_re2tt_dataset()
    elif "re3" in name:
        download_re3_dataset()
    elif "torai" in name:
        pass  # torai data is expected to be local
    elif "eventadl" in name:
        download_eventadl_dataset(name=name.replace("eventadl-", ""))
    else:
        raise Exception(f"{name} is not defined!")
