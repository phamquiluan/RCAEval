"""The SLI main.py hands to methods must not depend on the case's label (issue #88)."""
import pytest

from RCAEval.benchmark.sli import NoSLIError, select_sli

OB_COLS = ["time", "frontend_latency", "checkoutservice_latency", "productcatalogservice_latency"]
SS_COLS = ["time", "front-end_latency", "carts_latency", "orders_latency"]
TT_COLS = ["time", "ts-ui-dashboard_latency", "ts-auth-service_latency", "ts-order-service_latency"]
RE1_SS_COLS = ["time", "front-end_istio-latency-90", "carts_istio-latency-90", "orders_istio-latency-90"]
SS1_COLS = ["time", "front-end_cpu", "carts_lat_90", "orders_lat_90"]


@pytest.mark.parametrize(
    "system, columns, services, expected",
    [
        ("RE2/RE2-OB", OB_COLS, ["checkoutservice", "productcatalogservice"], "frontend_latency"),
        ("RE3/RE3-OB", OB_COLS, ["checkoutservice", "productcatalogservice"], "frontend_latency"),
        ("online-boutique", OB_COLS, ["checkoutservice", "productcatalogservice"], "frontend_latency"),
        ("RE2/RE2-SS", SS_COLS, ["carts", "orders"], "front-end_latency"),
        ("RE3/RE3-SS", SS_COLS, ["carts", "orders"], "front-end_latency"),
        ("sock-shop-2", RE1_SS_COLS, ["carts", "orders"], "front-end_istio-latency-90"),
        ("RE2/RE2-TT", TT_COLS, ["ts-auth-service", "ts-order-service"], "ts-ui-dashboard_latency"),
        ("train-ticket", TT_COLS, ["ts-auth-service", "ts-order-service"], "ts-ui-dashboard_latency"),
    ],
)
def test_entry_sli_ignores_the_root_cause_in_the_path(system, columns, services, expected):
    # the same data moved under another case's label must get the same SLI; it
    # used to become the root-cause service's latency
    slis = {select_sli(f"data/{system}/{s}_cpu/1/data.csv", columns) for s in services}
    assert slis == {expected}


def test_sock_shop_entry_sli_is_front_end():
    # RE2-SS used to fall back to "frontend_latency", which Sock Shop does not have
    assert select_sli("data/RE2/RE2-SS/carts_cpu/1/data.csv", ["time", "frontend_latency", "front-end_latency"]) == "front-end_latency"


def test_missing_entry_latency_raises():
    # it used to hand over a column the case does not have
    with pytest.raises(NoSLIError):
        select_sli("data/RE2/RE2-OB/checkoutservice_cpu/1/data.csv", ["time", "frontend_cpu"])


def test_sock_shop_1_has_no_sli():
    # sock-shop-1 records no front-end latency; it used to get front-end_cpu,
    # or the root cause's latency
    with pytest.raises(NoSLIError):
        select_sli("data/sock-shop-1/sock-shop/carts_cpu/1/data.csv", SS1_COLS)


def test_re1_ss_entry_sli_exists():
    # RE1-SS used to get "front-end_cpu", a column it does not have
    assert select_sli("data/sock-shop-2/carts_cpu/1/data.csv", RE1_SS_COLS) in RE1_SS_COLS


def test_unknown_system_raises():
    with pytest.raises(ValueError):
        select_sli("data/unknown/x_cpu/1/data.csv", ["time"])
