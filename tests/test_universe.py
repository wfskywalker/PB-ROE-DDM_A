import pandas as pd
import pytest

from pbroe.universe import (
    UniverseError,
    active_sw_membership,
    build_signal_universe,
    month_end_trade_dates,
)


def stock_basic_fixture():
    return pd.DataFrame(
        {
            "ts_code": [
                "000001.SZ",
                "600001.SH",
                "830001.BJ",
                "000002.SZ",
                "000003.SZ",
                "000004.SZ",
                "000005.SZ",
            ],
            "name": ["A", "B", "C", "D", "E", "F", "G"],
            "list_date": [20100101, 20100101, 20100101, 20250101, 20100101, 20100101, 20100101],
            "delist_date": [None, None, None, None, 20201231, None, None],
        }
    )


def sw_fixture():
    rows = []
    codes = ["000001.SZ", "600001.SH", "830001.BJ", "000002.SZ", "000003.SZ", "000004.SZ"]
    l1_names = ["银行", "电子", "电子", "电子", "电子", "电子"]
    for idx, (code, l1_name) in enumerate(zip(codes, l1_names)):
        rows.append(
            {
                "ts_code": code,
                "l1_code": f"L1{idx}",
                "l1_name": l1_name,
                "l2_code": f"L2{idx}",
                "l2_name": "二级",
                "l3_code": f"L3{idx}",
                "l3_name": "三级",
                "in_date": 20100101,
                "out_date": None,
            }
        )
    return pd.DataFrame(rows)


def trade_filter_fixture(signal_date=20240131):
    return pd.DataFrame(
        {
            "trade_date": [signal_date] * 7,
            "ts_code": stock_basic_fixture()["ts_code"],
            "is_st": [False, False, False, False, False, True, False],
        }
    )


def test_point_in_time_universe_filters_all_required_categories():
    result = build_signal_universe(
        20240131,
        stock_basic_fixture(),
        sw_fixture(),
        trade_filter_fixture(),
        pd.DataFrame(columns=["trade_date", "ts_code"]),
    )
    assert result.eligible["ts_code"].tolist() == ["600001.SH"]
    reasons = result.audit.set_index("ts_code")["exclusion_reason"].to_dict()
    assert reasons["000001.SZ"] == "EXCLUDED_INDUSTRY"
    assert reasons["830001.BJ"] == "INVALID_EXCHANGE"
    assert reasons["000002.SZ"] == "NOT_LISTED"
    assert reasons["000003.SZ"] == "NOT_LISTED"
    assert reasons["000004.SZ"] == "ST"
    assert reasons["000005.SZ"] == "MISSING_INDUSTRY"


def test_future_industry_transition_does_not_change_past_membership():
    sw = sw_fixture()
    transition = pd.DataFrame(
        [
            {
                "ts_code": "600001.SH",
                "l1_code": "FUTURE",
                "l1_name": "房地产",
                "l2_code": "FUTURE2",
                "l2_name": "未来二级",
                "l3_code": "FUTURE3",
                "l3_name": "未来三级",
                "in_date": 20250101,
                "out_date": None,
            }
        ]
    )
    sw = pd.concat([sw, transition], ignore_index=True)
    past = active_sw_membership(sw, 20240131).set_index("ts_code")
    assert past.loc["600001.SH", "l1_name"] == "电子"


def test_future_st_record_does_not_change_past_universe():
    future_st = pd.DataFrame(
        {"trade_date": [20240201], "ts_code": ["600001.SH"]}
    )
    result = build_signal_universe(
        20240131,
        stock_basic_fixture(),
        sw_fixture(),
        trade_filter_fixture(),
        future_st,
    )
    assert "600001.SH" in set(result.eligible["ts_code"])


def test_conflicting_active_membership_is_rejected():
    sw = sw_fixture()
    conflict = sw.loc[sw["ts_code"].eq("600001.SH")].copy()
    conflict["l3_code"] = "CONFLICT"
    sw = pd.concat([sw, conflict], ignore_index=True)
    with pytest.raises(UniverseError, match="conflicting active"):
        active_sw_membership(sw, 20240131)


def test_month_end_dates_use_only_open_days():
    calendar = pd.DataFrame(
        {
            "cal_date": [20240130, 20240131, 20240201, 20240228, 20240229],
            "is_open": [1, 0, 1, 1, 0],
        }
    )
    assert month_end_trade_dates(calendar) == [20240130, 20240228]
