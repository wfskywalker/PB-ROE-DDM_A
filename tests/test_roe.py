import pandas as pd
import pytest

from pbroe.roe import build_steady_roe


def universe_fixture():
    return pd.DataFrame(
        {
            "ts_code": ["A.SZ", "B.SZ", "C.SZ", "D.SZ", "E.SZ"],
            "l1_code": ["L1"] * 5,
            "l2_code": ["L2"] * 5,
            "l3_code": ["L3"] * 5,
        }
    )


def indicators_fixture():
    rows = []
    for code, values in {
        "A.SZ": [(2022, 10, 12), (2023, 20, 22), (2024, 30, 32)],
        "B.SZ": [(2022, 8, 10), (2023, 10, 12), (2024, 12, 14)],
        "C.SZ": [(2022, -12, -10), (2023, -10, -8), (2024, -8, -6)],
        "D.SZ": [(2024, 40, 50)],
        "E.SZ": [(2024, -40, -50)],
    }.items():
        for year, roe, waa in values:
            rows.append(
                {
                    "ts_code": code,
                    "ann_date": int(f"{year + 1}0430"),
                    "end_date": int(f"{year}1231"),
                    "roe": roe,
                    "roe_waa": waa,
                }
            )
    return pd.DataFrame(rows)


def test_three_year_mean_uses_roe_waa_consistently():
    result = build_steady_roe(
        indicators_fixture(), universe_fixture(), 20250430, min_peer_count=2
    )
    row = result.audit.set_index("ts_code").loc["A.SZ"]
    assert row["roe_metric"] == "roe_waa"
    assert row["steady_roe"] == pytest.approx((0.12 + 0.22 + 0.32) / 3)
    assert row["roe_observation_count"] == 3


def test_missing_waa_falls_back_for_entire_window_to_roe():
    indicators = indicators_fixture()
    indicators.loc[
        indicators["ts_code"].eq("A.SZ") & indicators["end_date"].eq(20231231),
        "roe_waa",
    ] = None
    result = build_steady_roe(indicators, universe_fixture(), 20250430, min_peer_count=2)
    row = result.audit.set_index("ts_code").loc["A.SZ"]
    assert row["roe_metric"] == "roe"
    assert row["steady_roe"] == pytest.approx(0.20)


def test_future_announcement_is_not_visible():
    indicators = indicators_fixture()
    result = build_steady_roe(indicators, universe_fixture(), 20240429, min_peer_count=2)
    row = result.audit.set_index("ts_code").loc["A.SZ"]
    assert row["roe_observation_count"] == 1
    assert row["latest_roe_ann_date"] <= 20240429


def test_new_listing_uses_documented_industry_shrinkage():
    result = build_steady_roe(
        indicators_fixture(), universe_fixture(), 20250430, min_peer_count=2
    )
    row = result.audit.set_index("ts_code").loc["D.SZ"]
    assert row["roe_observation_count"] == 1
    assert row["industry_prior_level"] == "SW3"
    expected = (1 / 3) * 0.50 + (2 / 3) * row["industry_roe_prior"]
    assert row["steady_roe"] == pytest.approx(expected)


def test_non_positive_steady_roe_is_excluded():
    result = build_steady_roe(
        indicators_fixture(), universe_fixture(), 20250430, min_peer_count=2
    )
    row = result.audit.set_index("ts_code").loc["C.SZ"]
    assert row["roe_reason"] == "NON_POSITIVE_STEADY_ROE"
    assert not row["roe_valid"]
