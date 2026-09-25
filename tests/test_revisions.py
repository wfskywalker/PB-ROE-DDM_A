import numpy as np
import pandas as pd
import pytest

from pbroe.revisions import build_revision_components, form_revision_score


def _targets():
    return pd.DataFrame([
        {"signal_date": 20240628, "ts_code": "000001.SZ", "forecast_year": 2026},
        {"signal_date": 20240628, "ts_code": "000002.SZ", "forecast_year": 2026},
    ])


def _reports():
    rows = []
    for code, scale in (("000001.SZ", 1.0), ("000002.SZ", 2.0)):
        for broker, change in (("甲证券股份有限公司", 0.2), ("乙证券研究所", -0.1)):
            rows += [
                {"ts_code": code, "report_date": 20240320, "org_name": broker,
                 "quarter": "2026Q4", "eps": scale, "create_time": "2024-03-20 09:00:00"},
                {"ts_code": code, "report_date": 20240620, "org_name": broker,
                 "quarter": "2026Q4", "eps": scale + change, "create_time": "2024-06-20 09:00:00"},
            ]
    return pd.DataFrame(rows)


def test_revision_matches_same_broker_and_same_fiscal_year():
    reports = _reports()
    reports.loc[len(reports)] = {
        "ts_code": "000001.SZ", "report_date": 20240621, "org_name": "甲证券",
        "quarter": "2027Q4", "eps": 99.0, "create_time": "2024-06-21 09:00:00",
    }
    pairs, stock = build_revision_components(reports, _targets())
    row = stock.set_index("ts_code").loc["000001.SZ"]
    assert row.revision_valid and row.matched_brokers == 2
    assert set(pairs.forecast_year) == {2026}
    assert 99.0 not in pairs.eps_current.tolist()


def test_revision_excludes_future_reports_and_uses_latest_asof():
    reports = _reports()
    reports.loc[len(reports)] = {
        "ts_code": "000001.SZ", "report_date": 20240701, "org_name": "甲证券",
        "quarter": "2026Q4", "eps": 88.0, "create_time": "2024-07-01 09:00:00",
    }
    pairs, _ = build_revision_components(reports, _targets())
    assert 88.0 not in pairs.eps_current.tolist()


def test_revision_requires_two_matched_brokers_and_one_new_forecast():
    reports = _reports()
    one_broker = reports.loc[~reports.org_name.str.startswith("乙")].copy()
    _, stock = build_revision_components(one_broker, _targets())
    assert not stock.revision_valid.any()
    stale = reports.loc[reports.report_date.eq(20240320)].copy()
    _, stock = build_revision_components(stale, _targets())
    assert not stock.revision_valid.any()
    assert stock.revision_reason.eq("NO_NEW_FORECAST_IN_90D").all()


def test_symmetric_revision_is_bounded_and_breadth_is_signed():
    pairs, stock = build_revision_components(_reports(), _targets())
    assert pairs.symmetric_revision.between(-2, 2).all()
    expected = ((2 * 0.2 / 2.2) + (2 * -0.1 / 1.9)) / 2
    row = stock.set_index("ts_code").loc["000001.SZ"]
    assert row.revision_magnitude == pytest.approx(expected)
    assert row.revision_breadth == pytest.approx(0.0)


def test_revision_score_is_fixed_average_of_component_percentile_ranks():
    _, stock = build_revision_components(_reports(), _targets())
    scored = form_revision_score(stock)
    assert scored.analyst_revision_score.between(0, 1).all()
    assert np.allclose(
        scored.analyst_revision_score,
        0.5 * (scored.revision_magnitude_rank + scored.revision_breadth_rank),
    )
