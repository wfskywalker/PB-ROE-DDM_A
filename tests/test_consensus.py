import pandas as pd
import pytest

from pbroe.consensus import (
    ConsensusError,
    build_consensus_asof,
    forecast_fiscal_year,
    manifest_record,
    normalize_broker_name,
    validate_download_manifest,
)


def reports_fixture():
    rows = []
    for year in (2024, 2025, 2026):
        rows.extend(
            [
                {
                    "ts_code": "000001.SZ",
                    "report_date": 20240110,
                    "org_name": "甲证券股份有限公司",
                    "quarter": f"{year}Q4",
                    "eps": float(year - 2023),
                    "np": float(year * 10),
                    "create_time": "2024-01-10 10:00:00",
                },
                {
                    "ts_code": "000001.SZ",
                    "report_date": 20240112,
                    "org_name": "乙证券研究所",
                    "quarter": f"{year}Q4",
                    "eps": float(year - 2023) + 0.2,
                    "np": float(year * 10 + 2),
                    "create_time": "2024-01-12 10:00:00",
                },
            ]
        )
    rows.extend(
        [
            {
                "ts_code": "000001.SZ",
                "report_date": 20240120,
                "org_name": "甲证券",
                "quarter": "2024Q4",
                "eps": 1.4,
                "np": 20244.0,
                "create_time": "2024-01-20 10:00:00",
            },
            {
                "ts_code": "000001.SZ",
                "report_date": 20240201,
                "org_name": "甲证券",
                "quarter": "2024Q4",
                "eps": 99.0,
                "np": 99.0,
                "create_time": "2024-02-01 10:00:00",
            },
            {
                "ts_code": "000002.SZ",
                "report_date": 20240110,
                "org_name": "甲证券",
                "quarter": "2024Q4",
                "eps": 1.0,
                "np": 1.0,
                "create_time": "2024-01-10 10:00:00",
            },
        ]
    )
    return pd.DataFrame(rows)


def test_broker_and_fiscal_year_normalization():
    assert normalize_broker_name(" 甲证券股份有限公司 ") == "甲证券"
    assert normalize_broker_name("甲 证券研究所") == "甲证券"
    assert forecast_fiscal_year("2026Q4") == 2026
    assert forecast_fiscal_year("FY2027") == 2027
    assert forecast_fiscal_year("unknown") is None


def test_consensus_is_point_in_time_and_latest_per_broker():
    result = build_consensus_asof(reports_fixture(), 20240131)
    assert result.eligible["ts_code"].tolist() == ["000001.SZ"]
    row = result.eligible.iloc[0]
    assert row["forecast_year_fy1"] == 2024
    assert row["forecast_year_fy2"] == 2025
    assert row["forecast_year_fy3"] == 2026
    assert row["broker_count_fy1"] == 2
    assert row["eps_fy1"] == pytest.approx((1.4 + 1.2) / 2)
    assert row["terminal_forecast_year"] == 2026
    assert row["terminal_broker_count"] == 2
    assert 99.0 not in result.latest_broker_forecasts["eps"].tolist()


def test_insufficient_brokers_is_reason_coded():
    result = build_consensus_asof(reports_fixture(), 20240131)
    row = result.audit.set_index("ts_code").loc["000002.SZ"]
    assert not row["consensus_valid"]
    assert row["consensus_reason"] == "NO_ELIGIBLE_TERMINAL_FORECAST"


def test_vendor_create_time_does_not_override_report_availability_date():
    reports = reports_fixture()
    reports.loc[len(reports)] = {
        "ts_code": "000001.SZ",
        "report_date": 20240115,
        "org_name": "丙证券",
        "quarter": "2024Q4",
        "eps": 88.0,
        "np": 88.0,
        "create_time": "2024-02-15 10:00:00",
    }
    result = build_consensus_asof(reports, 20240131)
    assert "丙证券" in result.latest_broker_forecasts["broker"].tolist()


def test_future_report_date_is_excluded():
    reports = reports_fixture()
    result = build_consensus_asof(reports, 20240131)
    assert 99.0 not in result.latest_broker_forecasts["eps"].tolist()


def test_empty_asof_window_returns_typed_empty_result():
    reports = reports_fixture()
    result = build_consensus_asof(reports, 20150130)
    assert result.audit.empty
    assert result.eligible.empty
    assert str(result.audit["ts_code"].dtype) == "string"
    assert result.funnel["valid_consensus_stocks"] == 0


def test_december_signal_keeps_signal_year_as_fy1():
    result = build_consensus_asof(reports_fixture(), 20241231, lookback_days=365)
    row = result.audit.set_index("ts_code").loc["000001.SZ"]
    assert [row[f"forecast_year_fy{i}"] for i in (1, 2, 3)] == [2024, 2025, 2026]
    assert row["terminal_forecast_year"] == 2026


def test_terminal_year_falls_back_and_uses_actual_available_horizon():
    result = build_consensus_asof(reports_fixture(), 20250131, lookback_days=400)
    row = result.eligible.set_index("ts_code").loc["000001.SZ"]
    assert row["terminal_forecast_year"] == 2026
    assert row["terminal_broker_count"] == 2


def test_manifest_counts_empty_dates_as_complete():
    manifest = pd.DataFrame(
        [manifest_record(20240101, 0), manifest_record(20240102, 3)]
    )
    summary = validate_download_manifest(manifest, [20240101, 20240102])
    assert summary == {
        "expected_dates": 2,
        "completed_dates": 2,
        "success_dates": 1,
        "empty_dates": 1,
        "downloaded_rows": 3,
    }


def test_manifest_rejects_missing_and_failed_dates():
    manifest = pd.DataFrame([manifest_record(20240101, 0)])
    with pytest.raises(ConsensusError, match="missing 1 dates"):
        validate_download_manifest(manifest, [20240101, 20240102])

    failed = pd.DataFrame([manifest_record(20240101, 0, error="network")])
    with pytest.raises(ConsensusError, match="failed dates"):
        validate_download_manifest(failed, [20240101])
