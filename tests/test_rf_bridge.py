import pandas as pd
import pytest

from pbroe.rf_bridge import build_monthly_effective_events, next_trading_date_map


def test_signal_becomes_effective_strictly_next_trading_day():
    mapped = next_trading_date_map(
        pd.Series([20240131, 20240202]),
        pd.Series([20240131, 20240201, 20240202, 20240205]),
    )
    assert mapped.tolist() == [20240201, 20240205]


def test_complete_monthly_events_reset_missing_factor_instead_of_carrying_it():
    universe = pd.DataFrame({
        "signal_date": [20240131, 20240131, 20240229, 20240229],
        "ts_code": ["000001.SZ", "600000.SH", "000001.SZ", "600000.SH"],
    })
    pbroe = pd.DataFrame({
        "signal_date": [20240131, 20240229],
        "ts_code": ["000001.SZ", "600000.SH"],
        "pbroe_iar_neutralized": [0.4, -0.2],
    })
    revision = pd.DataFrame({
        "signal_date": [20240131], "ts_code": ["600000.SH"],
        "analyst_revision_score": [0.7],
    })
    events = build_monthly_effective_events(
        universe, pbroe, revision,
        pd.Series([20240131, 20240201, 20240229, 20240301]),
    )
    feb_stock = events.loc[(events.signal_date == 20240229) & (events.ts_code == "000001.SZ")].iloc[0]
    assert feb_stock.execution_date == 20240301
    assert feb_stock.pbroe_iar_neutralized == 0.0
    assert feb_stock.pbroe_available == 0
    assert feb_stock.analyst_revision_score == 0.0
    assert feb_stock.analyst_revision_available == 0


def test_rf_feature_date_is_signal_date_and_execution_is_next_open():
    universe = pd.DataFrame({"signal_date": [20240131], "ts_code": ["000001.SZ"]})
    pbroe = pd.DataFrame({
        "signal_date": [20240131], "ts_code": ["000001.SZ"],
        "pbroe_iar_neutralized": [0.4],
    })
    revision = pd.DataFrame(columns=["signal_date", "ts_code", "analyst_revision_score"])
    event = build_monthly_effective_events(
        universe, pbroe, revision, pd.Series([20240131, 20240201])
    ).iloc[0]
    assert event.signal_date == 20240131
    assert event.execution_date == 20240201


def test_bridge_rejects_duplicate_monthly_keys():
    universe = pd.DataFrame({"signal_date": [20240131], "ts_code": ["000001.SZ"]})
    duplicated = pd.DataFrame({
        "signal_date": [20240131, 20240131],
        "ts_code": ["000001.SZ", "000001.SZ"],
        "pbroe_iar_neutralized": [1.0, 2.0],
    })
    revision = pd.DataFrame(columns=["signal_date", "ts_code", "analyst_revision_score"])
    with pytest.raises(ValueError, match="duplicate"):
        build_monthly_effective_events(universe, duplicated, revision, pd.Series([20240131, 20240201]))


def test_bridge_rejects_signal_without_future_trade_date():
    with pytest.raises(ValueError, match="no next trading date"):
        next_trading_date_map(pd.Series([20240131]), pd.Series([20240130, 20240131]))
