import pandas as pd

from pbroe.factor import generate_factor_snapshot


def test_missing_consensus_boolean_is_not_truthy():
    universe = pd.DataFrame({"ts_code": ["A.SZ"], "l3_code": ["L3"]})
    consensus = pd.DataFrame(columns=["ts_code", "consensus_valid", "terminal_forecast_year", "terminal_eps"])
    roe = pd.DataFrame({"ts_code": ["A.SZ"], "roe_valid": [True], "steady_roe": [.2], "roe_reason": ["VALID"]})
    beta = pd.DataFrame({"ts_code": ["A.SZ"], "beta": [1.0]})
    prices = pd.DataFrame({"ts_code": ["A.SZ"], "trade_date": [20250131], "close": [10.]})
    growth = pd.DataFrame({"l3_code": ["L3"], "perpetual_growth": [.03], "policy_version": ["v1"], "effective_from": [20200101]})
    result = generate_factor_snapshot(universe, consensus, roe, beta, prices, growth, 20250131, .017, .055)
    assert result.loc[0, "reason_code"] == "INVALID_CONSENSUS"
    assert not result.loc[0, "factor_valid"]
