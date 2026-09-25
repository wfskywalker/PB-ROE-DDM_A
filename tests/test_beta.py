import numpy as np
import pandas as pd

from pbroe.beta import estimate_trailing_beta


def test_beta_uses_only_provided_history_and_recovers_loading():
    dates = pd.date_range("2024-01-01", periods=150, freq="B")
    market = np.linspace(-0.01, 0.01, len(dates))
    base = 100 * np.cumprod(1 + market)
    rows = []
    for code, loading in [("A.SZ", 1.0), ("B.SZ", 2.0)]:
        path = 100 * np.cumprod(1 + loading * market)
        for date, close in zip(dates, path):
            rows.append({"trade_date": int(date.strftime("%Y%m%d")), "ts_code": code, "close": close, "adj_factor": 1.0})
    result = estimate_trailing_beta(pd.DataFrame(rows), 120, 100).set_index("ts_code")
    assert result.loc["B.SZ", "beta"] > result.loc["A.SZ", "beta"]
    assert result.beta_end_date.max() == int(dates[-1].strftime("%Y%m%d"))
