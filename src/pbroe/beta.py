"""Trailing-only market beta estimation."""

import numpy as np
import pandas as pd


def estimate_trailing_beta(
    prices: pd.DataFrame, window_days: int = 252, min_observations: int = 120
) -> pd.DataFrame:
    required = {"trade_date", "ts_code", "close", "adj_factor"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"beta prices missing columns: {sorted(missing)}")
    x = prices.copy().sort_values(["ts_code", "trade_date"])
    x["adjusted_close"] = pd.to_numeric(x.close, errors="coerce") * pd.to_numeric(
        x.adj_factor, errors="coerce"
    )
    x["stock_return"] = x.groupby("ts_code", observed=True).adjusted_close.pct_change(
        fill_method=None
    )
    x["market_return"] = x.groupby("trade_date", observed=True).stock_return.transform("mean")
    x = x.dropna(subset=["stock_return", "market_return"])
    x = x.groupby("ts_code", observed=True, group_keys=False).tail(window_days)
    rows = []
    for code, group in x.groupby("ts_code", observed=True):
        n = len(group)
        variance = group.market_return.var(ddof=1)
        beta = (
            group[["stock_return", "market_return"]].cov().iloc[0, 1] / variance
            if n >= min_observations and variance > 0
            else np.nan
        )
        rows.append(
            {
                "ts_code": code,
                "beta": beta,
                "beta_observations": n,
                "beta_start_date": int(group.trade_date.min()),
                "beta_end_date": int(group.trade_date.max()),
            }
        )
    return pd.DataFrame(rows)
