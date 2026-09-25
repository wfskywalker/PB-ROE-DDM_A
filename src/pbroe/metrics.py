"""Risk-adjusted and monotonicity metrics for monthly factor portfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_summary(
    returns: pd.Series,
    periods_per_year: int = 12,
    annual_risk_free_rate: float = 0.0,
) -> dict:
    values = pd.to_numeric(returns, errors="coerce").dropna().astype(float)
    if values.empty:
        raise ValueError("returns cannot be empty")
    if not np.isfinite(values).all() or values.le(-1).any():
        raise ValueError("returns must be finite and greater than -100%")
    wealth = pd.concat([pd.Series([1.0]), (1.0 + values).cumprod()], ignore_index=True)
    years = len(values) / periods_per_year
    cumulative = wealth.iloc[-1] - 1.0
    cagr = wealth.iloc[-1] ** (1.0 / years) - 1.0
    volatility = values.std(ddof=1) * np.sqrt(periods_per_year)
    periodic_rf = (1.0 + annual_risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = values - periodic_rf
    excess_std = excess.std(ddof=1)
    sharpe = (
        np.sqrt(periods_per_year) * excess.mean() / excess_std
        if np.isfinite(excess_std) and excess_std > 0 else np.nan
    )
    drawdown = wealth / wealth.cummax() - 1.0
    maximum_drawdown = drawdown.min()
    calmar = cagr / abs(maximum_drawdown) if maximum_drawdown < 0 else np.nan
    return {
        "months": int(len(values)),
        "cumulative_return": float(cumulative),
        "compound_annual_return": float(cagr),
        "arithmetic_annual_return": float(values.mean() * periods_per_year),
        "annualized_volatility": float(volatility),
        "sharpe_zero_rf": float(sharpe),
        "maximum_drawdown": float(maximum_drawdown),
        "calmar": float(calmar),
        "positive_month_ratio": float(values.gt(0).mean()),
    }


def monotonicity_summary(group_returns: pd.DataFrame) -> dict:
    """Evaluate increasing group-return monotonicity for columns ordered low-high."""
    wide = group_returns.copy().sort_index(axis=1)
    if wide.shape[1] < 2:
        raise ValueError("at least two groups are required")
    if wide.isna().any().any():
        raise ValueError("group returns must be complete")
    groups = pd.Series(wide.columns.astype(float), index=wide.columns)
    means = wide.mean(axis=0)
    full_spearman = groups.rank().corr(means.rank())
    adjacent = np.diff(means.to_numpy(dtype=float)) > 0
    monthly_spearman = wide.apply(
        lambda row: groups.rank().corr(row.rank()), axis=1
    )
    return {
        "groups": int(wide.shape[1]),
        "months": int(len(wide)),
        "full_history_group_spearman": float(full_spearman),
        "adjacent_pair_hit_ratio": float(adjacent.mean()),
        "adjacent_pairs_in_order": int(adjacent.sum()),
        "mean_monthly_group_spearman": float(monthly_spearman.mean()),
        "positive_monthly_group_spearman_ratio": float(monthly_spearman.gt(0).mean()),
    }
