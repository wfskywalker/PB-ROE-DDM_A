"""Deterministic monthly cross-sectional neutralization for PBROE_IAR."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _winsorized_zscore(series: pd.Series, lower: float, upper: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    lo, hi = values.quantile([lower, upper])
    clipped = values.clip(lo, hi)
    scale = clipped.std(ddof=0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"cannot standardize constant or invalid column {series.name!r}")
    return (clipped - clipped.mean()) / scale


def neutralize_month(
    frame: pd.DataFrame,
    factor_column: str = "pbroe_implied_ann_return",
    industry_column: str = "l1_name",
    market_value_column: str = "total_mv",
    beta_column: str = "beta",
    output_column: str = "pbroe_iar_neutralized",
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
) -> tuple[pd.DataFrame, dict]:
    """Residualize one signal-month factor cross-section.

    The regression uses an intercept, standardized log market value,
    standardized Beta, and level-1 industry fixed effects. The returned frame
    preserves the input rows and their order.
    """
    required = {
        "ts_code", factor_column, industry_column, market_value_column, beta_column
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"neutralization inputs missing columns: {sorted(missing)}")
    if not 0 <= winsor_lower < winsor_upper <= 1:
        raise ValueError("winsor limits must satisfy 0 <= lower < upper <= 1")
    if frame.empty:
        raise ValueError("cannot neutralize an empty cross-section")
    if frame.ts_code.duplicated().any():
        raise ValueError("ts_code must be unique within a signal month")

    x = frame.copy()
    numeric = x[[factor_column, market_value_column, beta_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = (
        numeric.notna().all(axis=1)
        & np.isfinite(numeric).all(axis=1)
        & numeric[market_value_column].gt(0)
        & x[industry_column].notna()
    )
    if not valid.all():
        bad = x.loc[~valid, "ts_code"].astype(str).tolist()[:5]
        raise ValueError(f"invalid neutralization rows, examples: {bad}")

    y = _winsorized_zscore(numeric[factor_column], winsor_lower, winsor_upper)
    log_mv = np.log(numeric[market_value_column])
    mv_z = _winsorized_zscore(log_mv.rename("log_total_mv"), winsor_lower, winsor_upper)
    beta_z = _winsorized_zscore(
        numeric[beta_column], winsor_lower, winsor_upper
    )

    industry = x[industry_column].astype(str)
    categories = sorted(industry.unique())
    industry = pd.Categorical(industry, categories=categories, ordered=True)
    dummies = pd.get_dummies(industry, prefix="industry", dtype=float)
    if len(dummies.columns) > 1:
        dummies = dummies.iloc[:, 1:]
    else:
        dummies = dummies.iloc[:, :0]

    design = pd.DataFrame(
        {"intercept": 1.0, "log_total_mv_z": mv_z, "beta_z": beta_z},
        index=x.index,
    ).join(dummies.set_axis(x.index))
    matrix = design.to_numpy(dtype=float)
    response = y.to_numpy(dtype=float)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        matrix, response, rcond=None
    )
    residual = response - matrix @ coefficients
    x[output_column] = residual
    x[f"{factor_column}_winsorized_z"] = response
    x["log_total_mv_z"] = mv_z.to_numpy()
    x["beta_z"] = beta_z.to_numpy()

    moments = matrix.T @ residual / len(x)
    smallest = singular_values[-1] if len(singular_values) else np.nan
    condition = (
        singular_values[0] / smallest
        if np.isfinite(smallest) and smallest > 0
        else np.inf
    )
    diagnostics = {
        "rows": int(len(x)),
        "industries": int(len(categories)),
        "design_columns": int(matrix.shape[1]),
        "design_rank": int(rank),
        "condition_number": float(condition),
        "max_abs_normal_equation": float(np.max(np.abs(moments))),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "reference_industry": categories[0],
    }
    return x, diagnostics


def neutralize_panel(
    panel: pd.DataFrame,
    signal_column: str = "signal_date",
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Neutralize every month independently and return monthly diagnostics."""
    if signal_column not in panel:
        raise ValueError(f"missing signal column {signal_column!r}")
    parts: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    for signal_date, month in panel.groupby(signal_column, sort=True):
        result, audit = neutralize_month(month, **kwargs)
        parts.append(result)
        diagnostics.append({signal_column: signal_date, **audit})
    output = pd.concat(parts).sort_index()
    return output, pd.DataFrame(diagnostics)
