"""Pre-registered PBROE refinement transforms for Stage 7H."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_level_change_score(
    panel: pd.DataFrame,
    source_column: str,
    output_column: str,
    signal_column: str = "signal_date",
    stock_column: str = "ts_code",
) -> pd.DataFrame:
    """Add the frozen 50/50 level plus consecutive-month change score.

    Rows without an observation at the immediately preceding scheduled signal
    month receive a missing score and therefore do not enter R1 onward.
    """
    required = {signal_column, stock_column, source_column}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"level/change inputs missing columns: {sorted(missing)}")
    if panel.duplicated([signal_column, stock_column]).any():
        raise ValueError("signal-date/stock keys must be unique")

    out = panel.copy()
    dates = pd.Index(sorted(out[signal_column].dropna().unique()))
    previous = dict(zip(dates[1:], dates[:-1]))
    out["level_rank"] = out.groupby(signal_column, observed=True)[source_column].rank(
        method="average", pct=True
    )
    prior = out[[signal_column, stock_column, "level_rank"]].rename(
        columns={signal_column: "prior_signal_date", "level_rank": "prior_level_rank"}
    )
    out["prior_signal_date"] = out[signal_column].map(previous)
    out = out.merge(
        prior,
        on=["prior_signal_date", stock_column],
        how="left",
        validate="many_to_one",
    )
    out["level_change"] = out["level_rank"] - out["prior_level_rank"]
    out["change_rank"] = out.groupby(signal_column, observed=True)["level_change"].rank(
        method="average", pct=True
    )
    out[output_column] = 0.5 * out["level_rank"] + 0.5 * out["change_rank"]
    return out


def leave_one_out_peer_clip(
    frame: pd.DataFrame,
    value_column: str = "steady_roe",
    hierarchy: tuple[tuple[str, int], ...] = (
        ("l3_code", 10), ("l2_code", 20), ("l1_code", 30)
    ),
    market_minimum: int = 50,
    standard_deviations: float = 2.0,
    absolute_ceiling: float = 0.50,
) -> pd.DataFrame:
    """Clip positive values to leave-one-out peer mean +/- k sample std.

    The first sufficiently populated industry level is used, followed by the
    whole eligible market. Arithmetic means are used throughout.
    """
    required = {"ts_code", value_column, *(name for name, _ in hierarchy)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"peer-clip inputs missing columns: {sorted(missing)}")
    if frame.ts_code.duplicated().any():
        raise ValueError("ts_code must be unique within a signal month")
    values = pd.to_numeric(frame[value_column], errors="coerce").astype(float)
    if values.isna().any() or (~np.isfinite(values)).any() or values.le(0).any():
        raise ValueError("peer clipping requires finite positive values")

    out = frame.copy()
    n = len(out)
    chosen_mean = np.full(n, np.nan)
    chosen_std = np.full(n, np.nan)
    chosen_count = np.zeros(n, dtype=int)
    chosen_level = np.full(n, "", dtype=object)

    for column, minimum_other_peers in hierarchy:
        grouped = out.groupby(column, dropna=False, observed=True)[value_column]
        count = grouped.transform("count").to_numpy(dtype=float)
        total = grouped.transform("sum").to_numpy(dtype=float)
        sumsq = grouped.transform(lambda x: np.square(x.astype(float)).sum()).to_numpy(dtype=float)
        others = count - 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = (total - values.to_numpy()) / others
            # Sample variance across the leave-one-out peers.
            variance = (
                (sumsq - np.square(values.to_numpy()))
                - others * np.square(mean)
            ) / (others - 1.0)
        std = np.sqrt(np.maximum(variance, 0.0))
        eligible = (
            np.isnan(chosen_mean)
            & out[column].notna().to_numpy()
            & (others >= minimum_other_peers)
            & np.isfinite(mean)
            & np.isfinite(std)
            & (std > 0)
        )
        chosen_mean[eligible] = mean[eligible]
        chosen_std[eligible] = std[eligible]
        chosen_count[eligible] = others[eligible].astype(int)
        chosen_level[eligible] = column

    market_count = n - 1
    market_mean = (values.sum() - values) / market_count
    market_var = (
        (np.square(values).sum() - np.square(values))
        - market_count * np.square(market_mean)
    ) / (market_count - 1)
    market_std = np.sqrt(np.maximum(market_var, 0.0))
    fallback = (
        np.isnan(chosen_mean)
        & (market_count >= market_minimum)
        & np.isfinite(market_mean)
        & np.isfinite(market_std)
        & (market_std > 0)
    )
    chosen_mean[fallback] = market_mean.to_numpy()[fallback]
    chosen_std[fallback] = market_std.to_numpy()[fallback]
    chosen_count[fallback] = market_count
    chosen_level[fallback] = "market"
    if np.isnan(chosen_mean).any():
        raise ValueError("no valid peer fallback for some rows")

    lower = chosen_mean - standard_deviations * chosen_std
    upper = chosen_mean + standard_deviations * chosen_std
    clipped = np.clip(values.to_numpy(), lower, upper)
    clipped = np.minimum(clipped, absolute_ceiling)
    out["peer_mean_roe"] = chosen_mean
    out["peer_std_roe"] = chosen_std
    out["peer_count_roe"] = chosen_count
    out["peer_level_roe"] = chosen_level
    out["steady_roe_r2"] = clipped
    out["roe_was_transformed"] = ~np.isclose(clipped, values.to_numpy(), rtol=0, atol=1e-15)
    return out


def add_confidence_score(
    panel: pd.DataFrame,
    source_column: str = "r2_score",
    dispersion_column: str = "eps_dispersion",
    sensitivity_column: str = "growth_sensitivity",
    output_column: str = "r3_score",
    signal_column: str = "signal_date",
) -> pd.DataFrame:
    """Shrink a centered score using low-dispersion/low-sensitivity confidence."""
    required = {signal_column, source_column, dispersion_column, sensitivity_column}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"confidence inputs missing columns: {sorted(missing)}")
    out = panel.copy()
    dispersion = pd.to_numeric(out[dispersion_column], errors="coerce")
    sensitivity = pd.to_numeric(out[sensitivity_column], errors="coerce")
    out["dispersion_confidence_rank"] = (-dispersion).groupby(out[signal_column]).rank(
        method="average", pct=True
    )
    # Invalid perturbations are pre-registered as worst confidence.
    sensitivity_worst = sensitivity.fillna(np.inf)
    out["sensitivity_confidence_rank"] = (-sensitivity_worst).groupby(
        out[signal_column]
    ).rank(method="average", pct=True)
    confidence = 0.5 * (
        out["dispersion_confidence_rank"] + out["sensitivity_confidence_rank"]
    )
    confidence = confidence.where(dispersion.notna(), 0.0)
    out["r3_confidence"] = confidence
    out[output_column] = 0.5 + (out[source_column] - 0.5) * (0.5 + 0.5 * confidence)
    return out
