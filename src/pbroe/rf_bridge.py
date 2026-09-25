"""Point-in-time bridge from monthly PBROE signals to daily RF rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["signal_date", "ts_code"]
VALUE_COLUMNS = ["pbroe_iar_neutralized", "analyst_revision_score"]
AVAILABILITY_COLUMNS = ["pbroe_available", "analyst_revision_available"]


def _validate_unique(frame: pd.DataFrame, name: str) -> None:
    missing = set(KEYS).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing keys: {sorted(missing)}")
    if frame.duplicated(KEYS).any():
        raise ValueError(f"{name} has duplicate signal_date/ts_code keys")


def next_trading_date_map(signal_dates: pd.Series, trading_dates: pd.Series) -> pd.Series:
    """Map each signal close to the strictly following trading date."""
    signals = pd.Series(signal_dates, copy=False).astype("int64")
    calendar = np.unique(pd.Series(trading_dates, copy=False).astype("int64"))
    positions = np.searchsorted(calendar, signals.to_numpy(), side="right")
    if (positions >= len(calendar)).any():
        missing = sorted(signals.loc[positions >= len(calendar)].unique().tolist())
        raise ValueError(f"no next trading date for signal dates: {missing}")
    return pd.Series(calendar[positions], index=signals.index, dtype="int64")


def build_monthly_effective_events(
    universe: pd.DataFrame,
    pbroe: pd.DataFrame,
    revision: pd.DataFrame,
    trading_dates: pd.Series,
) -> pd.DataFrame:
    """Create complete monthly reset events for every RF-universe stock.

    A complete universe row is required each month so a factor that becomes
    unavailable resets to value=0, availability=0 instead of carrying forward
    its stale observation.
    """
    _validate_unique(universe, "universe")
    _validate_unique(pbroe, "pbroe")
    _validate_unique(revision, "revision")
    if "pbroe_iar_neutralized" not in pbroe:
        raise ValueError("pbroe missing pbroe_iar_neutralized")
    if "analyst_revision_score" not in revision:
        raise ValueError("revision missing analyst_revision_score")

    out = universe[KEYS].copy()
    out["signal_date"] = out["signal_date"].astype("int64")
    out["ts_code"] = out["ts_code"].astype("string").str.upper()
    p = pbroe[KEYS + ["pbroe_iar_neutralized"]].copy()
    r = revision[KEYS + ["analyst_revision_score"]].copy()
    for frame in (p, r):
        frame["signal_date"] = frame["signal_date"].astype("int64")
        frame["ts_code"] = frame["ts_code"].astype("string").str.upper()
    out = out.merge(p, on=KEYS, how="left", validate="one_to_one")
    out = out.merge(r, on=KEYS, how="left", validate="one_to_one")

    for value, availability in zip(VALUE_COLUMNS, AVAILABILITY_COLUMNS, strict=True):
        finite = np.isfinite(pd.to_numeric(out[value], errors="coerce"))
        out[availability] = finite.astype("int8")
        out[value] = pd.to_numeric(out[value], errors="coerce").where(finite, 0.0).astype("float64")

    effective_by_signal = pd.DataFrame({"signal_date": sorted(out.signal_date.unique())})
    effective_by_signal["effective_date"] = next_trading_date_map(
        effective_by_signal["signal_date"], trading_dates
    )
    out = out.merge(effective_by_signal, on="signal_date", how="left", validate="many_to_one")
    out = out.rename(columns={"effective_date": "execution_date"})
    return out[["signal_date", "execution_date", "ts_code", *VALUE_COLUMNS, *AVAILABILITY_COLUMNS]].sort_values(
        ["signal_date", "ts_code"], ignore_index=True
    )
