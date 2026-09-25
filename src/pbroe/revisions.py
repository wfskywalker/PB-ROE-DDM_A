"""Point-in-time analyst EPS revision construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .consensus import forecast_fiscal_year, normalize_broker_name


class RevisionError(ValueError):
    """Raised when analyst-revision inputs violate the frozen contract."""


def build_revision_components(
    reports: pd.DataFrame,
    targets: pd.DataFrame,
    revision_days: int = 90,
    current_max_age_days: int = 180,
    baseline_max_age_days: int = 365,
    min_matched_brokers: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair same-broker, same-fiscal-year EPS and aggregate stock components.

    ``targets`` contains one terminal fiscal year per stock and signal date.
    The function is intentionally transparent and is used for golden tests;
    the full-history acceptance job uses an equivalent DuckDB implementation.
    """
    required_reports = {"ts_code", "report_date", "org_name", "quarter", "eps"}
    required_targets = {"signal_date", "ts_code", "forecast_year"}
    if required_reports - set(reports):
        raise RevisionError("reports missing required revision columns")
    if required_targets - set(targets):
        raise RevisionError("targets missing required revision columns")
    if targets.duplicated(["signal_date", "ts_code"]).any():
        raise RevisionError("targets contain duplicate stock-month keys")
    if not (0 < revision_days < current_max_age_days <= baseline_max_age_days):
        raise RevisionError("invalid revision window ordering")
    if min_matched_brokers < 2:
        raise RevisionError("min_matched_brokers must be at least 2")

    normalized = reports.copy()
    normalized["ts_code"] = normalized.ts_code.astype("string").str.upper()
    normalized["report_date"] = pd.to_datetime(
        normalized.report_date.astype("string").str.replace("-", "", regex=False).str[:8],
        format="%Y%m%d", errors="coerce",
    )
    normalized["broker"] = normalized.org_name.map(normalize_broker_name).astype("string")
    normalized["forecast_year"] = normalized.quarter.map(forecast_fiscal_year).astype("Int64")
    normalized["eps"] = pd.to_numeric(normalized.eps, errors="coerce")
    normalized["create_ts"] = pd.to_datetime(
        normalized.get("create_time", pd.Series(pd.NaT, index=normalized.index)),
        errors="coerce",
    )

    pair_parts: list[pd.DataFrame] = []
    for signal_date, target_month in targets.groupby("signal_date", sort=True):
        signal_ts = pd.Timestamp(str(int(signal_date)))
        candidates = target_month[["ts_code", "forecast_year"]].merge(
            normalized,
            on=["ts_code", "forecast_year"],
            how="inner",
            validate="one_to_many",
        )
        candidates = candidates.loc[
            candidates.report_date.between(
                signal_ts - pd.Timedelta(days=baseline_max_age_days),
                signal_ts,
                inclusive="both",
            )
            & candidates.broker.notna()
            & candidates.eps.notna()
        ].copy()
        candidates.sort_values(
            ["ts_code", "broker", "forecast_year", "report_date", "create_ts"],
            na_position="first", inplace=True,
        )
        current = candidates.loc[
            candidates.report_date.ge(signal_ts - pd.Timedelta(days=current_max_age_days))
        ].drop_duplicates(["ts_code", "broker", "forecast_year"], keep="last")
        prior = candidates.loc[
            candidates.report_date.le(signal_ts - pd.Timedelta(days=revision_days))
        ].drop_duplicates(["ts_code", "broker", "forecast_year"], keep="last")
        current = current[["ts_code", "broker", "forecast_year", "report_date", "eps"]].rename(
            columns={"report_date": "current_report_date", "eps": "eps_current"}
        )
        prior = prior[["ts_code", "broker", "forecast_year", "report_date", "eps"]].rename(
            columns={"report_date": "prior_report_date", "eps": "eps_prior"}
        )
        pairs = current.merge(
            prior, on=["ts_code", "broker", "forecast_year"],
            how="inner", validate="one_to_one",
        )
        denominator = pairs.eps_current.abs() + pairs.eps_prior.abs()
        pairs["symmetric_revision"] = (
            2.0 * (pairs.eps_current - pairs.eps_prior) / denominator.replace(0, np.nan)
        )
        pairs["active"] = pairs.current_report_date.gt(pairs.prior_report_date)
        pairs.insert(0, "signal_date", int(signal_date))
        pair_parts.append(pairs.loc[pairs.symmetric_revision.notna()])

    pairs = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    if pairs.empty:
        empty = targets.copy()
        empty["matched_brokers"] = 0
        empty["active_brokers"] = 0
        empty["revision_valid"] = False
        empty["revision_reason"] = "NO_MATCHED_BROKER_YEAR"
        return pairs, empty

    components = pairs.groupby(
        ["signal_date", "ts_code", "forecast_year"], observed=True
    ).agg(
        matched_brokers=("broker", "size"),
        active_brokers=("active", "sum"),
        revision_magnitude=("symmetric_revision", "mean"),
        revision_breadth=("symmetric_revision", lambda x: np.sign(x).mean()),
        up_brokers=("symmetric_revision", lambda x: int((x > 0).sum())),
        down_brokers=("symmetric_revision", lambda x: int((x < 0).sum())),
        flat_brokers=("symmetric_revision", lambda x: int((x == 0).sum())),
        min_current_report_date=("current_report_date", "min"),
        max_current_report_date=("current_report_date", "max"),
        min_prior_report_date=("prior_report_date", "min"),
        max_prior_report_date=("prior_report_date", "max"),
    ).reset_index()
    output = targets.merge(
        components, on=["signal_date", "ts_code", "forecast_year"],
        how="left", validate="one_to_one",
    )
    for column in ("matched_brokers", "active_brokers", "up_brokers", "down_brokers", "flat_brokers"):
        output[column] = output[column].fillna(0).astype("int64")
    output["revision_valid"] = output.matched_brokers.ge(min_matched_brokers) & output.active_brokers.ge(1)
    output["revision_reason"] = np.select(
        [output.matched_brokers.eq(0), output.matched_brokers.lt(min_matched_brokers), output.active_brokers.lt(1)],
        ["NO_MATCHED_BROKER_YEAR", "FEWER_THAN_TWO_MATCHED_BROKERS", "NO_NEW_FORECAST_IN_90D"],
        default="VALID",
    )
    return pairs, output


def form_revision_score(stock_components: pd.DataFrame) -> pd.DataFrame:
    """Create the frozen 50/50 magnitude-breadth percentile-rank score."""
    required = {"signal_date", "ts_code", "revision_valid", "revision_magnitude", "revision_breadth"}
    if required - set(stock_components):
        raise RevisionError("stock components missing score columns")
    valid = stock_components.loc[stock_components.revision_valid].copy()
    if valid.empty:
        return valid
    valid["revision_magnitude_rank"] = valid.groupby("signal_date", observed=True)[
        "revision_magnitude"
    ].rank(method="average", pct=True)
    valid["revision_breadth_rank"] = valid.groupby("signal_date", observed=True)[
        "revision_breadth"
    ].rank(method="average", pct=True)
    valid["analyst_revision_score"] = 0.5 * (
        valid.revision_magnitude_rank + valid.revision_breadth_rank
    )
    return valid.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)
