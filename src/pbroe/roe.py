"""Point-in-time annual ROE selection and new-listing industry shrinkage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class RoeError(ValueError):
    pass


@dataclass(frozen=True)
class RoeResult:
    signal_date: int
    audit: pd.DataFrame
    eligible: pd.DataFrame
    annual_selected: pd.DataFrame
    funnel: dict[str, int]


def build_steady_roe(
    indicators: pd.DataFrame,
    universe: pd.DataFrame,
    signal_date: int,
    history_years: int = 3,
    min_peer_count: int = 10,
    winsor_limits: tuple[float, float] = (0.01, 0.99),
) -> RoeResult:
    required_fin = {"ts_code", "ann_date", "end_date", "roe", "roe_waa"}
    required_uni = {"ts_code", "l1_code", "l2_code", "l3_code"}
    missing_fin = required_fin - set(indicators.columns)
    missing_uni = required_uni - set(universe.columns)
    if missing_fin:
        raise RoeError(f"financial indicators missing columns: {sorted(missing_fin)}")
    if missing_uni:
        raise RoeError(f"universe missing columns: {sorted(missing_uni)}")
    if history_years != 3:
        raise RoeError("V1 history_years must equal 3")
    if min_peer_count <= 0:
        raise RoeError("min_peer_count must be positive")

    fin = indicators.copy()
    fin["ts_code"] = fin["ts_code"].astype("string").str.upper()
    for column in ("ann_date", "end_date", "roe", "roe_waa"):
        fin[column] = pd.to_numeric(fin[column], errors="coerce")
    fin = fin.loc[
        fin["ann_date"].le(signal_date)
        & fin["end_date"].astype("Int64").astype("string").str.endswith("1231")
    ].copy()
    fin.sort_values(["ts_code", "end_date", "ann_date"], inplace=True)
    fin = fin.drop_duplicates(["ts_code", "end_date"], keep="last")
    fin.sort_values(["ts_code", "end_date"], ascending=[True, False], inplace=True)
    fin["history_rank"] = fin.groupby("ts_code", observed=True).cumcount() + 1
    selected = fin.loc[fin["history_rank"].le(history_years)].copy()

    rows = []
    for ts_code, group in selected.groupby("ts_code", observed=True):
        group = group.sort_values("end_date", ascending=False).head(history_years)
        waa_complete = group["roe_waa"].notna().all()
        metric = "roe_waa" if waa_complete else "roe"
        values = group[metric].dropna()
        values = values[np.isfinite(values)]
        rows.append(
            {
                "ts_code": ts_code,
                "roe_metric": metric,
                "roe_observation_count": len(values),
                "company_roe_mean": values.mean() / 100.0 if len(values) else np.nan,
                "latest_roe_ann_date": group.loc[group[metric].notna(), "ann_date"].max(),
                "roe_history_end_dates": ",".join(
                    group.loc[group[metric].notna(), "end_date"].astype(int).astype(str)
                ),
            }
        )
    company = pd.DataFrame(rows)

    base = universe.copy()
    base["ts_code"] = base["ts_code"].astype("string").str.upper()
    if base["ts_code"].duplicated().any():
        raise RoeError("universe contains duplicate ts_code values")
    audit = base.merge(company, on="ts_code", how="left", validate="one_to_one")
    audit["roe_observation_count"] = audit["roe_observation_count"].fillna(0).astype(int)

    peer_source = audit.loc[
        audit["roe_observation_count"].eq(history_years)
        & audit["company_roe_mean"].notna()
    ].copy()
    priors = _build_industry_priors(audit, peer_source, min_peer_count, winsor_limits)
    audit = audit.merge(priors, on="ts_code", how="left", validate="one_to_one")

    needs_prior = audit["roe_observation_count"].lt(history_years)
    has_prior = audit["industry_roe_prior"].notna()
    weight = audit["roe_observation_count"] / history_years
    audit["steady_roe"] = np.where(
        needs_prior & has_prior,
        weight * audit["company_roe_mean"].fillna(0.0)
        + (1.0 - weight) * audit["industry_roe_prior"],
        np.where(~needs_prior, audit["company_roe_mean"], np.nan),
    )
    audit["roe_reason"] = np.select(
        [needs_prior & ~has_prior, audit["steady_roe"].le(0)],
        ["MISSING_INDUSTRY_PRIOR", "NON_POSITIVE_STEADY_ROE"],
        default="VALID",
    )
    audit["roe_valid"] = audit["roe_reason"].eq("VALID") & audit["steady_roe"].notna()
    eligible = audit.loc[audit["roe_valid"]].copy().reset_index(drop=True)
    funnel = {
        "universe_stocks": len(audit),
        "stocks_with_any_roe": int(audit["roe_observation_count"].gt(0).sum()),
        "stocks_with_three_years": int(audit["roe_observation_count"].eq(3).sum()),
        "stocks_using_industry_shrinkage": int((needs_prior & has_prior).sum()),
        "valid_positive_steady_roe": len(eligible),
    }
    return RoeResult(int(signal_date), audit, eligible, selected.reset_index(drop=True), funnel)


def _build_industry_priors(
    targets: pd.DataFrame,
    peers: pd.DataFrame,
    min_peer_count: int,
    winsor_limits: tuple[float, float],
) -> pd.DataFrame:
    output = targets[["ts_code"]].copy()
    output["industry_roe_prior"] = np.nan
    output["industry_prior_level"] = pd.NA
    output["industry_peer_count"] = 0
    needs_prior = targets["roe_observation_count"].lt(3)
    for level, column in ((3, "l3_code"), (2, "l2_code"), (1, "l1_code")):
        unresolved = output["industry_roe_prior"].isna() & needs_prior
        for ts_code in output.loc[unresolved, "ts_code"]:
            row = targets.loc[targets["ts_code"].eq(ts_code)].iloc[0]
            candidates = peers.loc[
                peers[column].eq(row[column]) & ~peers["ts_code"].eq(ts_code),
                "company_roe_mean",
            ].dropna()
            if len(candidates) < min_peer_count:
                continue
            low, high = candidates.quantile(list(winsor_limits)).tolist()
            prior = candidates.clip(lower=low, upper=high).mean()
            mask = output["ts_code"].eq(ts_code)
            output.loc[mask, "industry_roe_prior"] = prior
            output.loc[mask, "industry_prior_level"] = f"SW{level}"
            output.loc[mask, "industry_peer_count"] = len(candidates)

    return output
