"""No-stage-2 PB-ROE valuation kernel and growth mapping."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValuationResult:
    implied_ann_return: float | None
    target_price: float | None
    justified_pb: float | None
    justified_pe: float | None
    horizon_years: float | None
    reason: str

    def to_dict(self):
        return asdict(self)


def value_pbroe(
    signal_date: int,
    fy3_year: int,
    close: float,
    eps_fy3: float,
    steady_roe: float,
    cost_of_equity: float,
    perpetual_growth: float,
    min_coe_growth_spread: float = 0.02,
) -> ValuationResult:
    values = [close, eps_fy3, steady_roe, cost_of_equity, perpetual_growth]
    if not all(np.isfinite(value) for value in values):
        return _invalid("NON_FINITE_INPUT")
    if close <= 0:
        return _invalid("NON_POSITIVE_PRICE")
    if eps_fy3 <= 0:
        return _invalid("NON_POSITIVE_EPS")
    if steady_roe <= 0:
        return _invalid("NON_POSITIVE_STEADY_ROE")
    spread = cost_of_equity - perpetual_growth
    if spread < min_coe_growth_spread:
        return _invalid("COE_G_SPREAD_TOO_SMALL")
    endpoint = pd.Timestamp(year=int(fy3_year), month=12, day=31)
    signal = pd.Timestamp(str(int(signal_date)))
    horizon = (endpoint - signal).days / 365.25
    if horizon <= 0:
        return _invalid("NON_POSITIVE_HORIZON")

    justified_pb = (steady_roe - perpetual_growth) / spread
    justified_pe = justified_pb / steady_roe
    target = eps_fy3 * justified_pe
    if not np.isfinite(target) or target <= 0:
        return _invalid("NON_POSITIVE_TARGET")
    implied = (target / close) ** (1.0 / horizon) - 1.0
    if not np.isfinite(implied):
        return _invalid("NON_FINITE_RETURN")
    return ValuationResult(implied, target, justified_pb, justified_pe, horizon, "VALID")


def materialize_growth_mapping(sw_members: pd.DataFrame, policy_path: str | Path) -> pd.DataFrame:
    with Path(policy_path).open("rb") as handle:
        policy = tomllib.load(handle)
    defaults = policy["sw1_default"]
    overrides = policy.get("sw3_override", {})
    allowed = set(float(value) for value in policy["buckets"])
    required = {"l1_code", "l1_name", "l2_code", "l2_name", "l3_code", "l3_name"}
    missing = required - set(sw_members.columns)
    if missing:
        raise ValueError(f"SW members missing columns: {sorted(missing)}")
    mapping = sw_members.loc[:, sorted(required)].dropna(subset=["l3_code"]).drop_duplicates()
    conflicts = mapping.groupby("l3_code", observed=True)["l1_name"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("SW3 codes map to multiple SW1 industries")
    mapping = mapping.drop_duplicates("l3_code", keep="last").copy()
    mapping["perpetual_growth"] = mapping.apply(
        lambda row: float(overrides.get(row["l3_code"], defaults.get(row["l1_name"], np.nan))),
        axis=1,
    )
    if mapping["perpetual_growth"].isna().any():
        missing_names = mapping.loc[mapping["perpetual_growth"].isna(), "l1_name"].unique()
        raise ValueError(f"unmapped SW1 policies: {sorted(missing_names)}")
    if not set(mapping["perpetual_growth"]).issubset(allowed):
        raise ValueError("growth mapping contains values outside policy buckets")
    mapping["policy_version"] = policy["version"]
    mapping["effective_from"] = int(policy["effective_from"])
    return mapping.sort_values("l3_code").reset_index(drop=True)


def _invalid(reason: str) -> ValuationResult:
    return ValuationResult(None, None, None, None, None, reason)
