"""Point-in-time signal universe construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EXCLUDED_SW_L1 = frozenset({"银行", "非银金融", "房地产"})
ALLOWED_SUFFIXES = (".SH", ".SZ")


class UniverseError(ValueError):
    """Raised when universe inputs are incomplete or ambiguous."""


@dataclass(frozen=True)
class UniverseResult:
    signal_date: int
    audit: pd.DataFrame
    eligible: pd.DataFrame
    funnel: dict[str, int]


def month_end_trade_dates(trade_calendar: pd.DataFrame) -> list[int]:
    """Return the final open exchange date in each calendar month."""
    required = {"cal_date", "is_open"}
    missing = required - set(trade_calendar.columns)
    if missing:
        raise UniverseError(f"trade calendar missing columns: {sorted(missing)}")
    opened = trade_calendar.loc[trade_calendar["is_open"].eq(1), ["cal_date"]].copy()
    opened["cal_date"] = pd.to_numeric(opened["cal_date"], errors="raise").astype("int64")
    opened["month"] = pd.to_datetime(
        opened["cal_date"].astype(str), format="%Y%m%d", errors="raise"
    ).dt.to_period("M")
    return opened.groupby("month", sort=True)["cal_date"].max().astype(int).tolist()


def active_sw_membership(sw_members: pd.DataFrame, signal_date: int) -> pd.DataFrame:
    """Select one Shenwan membership effective on ``signal_date`` per security."""
    required = {
        "ts_code",
        "l1_code",
        "l1_name",
        "l2_code",
        "l2_name",
        "l3_code",
        "l3_name",
        "in_date",
        "out_date",
    }
    missing = required - set(sw_members.columns)
    if missing:
        raise UniverseError(f"SW membership missing columns: {sorted(missing)}")

    members = sw_members.loc[:, sorted(required)].copy()
    members["ts_code"] = members["ts_code"].astype("string").str.upper()
    members["in_date"] = pd.to_numeric(members["in_date"], errors="coerce")
    members["out_date"] = pd.to_numeric(members["out_date"], errors="coerce")

    active = members.loc[
        members["in_date"].le(signal_date)
        & (members["out_date"].isna() | members["out_date"].ge(signal_date))
    ].copy()
    active.sort_values(["ts_code", "in_date"], inplace=True)

    latest_in_date = active.groupby("ts_code", observed=True)["in_date"].transform("max")
    latest = active.loc[active["in_date"].eq(latest_in_date)].copy()
    conflicts = latest.groupby("ts_code", observed=True)["l3_code"].nunique(dropna=False)
    conflict_codes = conflicts.loc[conflicts.gt(1)].index.tolist()
    if conflict_codes:
        sample = conflict_codes[:5]
        raise UniverseError(
            f"conflicting active SW memberships for {len(conflict_codes)} securities: {sample}"
        )

    return latest.drop_duplicates("ts_code", keep="last").reset_index(drop=True)


def build_signal_universe(
    signal_date: int,
    stock_basic: pd.DataFrame,
    sw_members: pd.DataFrame,
    trade_filter: pd.DataFrame,
    st_list: pd.DataFrame | None = None,
    excluded_sw_l1: Iterable[str] = EXCLUDED_SW_L1,
) -> UniverseResult:
    """Build an auditable point-in-time universe for one signal date."""
    signal_date = int(signal_date)
    basic_required = {"ts_code", "list_date", "delist_date"}
    missing = basic_required - set(stock_basic.columns)
    if missing:
        raise UniverseError(f"stock basic missing columns: {sorted(missing)}")
    if stock_basic["ts_code"].duplicated().any():
        raise UniverseError("stock basic contains duplicate ts_code values")

    basic_cols = [
        column
        for column in ("ts_code", "name", "market", "exchange", "list_date", "delist_date")
        if column in stock_basic.columns
    ]
    audit = stock_basic.loc[:, basic_cols].copy()
    audit["ts_code"] = audit["ts_code"].astype("string").str.upper()
    audit["list_date"] = pd.to_numeric(audit["list_date"], errors="coerce")
    audit["delist_date"] = pd.to_numeric(audit["delist_date"], errors="coerce")
    audit.insert(0, "signal_date", signal_date)

    audit["is_sh_sz"] = audit["ts_code"].str.endswith(ALLOWED_SUFFIXES, na=False)
    audit["is_listed"] = audit["list_date"].le(signal_date) & (
        audit["delist_date"].isna() | audit["delist_date"].gt(signal_date)
    )

    membership = active_sw_membership(sw_members, signal_date)
    audit = audit.merge(membership, on="ts_code", how="left", validate="one_to_one")
    audit["has_industry"] = audit["l1_name"].notna() & audit["l3_code"].notna()
    audit["is_excluded_industry"] = audit["l1_name"].isin(set(excluded_sw_l1))

    st_codes = _signal_date_st_codes(trade_filter, st_list, signal_date)
    audit["is_st"] = audit["ts_code"].isin(st_codes)

    conditions = [
        ~audit["is_sh_sz"],
        ~audit["is_listed"],
        ~audit["has_industry"],
        audit["is_excluded_industry"],
        audit["is_st"],
    ]
    reasons = [
        "INVALID_EXCHANGE",
        "NOT_LISTED",
        "MISSING_INDUSTRY",
        "EXCLUDED_INDUSTRY",
        "ST",
    ]
    audit["exclusion_reason"] = np.select(conditions, reasons, default="ELIGIBLE")
    audit["eligible"] = audit["exclusion_reason"].eq("ELIGIBLE")

    funnel = {
        "stock_basic": len(audit),
        "sh_sz": int(audit["is_sh_sz"].sum()),
        "listed_sh_sz": int((audit["is_sh_sz"] & audit["is_listed"]).sum()),
        "industry_mapped": int(
            (audit["is_sh_sz"] & audit["is_listed"] & audit["has_industry"]).sum()
        ),
        "industry_allowed": int(
            (
                audit["is_sh_sz"]
                & audit["is_listed"]
                & audit["has_industry"]
                & ~audit["is_excluded_industry"]
            ).sum()
        ),
        "non_st": int(
            (
                audit["is_sh_sz"]
                & audit["is_listed"]
                & audit["has_industry"]
                & ~audit["is_excluded_industry"]
                & ~audit["is_st"]
            ).sum()
        ),
        "eligible": int(audit["eligible"].sum()),
    }
    eligible = audit.loc[audit["eligible"]].copy().reset_index(drop=True)
    return UniverseResult(
        signal_date=signal_date,
        audit=audit.sort_values("ts_code").reset_index(drop=True),
        eligible=eligible.sort_values("ts_code").reset_index(drop=True),
        funnel=funnel,
    )


def load_asset_pricing_universe_inputs(
    asset_pricing_root: str | Path, signal_date: int
) -> dict[str, pd.DataFrame]:
    """Load the upstream AssetPricingML tables needed by Stage 2."""
    root = Path(asset_pricing_root)
    date = str(int(signal_date))
    year = date[:4]
    paths = {
        "stock_basic": root / "data/stock_data/info/stock_basic.parquet",
        "sw_members": root / "data/index_data/member_sw/sw_members.parquet",
        "trade_filter": root
        / f"data/stock_data/daily/trade_filter/{year}/{date}.parquet",
        "st_list": root / f"data/stock_data/daily/st_list/{year}/{date}.parquet",
    }
    missing_required = [
        str(paths[name])
        for name in ("stock_basic", "sw_members", "trade_filter")
        if not paths[name].is_file()
    ]
    if missing_required:
        raise UniverseError(f"missing required upstream files: {missing_required}")

    loaded = {
        "stock_basic": pd.read_parquet(paths["stock_basic"]),
        "sw_members": pd.read_parquet(paths["sw_members"]),
        "trade_filter": pd.read_parquet(paths["trade_filter"]),
    }
    loaded["st_list"] = (
        pd.read_parquet(paths["st_list"])
        if paths["st_list"].is_file()
        else pd.DataFrame(columns=["trade_date", "ts_code"])
    )
    return loaded


def _signal_date_st_codes(
    trade_filter: pd.DataFrame, st_list: pd.DataFrame | None, signal_date: int
) -> set[str]:
    required = {"trade_date", "ts_code", "is_st"}
    missing = required - set(trade_filter.columns)
    if missing:
        raise UniverseError(f"trade filter missing columns: {sorted(missing)}")

    daily_filter = trade_filter.loc[
        pd.to_numeric(trade_filter["trade_date"], errors="coerce").eq(signal_date)
    ].copy()
    if daily_filter.empty:
        raise UniverseError(f"trade filter has no rows for signal date {signal_date}")
    daily_filter["ts_code"] = daily_filter["ts_code"].astype("string").str.upper()
    is_st = daily_filter["is_st"].fillna(False).astype(bool)
    codes = set(daily_filter.loc[is_st, "ts_code"].dropna().tolist())

    if st_list is not None and not st_list.empty:
        st_required = {"trade_date", "ts_code"}
        st_missing = st_required - set(st_list.columns)
        if st_missing:
            raise UniverseError(f"ST list missing columns: {sorted(st_missing)}")
        st_daily = st_list.loc[
            pd.to_numeric(st_list["trade_date"], errors="coerce").eq(signal_date)
        ]
        codes.update(st_daily["ts_code"].astype("string").str.upper().dropna().tolist())
    return codes
