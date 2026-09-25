"""Point-in-time sell-side consensus reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import unicodedata
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


class ConsensusError(ValueError):
    """Raised when analyst-report inputs violate the consensus contract."""


@dataclass(frozen=True)
class ConsensusResult:
    signal_date: int
    audit: pd.DataFrame
    eligible: pd.DataFrame
    latest_broker_forecasts: pd.DataFrame
    funnel: dict[str, int]


REQUIRED_REPORT_COLUMNS = frozenset(
    {"ts_code", "report_date", "org_name", "quarter", "eps"}
)


def normalize_broker_name(
    value: object, aliases: Mapping[str, str] | None = None
) -> str | None:
    """Normalize common legal suffix and whitespace variations in broker names."""
    if value is None or pd.isna(value):
        return None
    name = unicodedata.normalize("NFKC", str(value)).strip()
    name = re.sub(r"[\s·•]+", "", name)
    name = re.sub(r"[()（）]", "", name)
    for suffix in (
        "股份有限公司",
        "有限责任公司",
        "有限公司",
        "研究中心",
        "研究所",
    ):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    alias_map = aliases or {}
    return alias_map.get(name, name) or None


def forecast_fiscal_year(value: object) -> int | None:
    """Extract a four-digit fiscal year from TuShare's quarter field."""
    if value is None or pd.isna(value):
        return None
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", str(value))
    return int(match.group(1)) if match else None


def build_consensus_asof(
    reports: pd.DataFrame,
    signal_date: int,
    lookback_days: int = 180,
    min_brokers: int = 2,
    forecast_year_count: int = 3,
    broker_aliases: Mapping[str, str] | None = None,
) -> ConsensusResult:
    """Rebuild FY1-FY3 consensus using only information observable by signal date."""
    missing = REQUIRED_REPORT_COLUMNS - set(reports.columns)
    if missing:
        raise ConsensusError(f"analyst reports missing columns: {sorted(missing)}")
    if lookback_days <= 0:
        raise ConsensusError("lookback_days must be positive")
    if min_brokers < 2:
        raise ConsensusError("min_brokers must be at least 2")
    if forecast_year_count != 3:
        raise ConsensusError("V1 forecast_year_count must equal 3")

    signal_ts = pd.Timestamp(str(int(signal_date)))
    lookback_start = signal_ts - pd.Timedelta(days=lookback_days)
    target_years = [signal_ts.year + offset for offset in range(forecast_year_count)]

    normalized = reports.copy()
    normalized["ts_code"] = normalized["ts_code"].astype("string").str.upper()
    normalized["report_date"] = _parse_yyyymmdd(normalized["report_date"])
    normalized["broker"] = normalized["org_name"].map(
        lambda value: normalize_broker_name(value, broker_aliases)
    ).astype("string")
    normalized["forecast_year"] = normalized["quarter"].map(forecast_fiscal_year)
    normalized["eps"] = pd.to_numeric(normalized["eps"], errors="coerce")
    if "np" in normalized.columns:
        normalized["np"] = pd.to_numeric(normalized["np"], errors="coerce")
    else:
        normalized["np"] = np.nan

    if "create_time" in normalized.columns:
        normalized["create_timestamp"] = pd.to_datetime(
            normalized["create_time"], errors="coerce"
        )
    else:
        normalized["create_timestamp"] = pd.NaT

    eligible_mask = (
        normalized["report_date"].between(lookback_start, signal_ts, inclusive="both")
        & normalized["forecast_year"].isin(target_years)
        & normalized["ts_code"].notna()
        & normalized["broker"].notna()
    )
    eligible_reports = normalized.loc[eligible_mask].copy()
    eligible_reports.sort_values(
        ["ts_code", "broker", "forecast_year", "report_date", "create_timestamp"],
        na_position="first",
        inplace=True,
    )
    latest = eligible_reports.drop_duplicates(
        ["ts_code", "broker", "forecast_year"], keep="last"
    ).reset_index(drop=True)

    all_codes = sorted(eligible_reports["ts_code"].dropna().astype(str).unique())
    audit = pd.DataFrame(
        {"ts_code": pd.Series(all_codes, dtype="string")}
    )
    audit.insert(0, "signal_date", int(signal_date))

    valid_eps = latest.loc[latest["eps"].notna()].copy()
    grouped = valid_eps.groupby(["ts_code", "forecast_year"], observed=True)
    aggregates = grouped.agg(
        eps_mean=("eps", "mean"),
        np_mean=("np", "mean"),
        broker_count=("broker", "nunique"),
        latest_report_date=("report_date", "max"),
    ).reset_index()

    for index, year in enumerate(target_years, start=1):
        year_data = aggregates.loc[aggregates["forecast_year"].eq(year)].copy()
        year_data.rename(
            columns={
                "forecast_year": f"forecast_year_fy{index}",
                "eps_mean": f"eps_fy{index}",
                "np_mean": f"np_fy{index}",
                "broker_count": f"broker_count_fy{index}",
                "latest_report_date": f"latest_report_date_fy{index}",
            },
            inplace=True,
        )
        audit = audit.merge(year_data, on="ts_code", how="left", validate="one_to_one")
        audit[f"forecast_year_fy{index}"] = audit[f"forecast_year_fy{index}"].fillna(
            year
        ).astype("int64")
        audit[f"broker_count_fy{index}"] = audit[f"broker_count_fy{index}"].fillna(
            0
        ).astype("int64")

    terminal = aggregates.loc[aggregates["broker_count"].ge(min_brokers)].copy()
    terminal["fiscal_year_end"] = pd.to_datetime(
        terminal["forecast_year"].astype("int64").astype(str) + "1231",
        format="%Y%m%d",
    )
    terminal = terminal.loc[terminal["fiscal_year_end"].gt(signal_ts)].copy()
    terminal.sort_values(["ts_code", "forecast_year"], inplace=True)
    terminal = terminal.drop_duplicates("ts_code", keep="last")
    terminal.rename(
        columns={
            "forecast_year": "terminal_forecast_year",
            "eps_mean": "terminal_eps",
            "np_mean": "terminal_np",
            "broker_count": "terminal_broker_count",
            "latest_report_date": "terminal_latest_report_date",
        },
        inplace=True,
    )
    terminal_columns = [
        "ts_code",
        "terminal_forecast_year",
        "terminal_eps",
        "terminal_np",
        "terminal_broker_count",
        "terminal_latest_report_date",
    ]
    audit = audit.merge(
        terminal[terminal_columns], on="ts_code", how="left", validate="one_to_one"
    )
    audit["terminal_broker_count"] = audit["terminal_broker_count"].fillna(0).astype("int64")
    audit["consensus_reason"] = np.where(
        audit["terminal_broker_count"].ge(min_brokers),
        "VALID",
        "NO_ELIGIBLE_TERMINAL_FORECAST",
    )
    audit["consensus_valid"] = audit["consensus_reason"].eq("VALID")
    audit.sort_values("ts_code", inplace=True)
    audit.reset_index(drop=True, inplace=True)
    eligible = audit.loc[audit["consensus_valid"]].copy().reset_index(drop=True)

    funnel = {
        "raw_rows": len(reports),
        "asof_lookback_target_rows": len(eligible_reports),
        "latest_broker_year_rows": len(latest),
        "stocks_with_target_reports": len(audit),
        "valid_consensus_stocks": len(eligible),
    }
    return ConsensusResult(
        signal_date=int(signal_date),
        audit=audit,
        eligible=eligible,
        latest_broker_forecasts=latest,
        funnel=funnel,
    )


def validate_download_manifest(
    manifest: pd.DataFrame, expected_dates: Sequence[int]
) -> dict[str, int]:
    """Validate resumable report downloads, including successful zero-row dates."""
    required = {"request_date", "status", "row_count", "completed_at"}
    missing = required - set(manifest.columns)
    if missing:
        raise ConsensusError(f"download manifest missing columns: {sorted(missing)}")
    if manifest["request_date"].duplicated().any():
        raise ConsensusError("download manifest contains duplicate request dates")

    work = manifest.copy()
    work["request_date"] = pd.to_numeric(work["request_date"], errors="coerce")
    expected = {int(value) for value in expected_dates}
    observed = set(work["request_date"].dropna().astype(int))
    missing_dates = sorted(expected - observed)
    if missing_dates:
        raise ConsensusError(
            f"download manifest missing {len(missing_dates)} dates; sample={missing_dates[:5]}"
        )
    selected = work.loc[work["request_date"].isin(expected)].copy()
    if not selected["status"].isin({"SUCCESS", "EMPTY"}).all():
        bad = selected.loc[~selected["status"].isin({"SUCCESS", "EMPTY"}), "request_date"]
        raise ConsensusError(f"download manifest has failed dates: {bad.astype(int).tolist()[:5]}")
    counts = pd.to_numeric(selected["row_count"], errors="coerce")
    if counts.isna().any() or counts.lt(0).any():
        raise ConsensusError("download manifest has invalid row counts")
    if not counts.loc[selected["status"].eq("EMPTY")].eq(0).all():
        raise ConsensusError("EMPTY manifest rows must have row_count=0")
    if selected["completed_at"].isna().any():
        raise ConsensusError("download manifest has missing completion timestamps")
    return {
        "expected_dates": len(expected),
        "completed_dates": len(selected),
        "success_dates": int(selected["status"].eq("SUCCESS").sum()),
        "empty_dates": int(selected["status"].eq("EMPTY").sum()),
        "downloaded_rows": int(counts.sum()),
    }


def manifest_record(request_date: int, row_count: int, error: str | None = None) -> dict:
    """Create one normalized completeness-manifest record."""
    count = int(row_count)
    if count < 0:
        raise ConsensusError("row_count cannot be negative")
    status = "FAILED" if error else ("EMPTY" if count == 0 else "SUCCESS")
    return {
        "request_date": int(request_date),
        "status": status,
        "row_count": count,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }


def load_report_partitions(
    report_root: str | Path, start_date: int, end_date: int
) -> pd.DataFrame:
    """Load available daily report partitions over an inclusive calendar range."""
    root = Path(report_root)
    frames: list[pd.DataFrame] = []
    for timestamp in pd.date_range(
        pd.Timestamp(str(int(start_date))), pd.Timestamp(str(int(end_date))), freq="D"
    ):
        date = timestamp.strftime("%Y%m%d")
        path = root / "daily" / date[:4] / f"{date}.parquet"
        if path.is_file():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=sorted(REQUIRED_REPORT_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _parse_yyyymmdd(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype("string").str.replace("-", "", regex=False).str.slice(0, 8)
    )
    return pd.to_datetime(normalized, format="%Y%m%d", errors="coerce")
