"""Resumable annual fina_indicator_vip ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Protocol

import pandas as pd

from .consensus import ConsensusError, manifest_record


FIELDS = ("ts_code", "ann_date", "end_date", "roe", "roe_waa", "roe_avg")


class FinancialApi(Protocol):
    def fina_indicator_vip(self, **kwargs) -> pd.DataFrame: ...


@dataclass(frozen=True)
class FinancialDownloadSummary:
    requested_periods: int
    skipped_periods: int
    success_periods: int
    empty_periods: int
    failed_periods: int
    downloaded_rows: int


class AnnualIndicatorDownloader:
    def __init__(self, api: FinancialApi, output_root: str | Path, page_limit=5000):
        self.api = api
        self.output_root = Path(output_root)
        self.data_root = self.output_root / "annual"
        self.manifest_path = self.output_root / "download_manifest.csv"
        self.page_limit = int(page_limit)
        self.data_root.mkdir(parents=True, exist_ok=True)

    def sync(self, start_year: int, end_year: int) -> FinancialDownloadSummary:
        periods = [int(f"{year}1231") for year in range(start_year, end_year + 1)]
        manifest = self._load_manifest()
        complete = set(
            manifest.loc[manifest.status.isin(["SUCCESS", "EMPTY"]), "request_date"].astype(int)
        ) if not manifest.empty else set()
        skipped = success = empty = failed = rows = 0
        for period in periods:
            if period in complete:
                skipped += 1
                continue
            try:
                frame = self._fetch(period)
                if frame.empty:
                    empty += 1
                else:
                    path = self.data_root / f"{period}.parquet"
                    temp = self.data_root / f".{period}.parquet.tmp"
                    frame.to_parquet(temp, index=False)
                    temp.replace(path)
                    success += 1
                    rows += len(frame)
                record = manifest_record(period, len(frame))
            except Exception as exc:
                failed += 1
                record = manifest_record(period, 0, f"{type(exc).__name__}: {exc}")
            manifest = self._upsert(manifest, record)
        return FinancialDownloadSummary(len(periods), skipped, success, empty, failed, rows)

    def _fetch(self, period: int) -> pd.DataFrame:
        chunks = []
        offset = 0
        while True:
            response = self.api.fina_indicator_vip(
                period=str(period), fields=",".join(FIELDS), limit=self.page_limit, offset=offset
            )
            if response is None or response.empty:
                break
            chunks.append(response)
            if len(response) < self.page_limit:
                break
            offset += self.page_limit
            time.sleep(0.13)
        if not chunks:
            return pd.DataFrame(columns=FIELDS)
        frame = pd.concat(chunks, ignore_index=True)
        for column in FIELDS:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame.loc[:, FIELDS]
        for column in ("ann_date", "end_date", "roe", "roe_waa", "roe_avg"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not frame["end_date"].dropna().eq(period).all():
            raise ConsensusError(f"fina_indicator_vip returned wrong period for {period}")
        return frame.drop_duplicates().reset_index(drop=True)

    def _load_manifest(self):
        if not self.manifest_path.is_file():
            return pd.DataFrame(columns=["request_date", "status", "row_count", "completed_at", "error"])
        return pd.read_csv(self.manifest_path)

    def _upsert(self, manifest, record):
        result = pd.concat([manifest, pd.DataFrame([record])], ignore_index=True)
        result.drop_duplicates("request_date", keep="last", inplace=True)
        result.sort_values("request_date", inplace=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        temp = self.output_root / ".download_manifest.csv.tmp"
        result.to_csv(temp, index=False)
        temp.replace(self.manifest_path)
        return result


def load_annual_indicators(root: str | Path) -> pd.DataFrame:
    files = sorted((Path(root) / "annual").glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=FIELDS)
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
