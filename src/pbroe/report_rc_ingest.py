"""Resumable TuShare report_rc ingestion with explicit zero-row manifests."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Callable, Protocol

import pandas as pd

from .consensus import ConsensusError, manifest_record, validate_download_manifest


REPORT_RC_FIELDS = (
    "ts_code",
    "name",
    "report_date",
    "report_title",
    "report_type",
    "classify",
    "org_name",
    "author_name",
    "quarter",
    "op_rt",
    "op_pr",
    "tp",
    "np",
    "eps",
    "pe",
    "rd",
    "roe",
    "ev_ebitda",
    "rating",
    "max_price",
    "min_price",
    "create_time",
    "imp_dg",
)


class ReportRcApi(Protocol):
    def report_rc(self, **kwargs) -> pd.DataFrame: ...


@dataclass(frozen=True)
class DownloadSummary:
    requested_dates: int
    skipped_dates: int
    success_dates: int
    empty_dates: int
    failed_dates: int
    downloaded_rows: int


class ReportRcDownloader:
    """Download one report-date partition at a time and checkpoint every date."""

    def __init__(
        self,
        api: ReportRcApi,
        output_root: str | Path,
        page_limit: int = 3000,
        range_chunk_days: int = 31,
        requests_per_minute: float = 480.0,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        progress_callback: Callable[[dict], None] | None = None,
        max_workers: int = 1,
    ):
        self.api = api
        self.output_root = Path(output_root)
        self.partition_root = self.output_root / "daily"
        self.manifest_path = self.output_root / "download_manifest.csv"
        self.page_limit = int(page_limit)
        self.range_chunk_days = int(range_chunk_days)
        self.sleep_seconds = 0.0 if requests_per_minute <= 0 else 60.0 / requests_per_minute
        self.max_retries = int(max_retries)
        self.retry_delay_seconds = float(retry_delay_seconds)
        self.progress_callback = progress_callback
        self.max_workers = int(max_workers)
        if self.page_limit <= 0:
            raise ConsensusError("page_limit must be positive")
        if self.page_limit > 3000:
            raise ConsensusError("report_rc page_limit cannot exceed the official 3000-row limit")
        if self.range_chunk_days <= 0:
            raise ConsensusError("range_chunk_days must be positive")
        if self.max_workers <= 0:
            raise ConsensusError("max_workers must be positive")
        self.partition_root.mkdir(parents=True, exist_ok=True)

    def sync(self, start_date: int, end_date: int) -> DownloadSummary:
        expected_dates = calendar_dates(start_date, end_date)
        manifest = self._load_manifest()
        completed = set(
            manifest.loc[
                manifest["status"].isin(["SUCCESS", "EMPTY"]), "request_date"
            ].astype(int)
        ) if not manifest.empty else set()

        skipped = len(set(expected_dates) & completed)
        success = empty = failed = rows = 0
        missing_dates = [value for value in expected_dates if value not in completed]
        chunks = _consecutive_chunks(missing_dates, self.range_chunk_days)
        jobs = [(number, chunk) for number, chunk in enumerate(chunks, start=1)]
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._fetch_chunk, number, chunk): (number, chunk)
                for number, chunk in jobs
            }
            for future in as_completed(future_map):
                chunk_number, date_chunk = future_map[future]
                start = date_chunk[0]
                end = date_chunk[-1]
                try:
                    frame = future.result()
                except Exception as exc:  # every failed date remains retryable
                    failed += len(date_chunk)
                    error = f"{type(exc).__name__}: {exc}"
                    for request_date in date_chunk:
                        manifest = self._upsert_manifest(
                            manifest, manifest_record(request_date, 0, error=error)
                        )
                    if self.progress_callback:
                        self.progress_callback(
                            {
                                "chunk": chunk_number,
                                "chunks": len(chunks),
                                "start_date": start,
                                "end_date": end,
                                "rows": 0,
                                "status": "FAILED",
                                "error": error,
                            }
                        )
                    continue

                rows += len(frame)
                grouped = {
                    int(request_date): partition.reset_index(drop=True)
                    for request_date, partition in frame.groupby("report_date", observed=True)
                } if not frame.empty else {}
                for request_date in date_chunk:
                    partition = grouped.get(request_date)
                    if partition is None or partition.empty:
                        empty += 1
                        record = manifest_record(request_date, 0)
                    else:
                        self._write_partition(request_date, partition)
                        success += 1
                        record = manifest_record(request_date, len(partition))
                    manifest = self._upsert_manifest(manifest, record)
                if self.progress_callback:
                    self.progress_callback(
                        {
                            "chunk": chunk_number,
                            "chunks": len(chunks),
                            "start_date": start,
                            "end_date": end,
                            "rows": len(frame),
                            "status": "SUCCESS",
                        }
                    )

        final_manifest = self._load_manifest()
        if failed == 0:
            validate_download_manifest(final_manifest, expected_dates)
        return DownloadSummary(
            requested_dates=len(expected_dates),
            skipped_dates=skipped,
            success_dates=success,
            empty_dates=empty,
            failed_dates=failed,
            downloaded_rows=rows,
        )

    def _fetch_chunk(self, chunk_number: int, date_chunk: list[int]) -> pd.DataFrame:
        del chunk_number
        return self._fetch_range(date_chunk[0], date_chunk[-1])

    def _fetch_range(self, start_date: int, end_date: int) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        offset = 0
        while True:
            response = None
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.api.report_rc(
                        start_date=str(start_date),
                        end_date=str(end_date),
                        fields=",".join(REPORT_RC_FIELDS),
                        limit=self.page_limit,
                        offset=offset,
                    )
                    break
                except Exception:
                    if attempt >= self.max_retries:
                        raise
                    time.sleep(self.retry_delay_seconds)
            if response is None or response.empty:
                break
            chunks.append(normalize_report_schema(response))
            if len(response) < self.page_limit:
                break
            offset += self.page_limit
            time.sleep(self.sleep_seconds)
        time.sleep(self.sleep_seconds)
        if not chunks:
            return pd.DataFrame(columns=REPORT_RC_FIELDS)
        combined = pd.concat(chunks, ignore_index=True)
        combined = combined.drop_duplicates().reset_index(drop=True)
        dates = pd.to_numeric(combined["report_date"], errors="coerce")
        if dates.isna().any() or not dates.between(start_date, end_date).all():
            raise ConsensusError("report_rc returned invalid or out-of-range report_date values")
        combined["report_date"] = dates.astype("int64")
        return combined

    def _write_partition(self, request_date: int, frame: pd.DataFrame) -> None:
        year_root = self.partition_root / str(request_date)[:4]
        year_root.mkdir(parents=True, exist_ok=True)
        final_path = year_root / f"{request_date}.parquet"
        temp_path = year_root / f".{request_date}.parquet.tmp"
        frame.to_parquet(temp_path, index=False)
        temp_path.replace(final_path)

    def _load_manifest(self) -> pd.DataFrame:
        if not self.manifest_path.is_file():
            return pd.DataFrame(
                columns=["request_date", "status", "row_count", "completed_at", "error"]
            )
        manifest = pd.read_csv(self.manifest_path)
        if not manifest.empty:
            manifest["request_date"] = pd.to_numeric(
                manifest["request_date"], errors="raise"
            ).astype("int64")
        return manifest

    def _upsert_manifest(self, manifest: pd.DataFrame, record: dict) -> pd.DataFrame:
        updated = pd.concat([manifest, pd.DataFrame([record])], ignore_index=True)
        updated.drop_duplicates("request_date", keep="last", inplace=True)
        updated.sort_values("request_date", inplace=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_root / ".download_manifest.csv.tmp"
        updated.to_csv(temp_path, index=False)
        temp_path.replace(self.manifest_path)
        return updated


def normalize_report_schema(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in REPORT_RC_FIELDS:
        if column not in work.columns:
            work[column] = pd.NA
    work = work.loc[:, REPORT_RC_FIELDS]
    work["report_date"] = pd.to_numeric(
        work["report_date"].astype("string").str.replace("-", "", regex=False),
        errors="coerce",
    ).astype("Int64")
    numeric_columns = (
        "op_rt",
        "op_pr",
        "tp",
        "np",
        "eps",
        "pe",
        "rd",
        "roe",
        "ev_ebitda",
        "max_price",
        "min_price",
        "imp_dg",
    )
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def calendar_dates(start_date: int, end_date: int) -> list[int]:
    start = pd.Timestamp(str(int(start_date)))
    end = pd.Timestamp(str(int(end_date)))
    if end < start:
        raise ConsensusError("end_date must be on or after start_date")
    return [int(value.strftime("%Y%m%d")) for value in pd.date_range(start, end, freq="D")]


def _consecutive_chunks(values: list[int], max_days: int) -> list[list[int]]:
    if not values:
        return []
    chunks: list[list[int]] = []
    current = [values[0]]
    for value in values[1:]:
        previous = pd.Timestamp(str(current[-1]))
        candidate = pd.Timestamp(str(value))
        if candidate - previous == pd.Timedelta(days=1) and len(current) < max_days:
            current.append(value)
        else:
            chunks.append(current)
            current = [value]
    chunks.append(current)
    return chunks
