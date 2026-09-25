from pathlib import Path

import pandas as pd

from pbroe.consensus import validate_download_manifest
from pbroe.report_rc_ingest import ReportRcDownloader, calendar_dates


class FakeApi:
    def __init__(self):
        self.calls = []

    def report_rc(self, **kwargs):
        start = int(kwargs["start_date"])
        end = int(kwargs["end_date"])
        self.calls.append((start, end))
        if start <= 20240102 <= end:
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "report_date": 20240102,
                        "org_name": "甲证券",
                        "quarter": "2024Q4",
                        "eps": "1.2",
                    }
                ]
            )
        return pd.DataFrame()


def test_downloader_checkpoints_success_and_empty_dates(tmp_path):
    api = FakeApi()
    downloader = ReportRcDownloader(api, tmp_path, requests_per_minute=0, max_workers=2)
    summary = downloader.sync(20240101, 20240102)
    assert summary.empty_dates == 1
    assert summary.success_dates == 1
    assert summary.failed_dates == 0
    assert (tmp_path / "daily/2024/20240102.parquet").is_file()

    manifest = pd.read_csv(tmp_path / "download_manifest.csv")
    validated = validate_download_manifest(manifest, [20240101, 20240102])
    assert validated["completed_dates"] == 2
    assert validated["downloaded_rows"] == 1

    second = downloader.sync(20240101, 20240102)
    assert second.skipped_dates == 2
    assert api.calls == [(20240101, 20240102)]


def test_calendar_dates_include_weekends():
    assert calendar_dates(20240105, 20240108) == [
        20240105,
        20240106,
        20240107,
        20240108,
    ]
