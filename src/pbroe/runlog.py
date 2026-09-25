"""Structured stage run logging."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageRunLogger:
    """Context manager that records stage lifecycle and acceptance evidence."""

    def __init__(self, log_root: str | Path, stage: int, run_id: str | None = None):
        self.log_root = Path(log_root)
        if not self.log_root.is_dir():
            raise ValueError(f"run log directory does not exist: {self.log_root}")
        self.stage = int(stage)
        self.run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        self.path = self.log_root / f"stage_{self.stage}_{self.run_id}.jsonl"
        self.started_at: str | None = None
        self.checks: list[dict[str, Any]] = []
        self.row_counts: dict[str, int] = {}

    def __enter__(self) -> "StageRunLogger":
        self.started_at = _utc_now()
        self._write(
            {
                "event": "stage_start",
                "run_id": self.run_id,
                "stage": self.stage,
                "started_at": self.started_at,
                "status": "RUNNING",
            }
        )
        return self

    def add_check(self, name: str, status: str, details: Any = None) -> None:
        normalized = status.upper()
        if normalized not in {"PASS", "FAIL", "WARN"}:
            raise ValueError("check status must be PASS, FAIL, or WARN")
        check = {"name": name, "status": normalized, "details": details}
        self.checks.append(check)
        self._write(
            {
                "event": "check",
                "run_id": self.run_id,
                "stage": self.stage,
                "timestamp": _utc_now(),
                **check,
            }
        )

    def add_row_count(self, name: str, value: int) -> None:
        count = int(value)
        if count < 0:
            raise ValueError("row count cannot be negative")
        self.row_counts[name] = count
        self._write(
            {
                "event": "row_count",
                "run_id": self.run_id,
                "stage": self.stage,
                "timestamp": _utc_now(),
                "name": name,
                "value": count,
            }
        )

    def __exit__(self, exc_type, exc, traceback) -> bool:
        failed_check = any(check["status"] == "FAIL" for check in self.checks)
        status = "FAIL" if exc is not None or failed_check else "PASS"
        self._write(
            {
                "event": "stage_end",
                "run_id": self.run_id,
                "stage": self.stage,
                "started_at": self.started_at,
                "ended_at": _utc_now(),
                "status": status,
                "checks": self.checks,
                "row_counts": self.row_counts,
                "exception": None
                if exc is None
                else {"type": exc_type.__name__, "message": str(exc)},
            }
        )
        return False

    def _write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
