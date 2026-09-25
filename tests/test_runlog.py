import json

import pytest

from pbroe.runlog import StageRunLogger


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_successful_stage_log_contains_required_fields(tmp_path):
    with StageRunLogger(tmp_path, stage=1, run_id="success") as run:
        run.add_check("import", "PASS", "ok")
        run.add_row_count("rows", 3)

    events = read_events(tmp_path / "stage_1_success.jsonl")
    assert events[0]["event"] == "stage_start"
    assert events[-1]["event"] == "stage_end"
    assert events[-1]["status"] == "PASS"
    assert events[-1]["row_counts"] == {"rows": 3}
    assert events[-1]["started_at"]
    assert events[-1]["ended_at"]


def test_exception_is_recorded_and_propagated(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with StageRunLogger(tmp_path, stage=1, run_id="failure"):
            raise RuntimeError("boom")

    events = read_events(tmp_path / "stage_1_failure.jsonl")
    final = events[-1]
    assert final["status"] == "FAIL"
    assert final["exception"] == {"type": "RuntimeError", "message": "boom"}


def test_invalid_check_status_is_rejected(tmp_path):
    with StageRunLogger(tmp_path, stage=1, run_id="bad-status") as run:
        with pytest.raises(ValueError, match="PASS, FAIL, or WARN"):
            run.add_check("bad", "UNKNOWN")
