import uuid
from unittest.mock import MagicMock, call, patch

import psycopg.types.json

from rag.worker import check_stage_drift, claim_next_job, recover_stuck_jobs


def _make_conn(fetchone=None, fetchall=None):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = fetchone
    conn.execute.return_value.fetchall.return_value = fetchall or []
    return conn


def test_recover_stuck_jobs_marks_failed():
    job_id = uuid.uuid4()
    conn = _make_conn(fetchall=[(job_id, "parsing")])

    result = recover_stuck_jobs(conn, 30)

    assert result == 1
    # Should have called execute twice: SELECT and UPDATE
    assert conn.execute.call_count == 2
    update_call_args = conn.execute.call_args_list[1]
    sql = update_call_args[0][0]
    params = update_call_args[0][1]
    assert "UPDATE" in sql
    assert "failed:parsing" in params[0]
    conn.commit.assert_called_once()


def test_recover_stuck_jobs_skips_fresh():
    conn = _make_conn(fetchall=[])

    result = recover_stuck_jobs(conn, 30)

    assert result == 0
    # Only the SELECT should have been called
    assert conn.execute.call_count == 1
    conn.commit.assert_called_once()


# --- U6: pinned-baseline drift guardrail (R11) ------------------------------


def _drift_conn(stage_log, baselines):
    conn = MagicMock()
    conn.execute.side_effect = [
        MagicMock(fetchone=MagicMock(return_value=(stage_log,))),
        MagicMock(fetchall=MagicMock(return_value=baselines)),
    ]
    return conn


@patch("rag.worker.log")
def test_check_stage_drift_warns_at_4x_baseline(mock_log):
    conn = _drift_conn({"chunking": {"duration_ms": 4000}}, [("chunking", 1000)])
    with patch("rag.worker.settings") as mock_settings:
        mock_settings.STAGE_DRIFT_WARN_FACTOR = 3.0
        check_stage_drift(conn, "job-1")
    mock_log.warning.assert_called_once()
    _, kwargs = mock_log.warning.call_args
    assert kwargs["stage"] == "chunking"
    assert kwargs["duration_ms"] == 4000
    assert kwargs["baseline_ms"] == 1000


@patch("rag.worker.log")
def test_check_stage_drift_no_warning_at_1x_baseline(mock_log):
    conn = _drift_conn({"chunking": {"duration_ms": 1000}}, [("chunking", 1000)])
    with patch("rag.worker.settings") as mock_settings:
        mock_settings.STAGE_DRIFT_WARN_FACTOR = 3.0
        check_stage_drift(conn, "job-1")
    mock_log.warning.assert_not_called()


@patch("rag.worker.log")
def test_check_stage_drift_no_warning_under_factor(mock_log):
    """2x the baseline is under the 3.0 factor -- no warning."""
    conn = _drift_conn({"chunking": {"duration_ms": 2000}}, [("chunking", 1000)])
    with patch("rag.worker.settings") as mock_settings:
        mock_settings.STAGE_DRIFT_WARN_FACTOR = 3.0
        check_stage_drift(conn, "job-1")
    mock_log.warning.assert_not_called()


@patch("rag.worker.log")
def test_check_stage_drift_skips_stage_with_no_baseline(mock_log):
    """A stage with no pinned baseline yet is skipped, not treated as drift."""
    conn = _drift_conn({"chunking": {"duration_ms": 999_999}}, [])
    with patch("rag.worker.settings") as mock_settings:
        mock_settings.STAGE_DRIFT_WARN_FACTOR = 3.0
        check_stage_drift(conn, "job-1")
    mock_log.warning.assert_not_called()


@patch("rag.worker.log")
def test_check_stage_drift_regression_guard_pinned_baseline_still_fires(mock_log):
    """R11 regression guard: even when the ENTIRE recent window has already
    drifted 5x above the original pinned baseline, a new job at that same
    elevated duration must still trigger the warning, because the comparison
    is against the frozen baseline row -- never recomputed from the recent
    window's (already-drifted) durations. A moving-median baseline would have
    silently absorbed this drift and never fired, which is exactly the
    failure mode that produced this plan's 86-minute-median incident."""
    original_baseline_ms = 1000
    drifted_duration_ms = original_baseline_ms * 5

    conn = _drift_conn(
        {"insight_extraction": {"duration_ms": drifted_duration_ms}},
        [("insight_extraction", original_baseline_ms)],
    )
    with patch("rag.worker.settings") as mock_settings:
        mock_settings.STAGE_DRIFT_WARN_FACTOR = 3.0
        check_stage_drift(conn, "job-new")

    mock_log.warning.assert_called_once()
    _, kwargs = mock_log.warning.call_args
    assert kwargs["baseline_ms"] == original_baseline_ms
    assert kwargs["duration_ms"] == drifted_duration_ms


def test_claim_next_job_returns_none_when_empty():
    conn = _make_conn(fetchone=None)

    result = claim_next_job(conn)

    assert result is None
    conn.rollback.assert_called_once()


def test_claim_next_job_claims_and_transitions():
    job_uuid = uuid.uuid4()
    source_uuid = uuid.uuid4()
    conn = _make_conn(fetchone=(job_uuid, source_uuid, None))

    result = claim_next_job(conn)

    assert result is not None
    job_id, source_id, start_stage = result
    assert job_id == str(job_uuid)
    assert source_id == str(source_uuid)
    assert start_stage == "parsing"

    # Verify the UPDATE was called with the expected stage transition.
    update_call = conn.execute.call_args_list[1]
    sql = update_call[0][0]
    params = update_call[0][1]
    assert "UPDATE jobs SET status = %s" in sql
    assert params == ("processing:parsing", "parsing", str(job_uuid))

    conn.commit.assert_called_once()
