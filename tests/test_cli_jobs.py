from typer.testing import CliRunner
from unittest.mock import patch, MagicMock

runner = CliRunner()


def _make_job_row(
    job_id="job-1", source_id="src-1", status="completed",
    stage="completed", stage_log=None, created="2026-04-17T00:00:00",
    error_detail=None,
):
    return (job_id, source_id, status, stage, stage_log or {}, created, created, error_detail)


def test_jobs_list_no_jobs():
    with patch("rag.cli.get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = []
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list"])
    assert result.exit_code == 0
    assert "No jobs" in result.output


def test_jobs_list_shows_jobs():
    row = _make_job_row()
    with patch("rag.cli.get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [row]
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list"])
    assert result.exit_code == 0
    assert "job-1" in result.output


def test_jobs_list_with_status_filter():
    with patch("rag.cli.get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = []
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--status", "failed:chunking"])
    assert result.exit_code == 0


def test_jobs_status_shows_record():
    row = _make_job_row(stage_log={"parsing": "2026-04-17T00:00:00"})
    with patch("rag.cli.get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = row
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "status", "job-1"])
    assert result.exit_code == 0
    assert "job-1" in result.output


def test_jobs_status_not_found():
    with patch("rag.cli.get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = None
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "status", "nonexistent"])
    assert result.exit_code == 1


def test_jobs_retry_calls_retry_job():
    with patch("rag.cli.retry_job", return_value={"job_id": "job-1", "status": "completed"}) as mock_retry:
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "retry", "job-1"])
    assert result.exit_code == 0
    mock_retry.assert_called_once_with("job-1", from_stage=None)


def test_jobs_retry_with_from_stage():
    with patch("rag.cli.retry_job", return_value={"job_id": "j", "status": "completed"}) as mock_retry:
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "retry", "job-1", "--from-stage", "chunking"])
    mock_retry.assert_called_once_with("job-1", from_stage="chunking")


def test_jobs_retry_shows_error_on_failure():
    with patch("rag.cli.retry_job", side_effect=ValueError("Job not in failed state")):
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "retry", "job-1"])
    assert result.exit_code == 1
    assert "Job not in failed state" in result.output


def test_jobs_cancel_sets_cancelled_status():
    with patch("rag.cli.get_connection") as mock_conn:
        conn_mock = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn_mock
        conn_mock.execute.return_value.fetchone.return_value = ("job-1", "src-1", "processing:chunking")
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "cancel", "job-1"])
    assert result.exit_code == 0
    update_calls = [str(c) for c in conn_mock.execute.call_args_list]
    assert any("cancelled" in c for c in update_calls)


def test_jobs_cancel_rejects_completed_job():
    with patch("rag.cli.get_connection") as mock_conn:
        conn_mock = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn_mock
        conn_mock.execute.return_value.fetchone.return_value = ("job-1", "src-1", "completed")
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "cancel", "job-1"])
    assert result.exit_code == 1


@patch("rag.cli.get_connection")
def test_jobs_status_shows_error_detail(mock_conn):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = (
        "job-uuid", "src-uuid", "failed:parsing", "parsing",
        {"parsing": "2026-04-18T10:00:00"},
        "2026-04-18 10:00:00", "2026-04-18 10:01:00",
        {"stage": "parsing", "message": "Parse failed", "traceback": "trace..."},
    )
    from rag.cli import app
    result = runner.invoke(app, ["jobs", "status", "job-uuid"])
    assert result.exit_code == 0
    assert "Error Detail" in result.output
    assert "Parse failed" in result.output


@patch("rag.cli.get_connection")
def test_jobs_status_no_error_detail(mock_conn):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = (
        "job-uuid", "src-uuid", "completed", "completed",
        {"parsing": "2026-04-18T10:00:00"},
        "2026-04-18 10:00:00", "2026-04-18 10:01:00",
        None,
    )
    from rag.cli import app
    result = runner.invoke(app, ["jobs", "status", "job-uuid"])
    assert result.exit_code == 0
    assert "Error Detail" not in result.output
