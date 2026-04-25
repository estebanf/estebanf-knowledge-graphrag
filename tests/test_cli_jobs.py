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


def test_jobs_list_failed_uses_prefix_filter():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--status", "failed"])

    assert result.exit_code == 0
    sql = conn.execute.call_args[0][0]
    params = conn.execute.call_args[0][1]
    assert "LIKE" in sql
    assert params == ("failed:%",)


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
    with patch(
        "rag.cli.retry_job",
        return_value={"job_id": "job-1", "status": "pending", "retry_from_stage": "chunking"},
    ) as mock_retry:
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "retry", "job-1"])
    assert result.exit_code == 0
    mock_retry.assert_called_once_with("job-1", from_stage=None)


def test_jobs_retry_with_from_stage():
    with patch(
        "rag.cli.retry_job",
        return_value={"job_id": "j", "status": "pending", "retry_from_stage": "chunking"},
    ) as mock_retry:
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
    with patch("rag.cli.cancel_job", create=True, return_value={"job_id": "job-1"}) as mock_cancel:
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "cancel", "job-1"])

    assert result.exit_code == 0
    mock_cancel.assert_called_once_with("job-1")


def test_jobs_cancel_rejects_completed_job():
    with patch("rag.cli.cancel_job", create=True, side_effect=ValueError("Job job-1 cannot be cancelled (status: completed)")):
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "cancel", "job-1"])
    assert result.exit_code == 1


def test_jobs_cancel_rejects_failed_job():
    with patch("rag.cli.cancel_job", create=True, side_effect=ValueError("Job job-1 cannot be cancelled (status: failed:chunking)")):
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "cancel", "job-1"])

    assert result.exit_code == 1
    assert "cannot be cancelled" in result.output


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


def test_jobs_list_stats_shows_counts():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = [
            ("completed", 42),
            ("failed", 5),
            ("pending", 3),
            ("processing", 1),
        ]
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--stats"])
    assert result.exit_code == 0
    assert "pending" in result.output
    assert "failed" in result.output
    assert "processing" in result.output
    assert "42" in result.output


def test_jobs_list_stats_groups_prefixed_statuses():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = [
            ("completed", 10),
            ("failed", 2),
            ("processing", 1),
        ]
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--stats"])
    assert result.exit_code == 0
    assert "completed" in result.output
    assert "failed" in result.output
    assert "processing" in result.output
    assert "10" in result.output
    assert "2" in result.output


def test_jobs_list_stats_empty_db():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--stats"])
    assert result.exit_code == 0
    assert "No jobs" in result.output


def test_jobs_list_retry_retries_all_failed():
    with patch("rag.cli.get_connection") as mock_conn, \
         patch("rag.cli.retry_job") as mock_retry:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = [
            ("job-1",), ("job-2",),
        ]
        mock_retry.return_value = {"job_id": "x", "status": "pending", "retry_from_stage": "chunking"}
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--retry"])
    assert result.exit_code == 0
    assert "2 jobs submitted for retry" in result.output
    assert mock_retry.call_count == 2


def test_jobs_list_retry_no_failed_jobs():
    with patch("rag.cli.get_connection") as mock_conn, \
         patch("rag.cli.retry_job") as mock_retry:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--retry"])
    assert result.exit_code == 0
    assert "No failed jobs" in result.output
    mock_retry.assert_not_called()


def test_jobs_list_retry_continues_on_per_job_error():
    with patch("rag.cli.get_connection") as mock_conn, \
         patch("rag.cli.retry_job") as mock_retry:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = [
            ("job-1",), ("job-2",),
        ]
        mock_retry.side_effect = [
            Exception("graph error"),
            {"job_id": "job-2", "status": "pending", "retry_from_stage": "chunking"},
        ]
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list", "--retry"])
    assert result.exit_code == 0
    assert mock_retry.call_count == 2
    assert "1 jobs submitted for retry" in result.output
