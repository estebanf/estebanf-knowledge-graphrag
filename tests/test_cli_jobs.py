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


def test_jobs_list_shows_legacy_graph_linking_stage_log_without_crashing():
    """A pre-U5 job whose stage_log/current_stage still names the removed
    `graph_linking` stage must render fine in `rag jobs list` — listing is
    read-only over stage_log keys and STAGE_ORDER no longer needs to contain
    the name for rendering to work."""
    row = _make_job_row(
        status="completed",
        stage="graph_linking",
        stage_log={
            "graph_extraction": "2026-04-17T00:00:00",
            "graph_linking": "2026-04-17T00:00:01",
        },
    )
    with patch("rag.cli.get_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [row]
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "list"])
    assert result.exit_code == 0
    assert "job-1" in result.output


@patch("rag.cli.get_connection")
def test_jobs_status_shows_legacy_graph_linking_stage_log_without_crashing(mock_conn):
    """Same guarantee for `rag jobs status`: a legacy-shaped stage_log
    containing the removed stage name renders without error."""
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = (
        "job-uuid", "src-uuid", "completed", "completed",
        {
            "graph_extraction": "2026-04-18T10:00:00",
            "graph_linking": "2026-04-18T10:00:01",
            "insight_extraction": "2026-04-18T10:00:02",
        },
        "2026-04-18 10:00:00", "2026-04-18 10:01:00",
        None,
    )
    from rag.cli import app
    result = runner.invoke(app, ["jobs", "status", "job-uuid"])
    assert result.exit_code == 0
    assert "job-uuid" in result.output


def test_jobs_stats_direct_db_renders_percentiles():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = [
            ("chunking", 10, 500.0, 900.0, 1200),
            ("insight_extraction", 10, 7000.0, 9000.0, 12000),
        ]
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "stats"])
    assert result.exit_code == 0
    assert "chunking" in result.output
    assert "insight_extraction" in result.output
    assert "500" in result.output


def test_jobs_stats_direct_db_no_data():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.return_value.fetchall.return_value = []
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "stats"])
    assert result.exit_code == 0
    assert "No stage duration data" in result.output


def test_jobs_stats_rejects_oversized_days():
    from rag.cli import app
    result = runner.invoke(app, ["jobs", "stats", "--days", "9999"])
    assert result.exit_code == 1
    assert "--days must be between" in result.output


def test_jobs_stats_set_baseline_direct_db_upserts():
    with patch("rag.cli.get_connection") as mock_conn:
        conn = MagicMock()
        mock_conn.return_value.__enter__.return_value = conn
        conn.execute.side_effect = [
            MagicMock(fetchall=MagicMock(return_value=[("chunking", 10, 500.0, 900.0, 1200)])),
            MagicMock(),  # INSERT ... ON CONFLICT
            MagicMock(fetchall=MagicMock(return_value=[("chunking", 500, "2026-07-03 12:00:00")])),
        ]
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "stats", "--set-baseline"])
    assert result.exit_code == 0
    assert "chunking" in result.output
    assert "500" in result.output
    upsert_calls = [c for c in conn.execute.call_args_list if "stage_duration_baseline" in c.args[0]]
    assert upsert_calls, "expected an INSERT into stage_duration_baseline"


def test_jobs_stats_api_mode_renders_table():
    with patch("rag.cli._use_api", return_value=True), \
         patch("rag.cli._get_client") as mock_get_client:
        client = MagicMock()
        client.__enter__.return_value = client
        client.job_stage_stats.return_value = {
            "days": 14,
            "stats": [
                {"stage": "chunking", "job_count": 10, "p50_ms": 500.0, "p90_ms": 900.0, "max_ms": 1200},
            ],
        }
        mock_get_client.return_value = client
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "stats", "--days", "14"])
    assert result.exit_code == 0
    assert "chunking" in result.output
    client.job_stage_stats.assert_called_once_with(days=14)


def test_jobs_stats_api_mode_set_baseline():
    with patch("rag.cli._use_api", return_value=True), \
         patch("rag.cli._get_client") as mock_get_client:
        client = MagicMock()
        client.__enter__.return_value = client
        client.set_stage_stats_baseline.return_value = {
            "days": 14,
            "baselines": [{"stage": "chunking", "baseline_ms": 500, "set_at": "2026-07-03T12:00:00"}],
        }
        mock_get_client.return_value = client
        from rag.cli import app
        result = runner.invoke(app, ["jobs", "stats", "--set-baseline"])
    assert result.exit_code == 0
    assert "chunking" in result.output
    client.set_stage_stats_baseline.assert_called_once_with(days=14)


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
    assert "1 job submitted for retry" in result.output
