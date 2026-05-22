"""Tests for the jobs API: list, status, retry, cancel."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from rag.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


@patch("rag.api.routes.jobs.get_connection")
def test_list_jobs_returns_recent_jobs(mock_get_conn) -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = [
        ("job-1", "src-1", "completed", None, {}, datetime(2026, 5, 1, 12), datetime(2026, 5, 1, 12, 30)),
        ("job-2", "src-2", "failed:chunking", "chunking", {"err": "x"}, datetime(2026, 5, 1, 11), datetime(2026, 5, 1, 11, 5)),
    ]
    mock_get_conn.return_value = conn

    resp = _client().get("/api/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["jobs"]) == 2
    assert body["jobs"][0]["id"] == "job-1"
    assert body["jobs"][0]["status"] == "completed"
    assert body["jobs"][1]["status"] == "failed:chunking"


@patch("rag.api.routes.jobs.get_connection")
def test_list_jobs_supports_status_filter(mock_get_conn) -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = []
    mock_get_conn.return_value = conn

    _client().get("/api/jobs?status=failed")
    # status=failed should map to LIKE 'failed:%' in the query.
    sql, params = conn.execute.call_args[0]
    assert "LIKE" in sql
    assert params == ("failed:%",)


@patch("rag.api.routes.jobs.get_connection")
def test_list_jobs_stats_endpoint(mock_get_conn) -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchall.return_value = [
        ("completed", 10),
        ("failed", 2),
        ("pending", 1),
    ]
    mock_get_conn.return_value = conn

    resp = _client().get("/api/jobs/stats")
    assert resp.status_code == 200
    assert resp.json() == {"stats": [
        {"status": "completed", "count": 10},
        {"status": "failed", "count": 2},
        {"status": "pending", "count": 1},
    ]}


@patch("rag.api.routes.jobs.get_connection")
def test_get_job_status_returns_detail(mock_get_conn) -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = (
        "job-1", "src-1", "completed", None, {"chunks": 12}, datetime(2026, 5, 1), datetime(2026, 5, 1, 0, 30), None,
    )
    mock_get_conn.return_value = conn

    resp = _client().get("/api/jobs/job-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "job-1"
    assert body["stage_log"] == {"chunks": 12}


@patch("rag.api.routes.jobs.get_connection")
def test_get_job_status_404(mock_get_conn) -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = None
    mock_get_conn.return_value = conn

    resp = _client().get("/api/jobs/missing")
    assert resp.status_code == 404


@patch("rag.api.routes.jobs.retry_job")
def test_retry_job_endpoint(mock_retry) -> None:
    mock_retry.return_value = {"job_id": "new-job", "retry_from_stage": "chunking"}
    resp = _client().post("/api/jobs/old-job/retry", json={"from_stage": "chunking"})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"job_id": "new-job", "retry_from_stage": "chunking"}
    mock_retry.assert_called_once_with("old-job", from_stage="chunking")


@patch("rag.api.routes.jobs.retry_job")
def test_retry_job_endpoint_handles_bad_request(mock_retry) -> None:
    mock_retry.side_effect = ValueError("Unknown stage: bogus")
    resp = _client().post("/api/jobs/x/retry", json={"from_stage": "bogus"})
    assert resp.status_code == 400


@patch("rag.api.routes.jobs.cancel_job")
def test_cancel_job_endpoint(mock_cancel) -> None:
    mock_cancel.return_value = {"job_id": "job-1", "status": "cancelled"}
    resp = _client().post("/api/jobs/job-1/cancel")
    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-1", "status": "cancelled"}
