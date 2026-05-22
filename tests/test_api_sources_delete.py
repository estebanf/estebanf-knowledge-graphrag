"""Tests for DELETE /api/sources/{source_id}."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from rag.api.main import create_app


def _client() -> TestClient:
    return TestClient(create_app())


@patch("rag.api.routes.sources.get_connection")
def test_soft_delete_marks_source_deleted(mock_get_conn) -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("/path/to/file",)
    mock_get_conn.return_value = conn

    resp = _client().delete("/api/sources/source-1")
    assert resp.status_code == 200
    assert resp.json() == {"source_id": "source-1", "hard": False}
    # SELECT, then UPDATE deleted_at, then audit, then commit.
    sqls = [call.args[0] for call in conn.execute.call_args_list]
    assert any("UPDATE sources" in s and "deleted_at" in s for s in sqls)


@patch("rag.api.routes.sources.delete_stored_file")
@patch("rag.api.routes.sources.delete_source_artifacts")
@patch("rag.api.routes.sources.get_graph_driver")
@patch("rag.api.routes.sources.get_connection")
def test_hard_delete_invokes_artifact_cleanup(
    mock_get_conn, mock_get_driver, mock_delete_artifacts, mock_delete_file
) -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("/path/to/file",)
    mock_get_conn.return_value = conn

    driver = MagicMock()
    driver.__enter__.return_value = driver
    mock_get_driver.return_value = driver

    resp = _client().delete("/api/sources/source-1?hard=true")
    assert resp.status_code == 200
    assert resp.json() == {"source_id": "source-1", "hard": True}
    mock_delete_artifacts.assert_called_once_with(conn, driver, "source-1")
    mock_delete_file.assert_called_once_with("source-1")


@patch("rag.api.routes.sources.get_connection")
def test_delete_returns_404_when_missing(mock_get_conn) -> None:
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = None
    mock_get_conn.return_value = conn

    resp = _client().delete("/api/sources/missing")
    assert resp.status_code == 404
