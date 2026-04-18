from unittest.mock import MagicMock, patch

import pytest

from rag.ingestion import cancel_job


@patch("rag.ingestion.get_connection")
def test_cancel_job_rejects_failed_status(mock_conn):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("job-1", "src-1", "failed:chunking", "chunking")

    with pytest.raises(ValueError, match="cannot be cancelled"):
        cancel_job("job-1")


@patch("rag.ingestion._write_audit_log")
@patch("rag.ingestion.get_connection")
def test_cancel_job_pending_does_not_cleanup(mock_conn, mock_audit):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("job-1", "src-1", "pending", None)

    result = cancel_job("job-1")

    assert result == {"job_id": "job-1", "status": "cancelled"}
    sql_calls = [call.args[0] for call in conn.execute.call_args_list]
    assert not any("DELETE FROM chunks" in sql for sql in sql_calls)
    mock_audit.assert_called_once()
    conn.commit.assert_called_once()


@patch("rag.ingestion._write_audit_log")
@patch("rag.ingestion.cleanup_from_stage")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.get_connection")
def test_cancel_job_processing_cleans_up_current_stage(mock_conn, mock_driver, mock_cleanup, mock_audit):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("job-1", "src-1", "processing:embedding", "embedding")
    mock_driver.return_value.__enter__.return_value = "driver"

    result = cancel_job("job-1")

    assert result == {"job_id": "job-1", "status": "cancelled"}
    mock_cleanup.assert_called_once_with(conn, "driver", "job-1", "src-1", "embedding")
    mock_audit.assert_called_once()
    conn.commit.assert_called_once()
