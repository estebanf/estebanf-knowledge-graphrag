from unittest.mock import MagicMock, patch

import pytest

from rag.remediation import ensure_schema_ready, remediate_image_source, remediate_source


def test_ensure_schema_ready_requires_entity_source_tracking():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None

    with pytest.raises(RuntimeError, match="entities.source_id"):
        ensure_schema_ready(conn)


@patch("rag.remediation._write_audit_log")
@patch("rag.remediation.cleanup_from_stage")
@patch("rag.remediation.verify_cleanup")
def test_remediate_source_requeues_completed_job_from_profiling(
    mock_verify_cleanup,
    mock_cleanup,
    mock_audit,
):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("completed",)

    remediate_source(conn, "driver", "source-1", "job-1", "file.pdf")

    mock_cleanup.assert_called_once_with(conn, "driver", "job-1", "source-1", "chunking")
    mock_verify_cleanup.assert_called_once_with(conn, "driver", "source-1")
    update_call = conn.execute.call_args_list[-1]
    assert "UPDATE jobs" in update_call.args[0]
    assert update_call.args[1] == ("job-1",)
    mock_audit.assert_called_once_with(
        conn,
        "job_retried",
        "job",
        "job-1",
        {"from_stage": "profiling", "reason": "heading_chunk_remediation", "file_name": "file.pdf"},
    )
    conn.commit.assert_called_once()


@patch("rag.remediation.cleanup_from_stage")
@patch("rag.remediation.verify_cleanup")
def test_remediate_source_rejects_non_completed_jobs(mock_verify_cleanup, mock_cleanup):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("failed:chunking",)

    with pytest.raises(RuntimeError, match="not completed"):
        remediate_source(conn, "driver", "source-1", "job-1", "file.pdf")

    mock_cleanup.assert_not_called()
    mock_verify_cleanup.assert_not_called()
    conn.commit.assert_not_called()


@patch("rag.remediation._write_audit_log")
@patch("rag.remediation.cleanup_from_stage")
@patch("rag.remediation.verify_cleanup")
def test_remediate_image_source_requeues_completed_job_from_parsing(
    mock_verify_cleanup,
    mock_cleanup,
    mock_audit,
):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("completed",)
    conn.execute.return_value.fetchall.return_value = []

    remediate_image_source(conn, "driver", "source-1", "job-1", "file.pdf")

    mock_cleanup.assert_called_once_with(conn, "driver", "job-1", "source-1", "parsing")
    mock_verify_cleanup.assert_called_once_with(conn, "driver", "source-1")
    update_call = [c for c in conn.execute.call_args_list if "UPDATE jobs" in str(c)]
    assert len(update_call) == 1
    assert "parsing" in str(update_call[0])
    mock_audit.assert_called_once()
    conn.commit.assert_called_once()


@patch("rag.remediation.cleanup_from_stage")
@patch("rag.remediation.verify_cleanup")
def test_remediate_image_source_rejects_non_completed_jobs(mock_verify_cleanup, mock_cleanup):
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = ("failed:parsing",)

    with pytest.raises(RuntimeError, match="not completed"):
        remediate_image_source(conn, "driver", "source-1", "job-1", "file.pdf")

    mock_cleanup.assert_not_called()
    conn.commit.assert_not_called()
