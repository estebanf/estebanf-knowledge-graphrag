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


@patch("rag.ingestion._write_audit_log")
@patch("rag.ingestion.extract_and_store_insights")
@patch("rag.ingestion.link_graph")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.get_connection")
def test_execute_pipeline_from_insight_stage_skips_graph_linking(
    mock_conn,
    mock_driver,
    mock_link_graph,
    mock_extract_insights,
    mock_audit,
):
    from rag.ingestion import execute_ingestion_pipeline

    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("/tmp/source.md", None, "# Existing")
    conn.execute.return_value.fetchall.return_value = [("chunk-1", "chunk text")]
    driver = MagicMock()
    mock_driver.return_value.__enter__.return_value = driver
    mock_extract_insights.return_value = {
        "chunks_processed": 1,
        "insights_extracted": 1,
        "insights_reused": 0,
    }

    result = execute_ingestion_pipeline("job-1", "source-1", start_stage="insight_extraction")

    assert result == {"source_id": "source-1", "job_id": "job-1", "status": "completed"}
    mock_link_graph.assert_not_called()
    mock_extract_insights.assert_called_once_with(conn, driver, "source-1", [("chunk-1", "chunk text")])
    stage_updates = [call.args for call in conn.execute.call_args_list if "UPDATE jobs SET status = %s" in call.args[0]]
    assert any(args[1][0] == "processing:insight_extraction" for args in stage_updates)
    mock_audit.assert_called()


def test_cleanup_from_insight_stage_removes_insight_artifacts():
    from rag.ingestion import cleanup_from_stage

    conn = MagicMock()
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    cleanup_from_stage(conn, driver, "job-1", "source-1", "insight_extraction")

    sql_calls = [call.args[0] for call in conn.execute.call_args_list]
    assert any("DELETE FROM chunk_insights" in sql for sql in sql_calls)
    assert any("DELETE FROM insights" in sql for sql in sql_calls)
    graph_queries = [call.args[0] for call in session.run.call_args_list]
    assert any("Insight" in query and "DETACH DELETE" in query for query in graph_queries)
    conn.commit.assert_called_once()
