from datetime import UTC, datetime, timedelta
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
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.get_connection")
def test_execute_pipeline_from_insight_stage_skips_graph_extraction(
    mock_conn,
    mock_driver,
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
        "failed_chunks": [],
        "related_edges": 0,
    }

    result = execute_ingestion_pipeline("job-1", "source-1", start_stage="insight_extraction")

    assert result == {"source_id": "source-1", "job_id": "job-1", "status": "completed"}
    mock_extract_insights.assert_called_once_with(conn, driver, "source-1", [("chunk-1", "chunk text")])
    stage_updates = [call.args for call in conn.execute.call_args_list if "UPDATE jobs SET status = %s" in call.args[0]]
    assert any(args[1][0] == "processing:insight_extraction" for args in stage_updates)
    mock_audit.assert_called()


def test_stage_order_has_no_graph_linking():
    """R14/KTD8: the no-op graph_linking stage is fully removed from the
    registry so newly created jobs never record it in stage_log."""
    from rag.ingestion import STAGE_ORDER

    assert "graph_linking" not in STAGE_ORDER
    assert STAGE_ORDER == [
        "parsing", "profiling", "chunking", "validation",
        "embedding", "graph_extraction", "insight_extraction",
    ]


@patch("rag.ingestion._write_audit_log")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.cleanup_from_stage")
@patch("rag.ingestion.get_connection")
def test_retry_job_remaps_legacy_graph_linking_stage_to_insight_extraction(
    mock_conn, mock_cleanup, mock_driver, mock_audit,
):
    """KTD8/R14: a legacy job whose failed status (or explicit --from-stage)
    names the removed `graph_linking` stage retries at `insight_extraction`
    instead of raising ValueError for an unknown stage."""
    from rag.ingestion import retry_job

    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("job-1", "source-1", "failed:graph_linking")
    mock_driver.return_value.__enter__.return_value = "driver"

    result = retry_job("job-1")

    assert result == {"job_id": "job-1", "status": "pending", "retry_from_stage": "insight_extraction"}
    mock_cleanup.assert_called_once_with(conn, "driver", "job-1", "source-1", "insight_extraction")


@patch("rag.ingestion._write_audit_log")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.cleanup_from_stage")
@patch("rag.ingestion.get_connection")
def test_retry_job_explicit_graph_linking_from_stage_remaps(
    mock_conn, mock_cleanup, mock_driver, mock_audit,
):
    """Same remap, but via an explicit `--from-stage graph_linking` request
    rather than a legacy failed-status string."""
    from rag.ingestion import retry_job

    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("job-1", "source-1", "failed:embedding")
    mock_driver.return_value.__enter__.return_value = "driver"

    result = retry_job("job-1", from_stage="graph_linking")

    assert result == {"job_id": "job-1", "status": "pending", "retry_from_stage": "insight_extraction"}
    mock_cleanup.assert_called_once_with(conn, "driver", "job-1", "source-1", "insight_extraction")


def test_execute_ingestion_pipeline_remaps_legacy_graph_linking_start_stage():
    """A worker that reads an old row's retry_from_stage = 'graph_linking'
    (persisted before this stage was removed) must resume at
    insight_extraction rather than raising 'Unknown stage'."""
    with patch("rag.ingestion.get_connection") as mock_conn, \
         patch("rag.ingestion.get_graph_driver") as mock_driver, \
         patch("rag.ingestion.extract_and_store_insights") as mock_extract_insights, \
         patch("rag.ingestion._write_audit_log"):
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
            "failed_chunks": [],
            "related_edges": 0,
        }

        result = execute_ingestion_pipeline("job-1", "source-1", start_stage="graph_linking")

        assert result == {"source_id": "source-1", "job_id": "job-1", "status": "completed"}
        mock_extract_insights.assert_called_once()


def test_cleanup_from_stage_legacy_graph_linking_matches_insight_extraction():
    """cleanup_from_stage("graph_linking") (legacy name) and
    cleanup_from_stage("insight_extraction") (its remap target) must behave
    identically — the remap doesn't change insight_extraction's own cleanup."""
    from rag.ingestion import cleanup_from_stage

    def _run(from_stage):
        conn = MagicMock()
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = lambda s: session
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        cleanup_from_stage(conn, driver, "job-1", "source-1", from_stage)
        return [call.args[0] for call in conn.execute.call_args_list]

    legacy_sql = _run("graph_linking")
    canonical_sql = _run("insight_extraction")
    assert legacy_sql == canonical_sql


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


def test_cleanup_from_insight_stage_removes_ghost_memgraph_insight_ae4b():
    """AE4b: a Memgraph Insight node reachable via CONTAINS from a chunk of
    this source, but with no matching Postgres `insights` row (simulating a
    mid-Phase-E crash where the Memgraph write committed but the Postgres
    transaction did not), is deleted by cleanup_from_stage — not just the
    CONTAINS-less orphans that `_cleanup_orphan_insights` already caught."""
    from rag.ingestion import cleanup_from_stage

    conn = MagicMock()
    # The ghost insight id has no matching row in Postgres `insights`.
    conn.execute.return_value.fetchall.return_value = []
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    # Call order inside cleanup: (1) CONTAINS-less orphan DETACH DELETE,
    # (2) source-scoped CONTAINS -> Insight lookup, (3) ghost DETACH DELETE.
    session.run.side_effect = [None, [{"insight_id": "ghost-1"}], None]

    cleanup_from_stage(conn, driver, "job-1", "source-1", "insight_extraction")

    assert session.run.call_count == 3
    lookup_call = session.run.call_args_list[1]
    assert "CONTAINS" in lookup_call.args[0]
    assert lookup_call.kwargs["source_id"] == "source-1"

    ghost_delete_call = session.run.call_args_list[2]
    assert "DETACH DELETE" in ghost_delete_call.args[0]
    assert ghost_delete_call.kwargs["ids"] == ["ghost-1"]


def test_cleanup_from_insight_stage_leaves_live_insight_nodes_alone():
    """Counterpart to AE4b: an Insight node whose id DOES have a matching
    Postgres row is left untouched by the ghost-node sweep."""
    from rag.ingestion import cleanup_from_stage

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [("live-1",)]
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    session.run.side_effect = [None, [{"insight_id": "live-1"}]]

    cleanup_from_stage(conn, driver, "job-1", "source-1", "insight_extraction")

    # Only the CONTAINS-less orphan sweep and the source-scoped lookup ran —
    # no ghost DETACH DELETE call, since the only insight found is live.
    assert session.run.call_count == 2


@patch("rag.ingestion._write_audit_log")
@patch("rag.ingestion.extract_and_store_insights")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.ingestion.get_connection")
def test_ae4_retry_after_mid_insight_extraction_failure_cleans_up_before_rerun(
    mock_conn,
    mock_driver,
    mock_extract_insights,
    mock_audit,
):
    """AE4: a job that failed mid-insight_extraction, retried via
    `retry_job` (the function `rag jobs retry` calls), runs
    `cleanup_from_stage` exactly once before the stage is re-run — the
    mechanism that keeps the retry from double-writing rows/edges."""
    from rag.ingestion import execute_ingestion_pipeline, retry_job

    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("/tmp/source.md", None, "# Existing")
    conn.execute.return_value.fetchall.return_value = [("chunk-1", "chunk text")]
    driver = MagicMock()
    mock_driver.return_value.__enter__.return_value = driver

    mock_extract_insights.side_effect = RuntimeError("simulated mid-Phase-C crash")

    with pytest.raises(RuntimeError):
        execute_ingestion_pipeline("job-1", "source-1", start_stage="insight_extraction")

    # Simulate `rag jobs retry job-1`.
    conn.execute.return_value.fetchone.return_value = ("job-1", "source-1", "failed:insight_extraction")
    retry_job("job-1")

    cleanup_sql_calls = [
        call for call in conn.execute.call_args_list
        if "DELETE FROM chunk_insights" in call.args[0]
    ]
    assert len(cleanup_sql_calls) == 1

    # Fix the crash and re-run the stage — this is the worker picking the
    # now-pending job back up.
    mock_extract_insights.side_effect = None
    mock_extract_insights.return_value = {
        "chunks_processed": 1,
        "insights_extracted": 1,
        "insights_reused": 0,
        "failed_chunks": [],
        "related_edges": 0,
    }
    conn.execute.return_value.fetchone.return_value = ("/tmp/source.md", None, "# Existing")

    result = execute_ingestion_pipeline("job-1", "source-1", start_stage="insight_extraction")

    assert result == {"source_id": "source-1", "job_id": "job-1", "status": "completed"}
    assert mock_extract_insights.call_count == 2
    # Cleanup ran exactly once, between the failed attempt and the
    # successful retry — not again during the successful run.
    cleanup_sql_calls_after = [
        call for call in conn.execute.call_args_list
        if "DELETE FROM chunk_insights" in call.args[0]
    ]
    assert len(cleanup_sql_calls_after) == 1


# --- U6: stage timing telemetry (R8) ---------------------------------------


def test_complete_stage_computes_duration_ms_from_stored_started_at():
    """`_complete_stage` reads the stage's own `started_at` back from the DB
    (not process-local state) and stores an honest `duration_ms`."""
    from rag.ingestion import _complete_stage

    started_at = (datetime.now(UTC) - timedelta(seconds=5)).replace(microsecond=0).isoformat()
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (started_at,)

    _complete_stage(conn, "job-1", "chunking", {"chunk_count": 3})

    update_call = conn.execute.call_args_list[-1]
    stage_entry = update_call.args[1][1].obj  # psycopg.types.json.Jsonb wraps the dict
    assert stage_entry["status"] == "completed"
    assert stage_entry["duration_ms"] is not None
    # Allow generous slack for test execution time.
    assert 4000 <= stage_entry["duration_ms"] <= 10000


def test_complete_stage_duration_ms_null_when_started_at_missing():
    """Legacy stage_log entries (predating this telemetry) have no
    `started_at` to compute from -- duration_ms is null, not a crash."""
    from rag.ingestion import _complete_stage

    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (None,)

    _complete_stage(conn, "job-1", "chunking", {})

    update_call = conn.execute.call_args_list[-1]
    stage_entry = update_call.args[1][1].obj
    assert stage_entry["duration_ms"] is None


@patch("rag.ingestion._write_audit_log")
def test_fail_stage_computes_duration_ms_from_stored_started_at(mock_audit):
    from rag.ingestion import _fail_stage

    started_at = (datetime.now(UTC) - timedelta(seconds=2)).replace(microsecond=0).isoformat()
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = (started_at,)

    _fail_stage(conn, "job-1", "embedding", RuntimeError("boom"))

    update_call = next(
        c for c in conn.execute.call_args_list
        if "UPDATE jobs" in c.args[0] and "stage_log" in c.args[0]
    )
    params = update_call.args[1]
    stage_entry = params[-2].obj
    assert stage_entry["status"] == "failed"
    assert stage_entry["duration_ms"] is not None
    assert stage_entry["duration_ms"] >= 1000


@patch("rag.ingestion._write_audit_log")
@patch("rag.ingestion.get_connection")
def test_parsing_stage_calls_update_stage_before_parsing(mock_conn, mock_audit):
    """R8/U6: the parsing block previously had no `_update_stage` call, so it
    never recorded a `started_at` and its `duration_ms` was always null. This
    asserts the fix: `_update_stage(conn, job_id, "parsing")` runs before
    `parse_document`."""
    from rag.ingestion import execute_ingestion_pipeline

    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn

    def _execute_side_effect(sql, params=None):
        result = MagicMock()
        if "SELECT s.storage_path" in sql:
            result.fetchone.return_value = ("/tmp/source.md", None, None)
        elif "SELECT metadata FROM sources" in sql:
            result.fetchone.return_value = ({},)
        elif "stage_log ->" in sql:
            result.fetchone.return_value = (None,)
        else:
            result.fetchone.return_value = None
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = _execute_side_effect

    with patch("rag.ingestion.parse_document") as mock_parse, \
         patch("rag.ingestion.extract_metadata", return_value={}), \
         patch("rag.ingestion.profile_document") as mock_profile, \
         patch("rag.ingestion.chunk_document", return_value=[]), \
         patch("rag.ingestion.validate_chunks", return_value=True), \
         patch("rag.ingestion.embed_and_store_chunks"), \
         patch("rag.ingestion.get_graph_driver") as mock_driver, \
         patch("rag.ingestion.extract_and_store_graph", return_value={}), \
         patch("rag.ingestion.extract_and_store_insights", return_value={
             "chunks_processed": 0, "insights_extracted": 0,
             "insights_reused": 0, "failed_chunks": [], "related_edges": 0,
         }):
        mock_parse.return_value = MagicMock(markdown="# doc")
        mock_profile.return_value = MagicMock(domain="general")
        mock_driver.return_value.__enter__.return_value = MagicMock()

        execute_ingestion_pipeline("job-1", "source-1", start_stage="parsing")

    update_stage_calls = [
        call for call in conn.execute.call_args_list
        if "current_stage = %s" in call.args[0] and call.args[1][1] == "parsing"
    ]
    assert update_stage_calls, "expected an _update_stage(..., 'parsing') UPDATE before parse_document ran"
    mock_parse.assert_called_once()
