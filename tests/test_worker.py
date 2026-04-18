import uuid
from unittest.mock import MagicMock, call, patch

import psycopg.types.json

from rag.worker import claim_next_job, recover_stuck_jobs


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


def test_claim_next_job_returns_none_when_empty():
    conn = _make_conn(fetchone=None)

    result = claim_next_job(conn)

    assert result is None
    conn.rollback.assert_called_once()


def test_claim_next_job_claims_and_transitions():
    job_uuid = uuid.uuid4()
    source_uuid = uuid.uuid4()
    conn = _make_conn(fetchone=(job_uuid, source_uuid))

    result = claim_next_job(conn)

    assert result is not None
    job_id, source_id = result
    assert job_id == str(job_uuid)
    assert source_id == str(source_uuid)

    # Verify the UPDATE was called with 'processing:parsing'
    update_call = conn.execute.call_args_list[1]
    sql = update_call[0][0]
    assert "processing:parsing" in sql

    conn.commit.assert_called_once()
