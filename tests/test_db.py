"""Tests for rag.db vector-index prewarming and its startup retry wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import psycopg

from rag.api.main import _prewarm_with_retry
from rag.db import PREWARM_INDEXES, prewarm_vector_indexes


def _fake_conn(execute_side_effect):
    """Build a fake autocommit connection whose .execute() is scripted."""
    conn = MagicMock()
    conn.execute.side_effect = execute_side_effect
    return conn


def _result(block_count: int):
    cursor = MagicMock()
    cursor.fetchone.return_value = (block_count,)
    return cursor


def test_prewarm_issues_pg_prewarm_for_both_indexes() -> None:
    conn = _fake_conn([_result(11), _result(22), _result(33)])
    with patch("rag.db.psycopg.connect", return_value=conn) as connect:
        blocks = prewarm_vector_indexes()

    assert connect.call_args.kwargs.get("autocommit") is True
    warmed = [call.args[1][0] for call in conn.execute.call_args_list]
    assert warmed == list(PREWARM_INDEXES)
    assert blocks == [11, 22, 33]
    conn.close.assert_called_once()


def test_prewarm_skips_when_extension_missing() -> None:
    # First index raises UndefinedFunction (pg_prewarm not installed) -> stop.
    conn = _fake_conn([psycopg.errors.UndefinedFunction("no pg_prewarm")])
    with patch("rag.db.psycopg.connect", return_value=conn):
        blocks = prewarm_vector_indexes()

    assert blocks == []
    assert conn.execute.call_count == 1
    conn.close.assert_called_once()


def test_prewarm_skips_absent_index_but_continues() -> None:
    conn = _fake_conn([psycopg.errors.UndefinedTable("no such index"), _result(7), _result(8)])
    with patch("rag.db.psycopg.connect", return_value=conn):
        blocks = prewarm_vector_indexes()

    assert blocks == [7, 8]
    assert conn.execute.call_count == 3
    conn.close.assert_called_once()


def test_prewarm_retry_recovers_after_postgres_not_ready() -> None:
    # First attempt fails (Postgres still starting), a later one succeeds.
    attempts = MagicMock(side_effect=[psycopg.OperationalError("starting up"), [3, 4]])
    with patch("rag.api.main.prewarm_vector_indexes", attempts):
        asyncio.run(_prewarm_with_retry(attempts=3, base_delay=0))

    assert attempts.call_count == 2


def test_prewarm_retry_gives_up_without_raising() -> None:
    # All attempts fail -> logged and swallowed, never raised into startup.
    attempts = MagicMock(side_effect=psycopg.OperationalError("down"))
    with patch("rag.api.main.prewarm_vector_indexes", attempts):
        asyncio.run(_prewarm_with_retry(attempts=3, base_delay=0))

    assert attempts.call_count == 3
