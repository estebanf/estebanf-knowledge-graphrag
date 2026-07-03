from contextlib import contextmanager
from typing import Generator

import psycopg

from rag.config import settings

# Binary-quantized HNSW prefilter indexes that every dense-retrieval query
# walks (scripts/migrate/009_binary_vector_prefilter_indexes.sql). Keeping them
# resident in shared_buffers is what makes the first query after a restart fast.
PREWARM_INDEXES = (
    "chunks_embedding_binary_hnsw_idx",
    "insights_embedding_binary_hnsw_idx",
)


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    conn = psycopg.connect(settings.POSTGRES_URL)
    try:
        yield conn
    finally:
        conn.close()


def prewarm_vector_indexes() -> list[int]:
    """Load the binary-quantized HNSW vector indexes into shared_buffers.

    Called at backend startup so the first dense-retrieval query after a restart
    pays no page-in penalty. Opens its own autocommit connection rather than
    reusing a caller's. Best-effort per index: a missing pg_prewarm extension
    (fresh install before migration 011) or an absent index (before migration
    009) is skipped rather than raised. Connection-level failures (Postgres not
    yet accepting connections) do propagate, so the caller can retry.

    Returns the block count pg_prewarm reported for each index it warmed.
    """
    conn = psycopg.connect(settings.POSTGRES_URL, autocommit=True)
    blocks: list[int] = []
    try:
        for index_name in PREWARM_INDEXES:
            try:
                row = conn.execute("SELECT pg_prewarm(%s)", (index_name,)).fetchone()
            except psycopg.errors.UndefinedFunction:
                # pg_prewarm extension not installed — nothing to warm.
                break
            except psycopg.errors.UndefinedTable:
                # Index not created yet on this database; skip it.
                continue
            if row is not None:
                blocks.append(row[0])
    finally:
        conn.close()
    return blocks


def vacuum_analyze_entities() -> None:
    """Reclaim dead rows left by entity-merge scripts.

    VACUUM cannot run inside a transaction block, so this opens its own
    autocommit connection rather than reusing a caller's connection.
    """
    conn = psycopg.connect(settings.POSTGRES_URL, autocommit=True)
    try:
        conn.execute("VACUUM (ANALYZE) entities")
    finally:
        conn.close()
