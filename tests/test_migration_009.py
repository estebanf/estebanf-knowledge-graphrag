"""Validate binary prefilter indexes for 4096-dimensional retrieval."""

from pathlib import Path

MIGRATION = Path(__file__).resolve().parent.parent / "scripts" / "migrate" / "009_binary_vector_prefilter_indexes.sql"


def test_migration_file_exists() -> None:
    assert MIGRATION.is_file()


def test_migration_creates_binary_hnsw_indexes() -> None:
    sql = MIGRATION.read_text()
    assert "chunks_embedding_binary_hnsw_idx" in sql
    assert "insights_embedding_binary_hnsw_idx" in sql
    assert "USING hnsw ((binary_quantize(embedding)::bit(4096)) bit_hamming_ops)" in sql


def test_migration_creates_insight_sparse_index() -> None:
    sql = MIGRATION.read_text()
    assert "insights_content_fts_idx" in sql
    assert "USING gin (to_tsvector('english', coalesce(content, '')))" in sql


def test_migration_uses_concurrent_idempotent_index_creation() -> None:
    sql = MIGRATION.read_text()
    assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" in sql
    assert "BEGIN" not in sql
    assert "COMMIT" not in sql
