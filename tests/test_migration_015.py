"""Validate migration 015 (Jev entity audit + review-queue tables).

Static check (no Postgres in unit tests): the DDL declares the expected tables,
is idempotent, and `scripts/init/postgres/02_schema.sql` carries the same
tables so fresh installs match migrated ones.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIGRATION = ROOT / "scripts" / "migrate" / "015_entity_audit.sql"
INIT_SCHEMA = ROOT / "scripts" / "init" / "postgres" / "02_schema.sql"

TABLES = ("entity_mention_audit", "entity_merge_review", "insight_pair_review")


def _strip_comments(sql: str) -> str:
    return "\n".join(line for line in sql.splitlines() if not line.strip().startswith("--"))


def _table_body(sql: str, table: str) -> str:
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\s*\);", sql, flags=re.DOTALL
    )
    assert match, f"missing CREATE TABLE IF NOT EXISTS {table}"
    return match.group(1)


def _columns(body: str) -> set[str]:
    cols = set()
    for line in body.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith(("PRIMARY KEY", "UNIQUE", "--")):
            continue
        cols.add(line.split()[0])
    return cols


def test_migration_file_exists() -> None:
    assert MIGRATION.is_file()


def test_migration_creates_required_tables() -> None:
    sql = MIGRATION.read_text()
    for table in TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql


def test_migration_is_idempotent() -> None:
    sql = MIGRATION.read_text()
    assert "CREATE TABLE " not in sql.replace("CREATE TABLE IF NOT EXISTS", "")
    assert "CREATE INDEX " not in sql.replace("CREATE INDEX IF NOT EXISTS", "")
    assert "DROP " not in _strip_comments(sql).upper()


def test_entity_mention_audit_columns_and_keys() -> None:
    sql = MIGRATION.read_text()
    body = _table_body(sql, "entity_mention_audit")
    assert _columns(body) == {
        "chunk_id",
        "entity_id",
        "is_entity",
        "entity_type",
        "type_confidence",
        "salience",
        "model",
        "scored_at",
    }
    assert "PRIMARY KEY (chunk_id, entity_id)" in body
    assert "scored_at       timestamptz NOT NULL DEFAULT now()" in body
    assert "entity_mention_audit_entity_id_idx ON entity_mention_audit(entity_id)" in sql
    assert "entity_mention_audit_scored_at_idx ON entity_mention_audit(scored_at)" in sql


def test_audit_table_has_no_foreign_keys() -> None:
    """entities/chunks are hard-deleted by merge and source-delete paths; an FK
    without cascade would break them. Stale audit rows are filtered by join."""
    sql = MIGRATION.read_text()
    assert "REFERENCES" not in _strip_comments(sql)


def test_review_tables_columns_and_unique_pairs() -> None:
    sql = MIGRATION.read_text()
    merge_body = _table_body(sql, "entity_merge_review")
    assert _columns(merge_body) == {
        "id", "entity_a", "entity_b", "score", "reason", "status", "created_at",
    }
    assert "UNIQUE (entity_a, entity_b)" in merge_body
    assert "bigserial" in merge_body
    assert "DEFAULT 'pending'" in merge_body

    pair_body = _table_body(sql, "insight_pair_review")
    assert _columns(pair_body) == {
        "id", "insight_a", "insight_b", "cosine", "score", "status", "created_at",
    }
    assert "UNIQUE (insight_a, insight_b)" in pair_body
    assert "DEFAULT 'pending'" in pair_body


def test_init_schema_contains_same_tables() -> None:
    migration = MIGRATION.read_text()
    init = INIT_SCHEMA.read_text()
    for table in TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in init
        assert _columns(_table_body(init, table)) == _columns(_table_body(migration, table))
    assert "entity_mention_audit_entity_id_idx" in init
    assert "entity_mention_audit_scored_at_idx" in init
