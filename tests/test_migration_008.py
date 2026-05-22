"""Validate that migration 008 declares the expected schema for auth + workers.

This is a static check (we cannot spin up Postgres in unit tests). End-to-end
application against a real DB is exercised in the E2E smoke (Phase 11).
"""

from pathlib import Path

MIGRATION = Path(__file__).resolve().parent.parent / "scripts" / "migrate" / "008_auth_and_workers.sql"


def test_migration_file_exists() -> None:
    assert MIGRATION.is_file()


def test_migration_creates_required_tables() -> None:
    sql = MIGRATION.read_text()
    for table in ("users", "user_sessions", "worker_processes", "api_key_audit"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql, f"missing CREATE TABLE for {table}"


def test_migration_is_idempotent() -> None:
    sql = MIGRATION.read_text()
    assert "IF NOT EXISTS" in sql
    assert "CREATE TABLE " not in sql.replace("CREATE TABLE IF NOT EXISTS", "")
    assert "CREATE INDEX " not in sql.replace("CREATE INDEX IF NOT EXISTS", "")


def test_users_table_has_required_columns() -> None:
    sql = MIGRATION.read_text()
    for col in ("username", "password_hash", "is_active", "created_at", "last_login_at"):
        assert col in sql


def test_user_sessions_references_users() -> None:
    sql = MIGRATION.read_text()
    assert "REFERENCES users(id)" in sql
    assert "expires_at" in sql
    assert "revoked_at" in sql


def test_worker_processes_status_constrained() -> None:
    sql = MIGRATION.read_text()
    for status in ("starting", "running", "stopped", "crashed"):
        assert f"'{status}'" in sql
    assert "log_path" in sql
    assert "exit_code" in sql


def test_api_key_audit_columns() -> None:
    sql = MIGRATION.read_text()
    for col in ("key_id", "route", "ts", "ip", "user_agent"):
        assert col in sql
