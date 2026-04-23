from __future__ import annotations

from dataclasses import dataclass

from rag.db import get_connection
from rag.graph_db import get_graph_driver
from rag.ingestion import _write_audit_log, cleanup_from_stage


IDENTIFY_HEADING_ONLY_SOURCES_SQL = """
SELECT
    s.id AS source_id,
    j.id AS job_id,
    s.file_name,
    COUNT(*) AS heading_only_chunks
FROM chunks c
JOIN sources s ON s.id = c.source_id
JOIN jobs j ON j.source_id = s.id
WHERE j.status = 'completed'
  AND c.deleted_at IS NULL
  AND c.content ~ '^#{1,6}\\s+\\S'
  AND c.content NOT LIKE '%' || E'\\n' || '%'
GROUP BY s.id, j.id, s.file_name
ORDER BY heading_only_chunks DESC, s.file_name ASC
"""


@dataclass(frozen=True)
class AffectedSource:
    source_id: str
    job_id: str
    file_name: str
    heading_only_chunks: int


def ensure_schema_ready(conn) -> None:
    row = conn.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'entities'
          AND column_name = 'source_id'
        """
    ).fetchone()
    if not row:
        raise RuntimeError(
            "entities.source_id is required for safe remediation. "
            "Apply migration scripts/migrate/003_add_entity_source_tracking.sql first."
        )


def get_affected_sources(conn, only_source_id: str | None = None, limit: int | None = None) -> list[AffectedSource]:
    rows = conn.execute(IDENTIFY_HEADING_ONLY_SOURCES_SQL).fetchall()
    affected = [
        AffectedSource(
            source_id=str(row[0]),
            job_id=str(row[1]),
            file_name=row[2],
            heading_only_chunks=int(row[3]),
        )
        for row in rows
    ]
    if only_source_id:
        affected = [row for row in affected if row.source_id == only_source_id]
    if limit is not None:
        affected = affected[:limit]
    return affected


def _scalar_postgres(conn, sql: str, *params) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def _scalar_graph(driver, query: str, **params) -> int:
    if not driver:
        return 0
    with driver.session() as session:
        result = session.run(query, **params)
        row = result.single()
        if not row:
            return 0
        value = row[0]
        return int(value or 0)


def get_preflight_counts(conn, driver, source_id: str) -> dict[str, int]:
    return {
        "postgres_chunks": _scalar_postgres(
            conn,
            "SELECT COUNT(*) FROM chunks WHERE source_id = %s AND deleted_at IS NULL",
            source_id,
        ),
        "postgres_entities": _scalar_postgres(
            conn,
            "SELECT COUNT(*) FROM entities WHERE source_id = %s",
            source_id,
        ),
        "memgraph_chunks": _scalar_graph(
            driver,
            "MATCH (c:Chunk {source_id: $source_id}) RETURN count(c)",
            source_id=source_id,
        ),
        "memgraph_sources": _scalar_graph(
            driver,
            "MATCH (s:Source {source_id: $source_id}) RETURN count(s)",
            source_id=source_id,
        ),
    }


def verify_cleanup(conn, driver, source_id: str) -> None:
    counts = get_preflight_counts(conn, driver, source_id)
    remaining = {name: count for name, count in counts.items() if count != 0}
    if remaining:
        details = ", ".join(f"{name}={count}" for name, count in sorted(remaining.items()))
        raise RuntimeError(f"Cleanup verification failed for source {source_id}: {details}")


def verify_graph_entity_cleanup(driver, entity_ids: list[str]) -> None:
    if not entity_ids or not driver:
        return
    remaining = _scalar_graph(
        driver,
        "MATCH (e:Entity) WHERE e.entity_id IN $entity_ids RETURN count(e)",
        entity_ids=entity_ids,
    )
    if remaining:
        raise RuntimeError(
            f"Cleanup verification failed for entity nodes: remaining={remaining}"
        )


def remediate_source(conn, driver, source_id: str, job_id: str, file_name: str) -> None:
    row = conn.execute(
        "SELECT status FROM jobs WHERE id = %s",
        (job_id,),
    ).fetchone()
    if not row:
        raise RuntimeError(f"Job not found: {job_id}")
    if row[0] != "completed":
        raise RuntimeError(f"Job {job_id} is not completed (status: {row[0]})")

    entity_rows = conn.execute(
        "SELECT id FROM entities WHERE source_id = %s",
        (source_id,),
    ).fetchall()
    entity_ids = [str(row[0]) for row in entity_rows]

    cleanup_from_stage(conn, driver, job_id, source_id, "chunking")
    verify_cleanup(conn, driver, source_id)
    verify_graph_entity_cleanup(driver, entity_ids)
    conn.execute(
        """
        UPDATE jobs
        SET status = 'pending',
            current_stage = NULL,
            retry_from_stage = 'profiling',
            error_detail = NULL,
            updated_at = now()
        WHERE id = %s
        """,
        (job_id,),
    )
    _write_audit_log(
        conn,
        "job_retried",
        "job",
        job_id,
        {"from_stage": "profiling", "reason": "heading_chunk_remediation", "file_name": file_name},
    )
    conn.commit()


def remediate(dry_run: bool = False, only_source_id: str | None = None, limit: int | None = None) -> list[AffectedSource]:
    with get_connection() as conn:
        ensure_schema_ready(conn)
        affected = get_affected_sources(conn, only_source_id=only_source_id, limit=limit)

    if dry_run or not affected:
        return affected

    with get_graph_driver() as driver:
        for source in affected:
            with get_connection() as conn:
                remediate_source(conn, driver, source.source_id, source.job_id, source.file_name)
    return affected
