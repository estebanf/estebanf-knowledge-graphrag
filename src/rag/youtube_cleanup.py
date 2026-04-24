from __future__ import annotations

from collections.abc import Sequence

from rag.db import get_connection
from rag.graph_db import get_graph_driver
from rag.ingestion import delete_source_artifacts
from rag.storage import delete_stored_file


def list_youtube_sources(conn, source_id: str | None = None, limit: int | None = None) -> list[dict[str, str]]:
    sql = """
        SELECT id, COALESCE(name, ''), COALESCE(file_name, '')
        FROM sources
        WHERE deleted_at IS NULL
          AND metadata->>'kind' = 'youtube'
    """
    params: list[object] = []

    if source_id:
        sql += " AND id = %s"
        params.append(source_id)

    sql += " ORDER BY created_at"

    if limit is not None:
        sql += " LIMIT %s"
        params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [
        {"source_id": str(row[0]), "name": row[1], "file_name": row[2]}
        for row in rows
    ]


def purge_youtube_sources(
    conn,
    driver,
    execute: bool = False,
    source_id: str | None = None,
    limit: int | None = None,
    delete_source_artifacts_fn=delete_source_artifacts,
    delete_stored_file_fn=delete_stored_file,
    matches: Sequence[dict[str, str]] | None = None,
) -> list[str]:
    matched_sources = list(matches) if matches is not None else list_youtube_sources(
        conn,
        source_id=source_id,
        limit=limit,
    )

    if not matched_sources:
        prefix = "DRY RUN - " if not execute else ""
        print(f"{prefix}No matching youtube sources found.")
        return []

    prefix = "DRY RUN - " if not execute else ""
    print(f"{prefix}Found {len(matched_sources)} matching youtube source(s):")
    for match in matched_sources:
        print(
            f"  {match['source_id']} "
            f"name={match['name'] or '-'} "
            f"file={match['file_name'] or '-'}"
        )

    if not execute:
        print("Re-run with --execute to delete these sources.")
        return []

    deleted: list[str] = []
    for match in matched_sources:
        matched_source_id = match["source_id"]
        delete_source_artifacts_fn(conn, driver, matched_source_id)
        conn.commit()
        delete_stored_file_fn(matched_source_id)
        deleted.append(matched_source_id)
        print(f"Deleted {matched_source_id}")

    print(f"Deleted {len(deleted)} youtube source(s).")
    return deleted


def run(execute: bool = False, source_id: str | None = None, limit: int | None = None) -> list[str]:
    with get_connection() as conn:
        with get_graph_driver() as driver:
            return purge_youtube_sources(
                conn=conn,
                driver=driver,
                execute=execute,
                source_id=source_id,
                limit=limit,
            )
