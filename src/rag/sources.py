import json
from typing import Callable

from rag.db import get_connection


def get_source_detail(
    source_id: str,
    *,
    connection_factory: Callable = get_connection,
) -> dict | None:
    with connection_factory() as conn:
        row = conn.execute(
            """
            SELECT id, name, file_name, file_type, storage_path, metadata, markdown_content
            FROM sources
            WHERE id = %s AND deleted_at IS NULL
            """,
            (source_id,),
        ).fetchone()

    if row is None:
        return None

    # Existing CLI tests stub a legacy one-column markdown-only query result.
    if len(row) == 1:
        return {
            "source_id": source_id,
            "name": None,
            "file_name": None,
            "file_type": None,
            "storage_path": "",
            "metadata": {},
            "markdown_content": row[0] or "",
        }

    return {
        "source_id": str(row[0]),
        "name": row[1],
        "file_name": row[2],
        "file_type": row[3],
        "storage_path": row[4] or "",
        "metadata": row[5] or {},
        "markdown_content": row[6] or "",
    }


def list_recent_sources(
    *,
    limit: int = 20,
    offset: int = 0,
    metadata_filters: list[tuple[str, str]] | None = None,
    connection_factory: Callable = get_connection,
) -> dict:
    metadata_filters = metadata_filters or []
    where_sql = "s.deleted_at IS NULL"
    params: list = []
    if metadata_filters:
        clauses = []
        for key, value in metadata_filters:
            clauses.append("s.metadata @> %s::jsonb")
            params.append(json.dumps({key: value}))
        where_sql += f" AND ({' OR '.join(clauses)})"

    with connection_factory() as conn:
        total_row = conn.execute(
            f"SELECT COUNT(*) FROM sources s WHERE {where_sql}",
            tuple(params),
        ).fetchone()
        rows = conn.execute(
            f"""
            SELECT s.id, s.name, s.file_name, s.file_type, s.metadata, s.created_at,
                   COUNT(DISTINCT ci.insight_id) AS insight_count
            FROM sources s
            LEFT JOIN chunks c ON c.source_id = s.id AND c.deleted_at IS NULL
            LEFT JOIN chunk_insights ci ON ci.chunk_id = c.id
            WHERE {where_sql}
            GROUP BY s.id, s.name, s.file_name, s.file_type, s.metadata, s.created_at
            ORDER BY s.created_at DESC
            LIMIT %s
            OFFSET %s
            """,
            (*params, limit, offset),
        ).fetchall()

    return {
        "sources": [
            {
                "source_id": str(row[0]),
                "name": row[1],
                "file_name": row[2],
                "file_type": row[3],
                "metadata": row[4] or {},
                "created_at": row[5],
                "insight_count": int(row[6] or 0),
            }
            for row in rows
        ],
        "total": int(total_row[0] or 0) if total_row else 0,
    }


def list_source_insights(
    source_id: str,
    *,
    connection_factory: Callable = get_connection,
) -> list[dict]:
    with connection_factory() as conn:
        rows = conn.execute(
            """
            SELECT i.id, i.content, ci.topics, c.id, c.chunk_index, c.content
            FROM chunks c
            JOIN chunk_insights ci ON ci.chunk_id = c.id
            JOIN insights i ON i.id = ci.insight_id
            WHERE c.source_id = %s AND c.deleted_at IS NULL
            ORDER BY c.chunk_index NULLS LAST, i.created_at, i.id
            """,
            (source_id,),
        ).fetchall()

    return [
        {
            "insight_id": str(row[0]),
            "insight": row[1] or "",
            "topics": list(row[2] or []),
            "chunk_id": str(row[3]),
            "chunk_index": row[4],
            "chunk_preview": _preview(row[5] or ""),
        }
        for row in rows
    ]


def _preview(content: str, *, max_length: int = 180) -> str:
    normalized = " ".join(content.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"
