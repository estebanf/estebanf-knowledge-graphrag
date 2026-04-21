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
