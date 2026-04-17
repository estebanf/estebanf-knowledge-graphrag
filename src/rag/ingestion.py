import hashlib
import uuid
from pathlib import Path

import psycopg

from rag.db import get_connection
from rag.metadata_extraction import extract_metadata
from rag.parser import ParseError, parse_to_markdown
from rag.storage import delete_stored_file, store_file


def compute_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def check_duplicate(conn: psycopg.Connection, md5: str) -> str | None:
    row = conn.execute(
        "SELECT id FROM sources WHERE md5 = %s AND deleted_at IS NULL",
        (md5,),
    ).fetchone()
    return str(row[0]) if row else None


def ingest_file(
    file_path: Path,
    name: str | None = None,
    metadata: dict | None = None,
) -> dict:
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    md5 = compute_md5(file_path)
    source_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    file_type = file_path.suffix.lstrip(".").lower()
    source_name = name or file_path.stem

    with get_connection() as conn:
        existing = check_duplicate(conn, md5)
        if existing:
            raise ValueError(f"Duplicate: file already ingested as source {existing}")

        stored_path = store_file(source_id, file_path)

        try:
            conn.execute(
                """
                INSERT INTO sources (id, name, file_name, file_type, storage_path, md5, version, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, 1, %s)
                """,
                (
                    source_id,
                    source_name,
                    file_path.name,
                    file_type,
                    str(stored_path),
                    md5,
                    psycopg.types.json.Jsonb(metadata or {}),
                ),
            )

            conn.execute(
                """
                INSERT INTO jobs (id, source_id, status, current_stage, stage_log)
                VALUES (%s, %s, 'processing:parsing', 'parsing', '[]'::jsonb)
                """,
                (job_id, source_id),
            )
            conn.commit()

            markdown = parse_to_markdown(stored_path)

            # LLM-extracted metadata; user-supplied values take precedence
            extracted = extract_metadata(markdown)
            combined_metadata = {**extracted, **(metadata or {})}

            conn.execute(
                "UPDATE sources SET markdown_content = %s, metadata = %s WHERE id = %s",
                (markdown, psycopg.types.json.Jsonb(combined_metadata), source_id),
            )
            conn.execute(
                "UPDATE jobs SET status = 'completed', current_stage = 'completed', updated_at = now() WHERE id = %s",
                (job_id,),
            )
            conn.commit()

        except ParseError:
            conn.execute(
                "UPDATE jobs SET status = 'failed:parsing', current_stage = 'parsing', updated_at = now() WHERE id = %s",
                (job_id,),
            )
            conn.commit()
            raise

        except Exception:
            conn.rollback()
            delete_stored_file(source_id)
            raise

    return {"source_id": source_id, "job_id": job_id, "status": "completed"}
