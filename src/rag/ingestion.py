import hashlib
import uuid
from pathlib import Path

import psycopg

from rag.chunk_validation import validate_chunks
from rag.chunking import ChunkData, chunk_document
from rag.db import get_connection
from rag.embedding import embed_and_store_chunks
from rag.metadata_extraction import extract_metadata
from rag.parser import ParseError, parse_to_markdown
from rag.profiling import profile_document
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


def _update_stage(conn: psycopg.Connection, job_id: str, stage: str) -> None:
    conn.execute(
        "UPDATE jobs SET status = %s, current_stage = %s, updated_at = now() WHERE id = %s",
        (f"processing:{stage}", stage, job_id),
    )
    conn.commit()


def _fail_stage(conn: psycopg.Connection, job_id: str, stage: str) -> None:
    conn.execute(
        "UPDATE jobs SET status = %s, current_stage = %s, updated_at = now() WHERE id = %s",
        (f"failed:{stage}", stage, job_id),
    )
    conn.commit()


def _insert_chunks(
    conn: psycopg.Connection,
    source_id: str,
    job_id: str,
    chunks: list[ChunkData],
) -> list[tuple[str, str]]:
    """
    Insert chunks into DB. Returns list of (chunk_id, content) for embedding stage.

    ChunkData.parent_chunk_id may contain a digit string (the parent's chunk_index).
    This function resolves those digit strings to actual DB UUIDs after all chunks
    are inserted.
    """
    if not chunks:
        return []

    # First pass: insert all chunks, track index->uuid mapping
    index_to_uuid: dict[int, str] = {}
    chunk_ids: list[str] = []

    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        chunk_ids.append(chunk_id)
        index_to_uuid[chunk.chunk_index] = chunk_id

        conn.execute(
            """
            INSERT INTO chunks
              (id, source_id, job_id, content, token_count, chunk_index,
               parent_chunk_id, chunking_strategy, chunking_config, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                chunk_id,
                source_id,
                job_id,
                chunk.content,
                chunk.token_count,
                chunk.chunk_index,
                None,  # placeholder; resolved in second pass
                chunk.chunking_strategy,
                psycopg.types.json.Jsonb(chunk.chunking_config),
                psycopg.types.json.Jsonb(chunk.metadata),
            ),
        )

    # Second pass: resolve parent_chunk_id digit strings to UUIDs
    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk.parent_chunk_id is not None and chunk.parent_chunk_id.isdigit():
            parent_uuid = index_to_uuid.get(int(chunk.parent_chunk_id))
            if parent_uuid:
                conn.execute(
                    "UPDATE chunks SET parent_chunk_id = %s WHERE id = %s",
                    (parent_uuid, chunk_id),
                )

    return [(chunk_id, chunk.content) for chunk_id, chunk in zip(chunk_ids, chunks)]


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

            # --- Parsing ---
            try:
                markdown = parse_to_markdown(stored_path)
            except ParseError:
                _fail_stage(conn, job_id, "parsing")
                raise

            extracted = extract_metadata(markdown)
            combined_metadata = {**extracted, **(metadata or {})}
            conn.execute(
                "UPDATE sources SET markdown_content = %s, metadata = %s WHERE id = %s",
                (markdown, psycopg.types.json.Jsonb(combined_metadata), source_id),
            )
            conn.commit()

            # --- Profiling ---
            _update_stage(conn, job_id, "profiling")
            profile = profile_document(markdown)

            # --- Chunking ---
            _update_stage(conn, job_id, "chunking")
            try:
                chunks = chunk_document(markdown, profile)
                chunk_rows = _insert_chunks(conn, source_id, job_id, chunks)
                conn.commit()
            except Exception:
                _fail_stage(conn, job_id, "chunking")
                raise

            # --- Validation ---
            _update_stage(conn, job_id, "validation")
            try:
                passed = validate_chunks(chunks, domain=profile.domain)
                if not passed:
                    conn.execute(
                        "UPDATE chunks SET deleted_at = now() WHERE job_id = %s",
                        (job_id,),
                    )
                    _fail_stage(conn, job_id, "validation")
                    raise ValueError("Chunk validation failed: too many low-quality chunks")
            except ValueError:
                raise
            except Exception:
                _fail_stage(conn, job_id, "validation")
                raise

            # --- Embedding ---
            _update_stage(conn, job_id, "embedding")
            try:
                embed_and_store_chunks(conn, chunk_rows)
                conn.commit()
            except Exception:
                _fail_stage(conn, job_id, "embedding")
                raise

            conn.execute(
                "UPDATE jobs SET status = 'completed', current_stage = 'completed', updated_at = now() WHERE id = %s",
                (job_id,),
            )
            conn.commit()

        except Exception:
            conn.rollback()
            delete_stored_file(source_id)
            raise

    return {"source_id": source_id, "job_id": job_id, "status": "completed"}
