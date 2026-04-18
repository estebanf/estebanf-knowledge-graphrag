import hashlib
import uuid
from pathlib import Path

import psycopg
import psycopg.types.json

from rag.chunk_validation import validate_chunks
from rag.chunking import ChunkData, chunk_document
from rag.db import get_connection
from rag.embedding import embed_and_store_chunks
from rag.graph_db import get_graph_driver
from rag.graph_extraction import extract_and_store_graph
from rag.graph_linking import link_graph
from rag.metadata_extraction import extract_metadata
from rag.parser import ParseError, parse_to_markdown
from rag.profiling import profile_document
from rag.storage import delete_stored_file, store_file

STAGE_ORDER = [
    "parsing", "profiling", "chunking", "validation",
    "embedding", "graph_extraction", "graph_linking",
]


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
        """UPDATE jobs SET status = %s, current_stage = %s, updated_at = now(),
           stage_log = COALESCE(stage_log, '{}'::jsonb) || jsonb_build_object(%s::text, to_char(now(), 'YYYY-MM-DD"T"HH24:MI:SS'))
           WHERE id = %s""",
        (f"processing:{stage}", stage, stage, job_id),
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
    if not chunks:
        return []
    index_to_uuid: dict[int, str] = {}
    chunk_ids: list[str] = []
    for chunk in chunks:
        chunk_id = str(uuid.uuid4())
        chunk_ids.append(chunk_id)
        index_to_uuid[chunk.chunk_index] = chunk_id
        conn.execute(
            """INSERT INTO chunks
              (id, source_id, job_id, content, token_count, chunk_index,
               parent_chunk_id, chunking_strategy, chunking_config, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                chunk_id, source_id, job_id,
                chunk.content, chunk.token_count, chunk.chunk_index,
                None,
                chunk.chunking_strategy,
                psycopg.types.json.Jsonb(chunk.chunking_config),
                psycopg.types.json.Jsonb(chunk.metadata),
            ),
        )
    for chunk, chunk_id in zip(chunks, chunk_ids):
        if chunk.parent_chunk_id is not None and chunk.parent_chunk_id.isdigit():
            parent_uuid = index_to_uuid.get(int(chunk.parent_chunk_id))
            if parent_uuid:
                conn.execute(
                    "UPDATE chunks SET parent_chunk_id = %s WHERE id = %s",
                    (parent_uuid, chunk_id),
                )
    return [(chunk_id, chunk.content) for chunk_id, chunk in zip(chunk_ids, chunks)]


def _cleanup_graph_artifacts(conn: psycopg.Connection, driver, source_id: str) -> None:
    entity_ids = [
        str(r[0]) for r in conn.execute(
            "SELECT id FROM entities WHERE source_id = %s", (source_id,)
        ).fetchall()
    ]
    conn.execute("DELETE FROM entities WHERE source_id = %s", (source_id,))
    if entity_ids and driver:
        with driver.session() as session:
            session.run(
                "MATCH (e:Entity) WHERE e.entity_id IN $ids DETACH DELETE e",
                ids=entity_ids,
            )


def cleanup_from_stage(
    conn: psycopg.Connection,
    driver,
    job_id: str,
    source_id: str,
    from_stage: str,
) -> None:
    idx = STAGE_ORDER.index(from_stage)
    chunking_idx = STAGE_ORDER.index("chunking")
    embedding_idx = STAGE_ORDER.index("embedding")
    graph_extraction_idx = STAGE_ORDER.index("graph_extraction")
    graph_linking_idx = STAGE_ORDER.index("graph_linking")

    if idx <= chunking_idx:
        conn.execute("DELETE FROM chunks WHERE job_id = %s", (job_id,))
        _cleanup_graph_artifacts(conn, driver, source_id)
    elif idx <= embedding_idx:
        conn.execute("UPDATE chunks SET embedding = NULL WHERE job_id = %s", (job_id,))
        _cleanup_graph_artifacts(conn, driver, source_id)
    elif idx <= graph_extraction_idx:
        _cleanup_graph_artifacts(conn, driver, source_id)
    elif idx <= graph_linking_idx:
        entity_ids = [
            str(r[0]) for r in conn.execute(
                "SELECT id FROM entities WHERE source_id = %s", (source_id,)
            ).fetchall()
        ]
        if entity_ids and driver:
            with driver.session() as session:
                session.run(
                    "MATCH (e:Entity)-[r:MENTIONED_IN]->() WHERE e.entity_id IN $ids DELETE r",
                    ids=entity_ids,
                )
    conn.commit()


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
                """INSERT INTO sources (id, name, file_name, file_type, storage_path, md5, version, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, 1, %s)""",
                (source_id, source_name, file_path.name, file_type,
                 str(stored_path), md5, psycopg.types.json.Jsonb(metadata or {})),
            )
            conn.execute(
                "INSERT INTO jobs (id, source_id, status, current_stage, stage_log) VALUES (%s, %s, 'processing:parsing', 'parsing', '{}'::jsonb)",
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
                    conn.execute("UPDATE chunks SET deleted_at = now() WHERE job_id = %s", (job_id,))
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

            # --- Graph Extraction ---
            _update_stage(conn, job_id, "graph_extraction")
            with get_graph_driver() as driver:
                with driver.session() as session:
                    session.run(
                        "MERGE (s:Source {source_id: $source_id})",
                        source_id=source_id,
                    )
                    for chunk_id, _ in chunk_rows:
                        session.run(
                            "MERGE (c:Chunk {chunk_id: $chunk_id}) SET c.source_id = $source_id",
                            chunk_id=chunk_id, source_id=source_id,
                        )
                        session.run(
                            "MATCH (s:Source {source_id: $source_id}), (c:Chunk {chunk_id: $chunk_id}) "
                            "MERGE (s)-[:INCLUDES]->(c)",
                            source_id=source_id, chunk_id=chunk_id,
                        )
                try:
                    extract_and_store_graph(conn, driver, source_id, job_id, chunk_rows)
                except Exception:
                    _fail_stage(conn, job_id, "graph_extraction")
                    raise

                # --- Graph Linking ---
                _update_stage(conn, job_id, "graph_linking")
                try:
                    link_graph(conn, driver, source_id, job_id)
                except Exception:
                    _fail_stage(conn, job_id, "graph_linking")
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


def retry_job(job_id: str, from_stage: str | None = None) -> dict:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, source_id, status, current_stage FROM jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"Job not found: {job_id}")
        _, source_id, status, current_stage = row
        source_id = str(source_id)
        if not status.startswith("failed:"):
            raise ValueError(f"Job {job_id} is not in a failed state (status: {status})")

        failed_stage = status.split(":", 1)[1]
        start_stage = from_stage or failed_stage

        if start_stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {start_stage}")

        source_row = conn.execute(
            "SELECT storage_path, metadata, markdown_content FROM sources WHERE id = %s",
            (source_id,),
        ).fetchone()
        if not source_row:
            raise ValueError(f"Source not found for job {job_id}")

        stored_path = Path(source_row[0])
        profile = None
        chunks = []
        chunk_rows = []

        with get_graph_driver() as driver:
            cleanup_from_stage(conn, driver, job_id, source_id, start_stage)

            conn.execute(
                "UPDATE jobs SET retry_of = %s, retry_from_stage = %s, updated_at = now() WHERE id = %s",
                (job_id, start_stage, job_id),
            )

            start_idx = STAGE_ORDER.index(start_stage)

            if start_idx <= STAGE_ORDER.index("parsing"):
                _update_stage(conn, job_id, "parsing")
                try:
                    markdown = parse_to_markdown(stored_path)
                    extracted = extract_metadata(markdown)
                    combined = {**extracted, **(source_row[1] or {})}
                    conn.execute(
                        "UPDATE sources SET markdown_content = %s, metadata = %s WHERE id = %s",
                        (markdown, psycopg.types.json.Jsonb(combined), source_id),
                    )
                    conn.commit()
                except Exception:
                    _fail_stage(conn, job_id, "parsing")
                    raise
            else:
                markdown = source_row[2]

            if start_idx <= STAGE_ORDER.index("profiling"):
                _update_stage(conn, job_id, "profiling")
                profile = profile_document(markdown)
            else:
                from rag.profiling import _DEFAULT_PROFILE
                profile = _DEFAULT_PROFILE

            if start_idx <= STAGE_ORDER.index("chunking"):
                _update_stage(conn, job_id, "chunking")
                try:
                    chunks = chunk_document(markdown, profile)
                    chunk_rows = _insert_chunks(conn, source_id, job_id, chunks)
                    conn.commit()
                except Exception:
                    _fail_stage(conn, job_id, "chunking")
                    raise
            else:
                rows = conn.execute(
                    "SELECT id, content FROM chunks WHERE job_id = %s AND deleted_at IS NULL ORDER BY chunk_index",
                    (job_id,),
                ).fetchall()
                chunk_rows = [(str(r[0]), r[1]) for r in rows]

            if start_idx <= STAGE_ORDER.index("validation"):
                _update_stage(conn, job_id, "validation")
                try:
                    passed = validate_chunks(chunks, domain=profile.domain)
                    if not passed:
                        conn.execute("UPDATE chunks SET deleted_at = now() WHERE job_id = %s", (job_id,))
                        _fail_stage(conn, job_id, "validation")
                        raise ValueError("Chunk validation failed")
                except ValueError:
                    raise
                except Exception:
                    _fail_stage(conn, job_id, "validation")
                    raise

            if start_idx <= STAGE_ORDER.index("embedding"):
                _update_stage(conn, job_id, "embedding")
                try:
                    embed_and_store_chunks(conn, chunk_rows)
                    conn.commit()
                except Exception:
                    _fail_stage(conn, job_id, "embedding")
                    raise

            if start_idx <= STAGE_ORDER.index("graph_extraction"):
                _update_stage(conn, job_id, "graph_extraction")
                with driver.session() as session:
                    session.run("MERGE (s:Source {source_id: $sid})", sid=source_id)
                    for chunk_id, _ in chunk_rows:
                        session.run(
                            "MERGE (c:Chunk {chunk_id: $cid}) SET c.source_id = $sid",
                            cid=chunk_id, sid=source_id,
                        )
                        session.run(
                            "MATCH (s:Source {source_id: $sid}), (c:Chunk {chunk_id: $cid}) MERGE (s)-[:INCLUDES]->(c)",
                            sid=source_id, cid=chunk_id,
                        )
                try:
                    extract_and_store_graph(conn, driver, source_id, job_id, chunk_rows)
                except Exception:
                    _fail_stage(conn, job_id, "graph_extraction")
                    raise

            if start_idx <= STAGE_ORDER.index("graph_linking"):
                _update_stage(conn, job_id, "graph_linking")
                try:
                    link_graph(conn, driver, source_id, job_id)
                except Exception:
                    _fail_stage(conn, job_id, "graph_linking")
                    raise

            conn.execute(
                "UPDATE jobs SET status = 'completed', current_stage = 'completed', updated_at = now() WHERE id = %s",
                (job_id,),
            )
            conn.commit()

    return {"job_id": job_id, "status": "completed"}
