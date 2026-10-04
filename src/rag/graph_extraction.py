import html
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import jsonschema

from rag import opencode, prompts
from rag.config import settings
from rag.embedding import get_embeddings

log = logging.getLogger(__name__)

_ENTITY_TYPES = ["ORGANIZATION", "PERSON", "POLICY", "PRODUCT", "REGULATION", "CONCEPT", "LOCATION"]

_ENTITY_SCHEMA = {
    "type": "object",
    "required": ["canonical_name", "entity_type"],
    "properties": {
        "canonical_name": {"type": "string", "minLength": 1},
        "entity_type": {"type": "string", "enum": _ENTITY_TYPES},
        "aliases": {"type": "array", "items": {"type": "string"}},
    },
}

_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_TRAILING_PUNCT_RE = re.compile(r"^[.,;:!?\-\s]+|[.,;:!?\-\s]+$")


def _normalise_name(name: str) -> str:
    name = html.unescape(name)
    name = _WHITESPACE_RE.sub(" ", name).strip()
    name = _LEADING_TRAILING_PUNCT_RE.sub("", name)
    return name


def _validate_entity(entity: dict) -> bool:
    try:
        jsonschema.validate(entity, _ENTITY_SCHEMA)
        return True
    except jsonschema.ValidationError:
        return False


def _normalise_entity_type(value) -> str:
    """Map model output like "organization" / "Product" / "Named-Entity" onto
    the uppercase underscore form the schema enum expects. Off-list values
    stay off-list and are dropped by validation."""
    return re.sub(r"[\s\-]+", "_", str(value).strip()).upper()


def extract_entities(chunk_content: str, session_id: str | None = None) -> list[dict]:
    """Call the LLM (OpenCode Go) to extract entities from one chunk's content.

    `session_id` is the OpenCode conversation ID (`x-opencode-session`); the
    caller mints one per source so every chunk of one ingestion shares it.

    Contract (R7): a missing API key is the documented graceful dev/CI/smoke
    path and returns `[]` rather than raising. Any other failure (HTTP
    error, malformed response, timeout, ...) is NOT swallowed here — it
    raises, so the parallel wrapper (`_extract_chunk_entities_parallel`) can
    distinguish "chunk genuinely produced zero entities" from "chunk-level
    LLM failure" and record the latter in `failed_chunks` instead of
    silently losing it.
    """
    if not settings.OPENCODE_API_KEY:
        return []
    prompt = prompts.ENTITY_EXTRACTION.format(
        types=", ".join(_ENTITY_TYPES),
        text=chunk_content[:4000],
    )
    parsed = opencode.json_call(
        settings.MODEL_ENTITY_EXTRACTION,
        prompt,
        session_id=session_id,
        reasoning_effort=settings.EXTRACTION_REASONING_EFFORT,
        timeout=60,
        scope="entities",
    )
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
    if not isinstance(parsed, list):
        parsed = []
    result = []
    for e in parsed:
        if not isinstance(e, dict):
            continue
        e = {**e, "entity_type": _normalise_entity_type(e.get("entity_type", ""))}
        if not _validate_entity(e):
            continue
        e["canonical_name"] = _normalise_name(e["canonical_name"])
        e["aliases"] = [_normalise_name(a) for a in e.get("aliases", [])]
        result.append(e)
    return result


def store_entities_and_edges(
    conn,
    driver,
    source_id: str,
    chunk_entities: list[tuple[str, list[dict]]],
) -> list[str]:
    """Batched storage pass for a whole source's extracted entities.

    `chunk_entities` is a list of (chunk_id, entities) pairs — one entry per
    successfully-extracted chunk. Entity name embeddings are batched via a
    single `get_embeddings` call across the whole source. Postgres exact-name
    dedup (SELECT-first on `canonical_name`) stays a serial per-name lookup
    (R6 — not vectorized, not batched differently) since it must observe
    inserts made earlier in the same loop (a name repeated across chunks in
    this batch must resolve to one row). Only the Memgraph write side
    batches: one `UNWIND` for `Entity` node MERGEs, one `UNWIND` for
    `MENTIONS` edge MERGEs, covering the whole source instead of one
    `session.run` pair per chunk.

    Returns the list of entity ids assigned, one per (chunk, entity) mention
    in input order (duplicates included, matching the old per-chunk
    function's contract of returning ids "for this chunk's entities").
    """
    flat: list[tuple[str, dict]] = [
        (chunk_id, entity)
        for chunk_id, entities in chunk_entities
        for entity in entities
    ]
    if not flat:
        return []

    names = [entity["canonical_name"] for _chunk_id, entity in flat]
    try:
        vecs = get_embeddings(names)
    except Exception:
        vecs = [None] * len(names)
    if len(vecs) != len(names):
        vecs = [None] * len(names)

    entity_id_by_name: dict[str, str] = {}
    nodes_by_id: dict[str, dict] = {}
    seen_edges: set[tuple[str, str]] = set()
    edges: list[dict] = []
    entity_ids: list[str] = []

    for (chunk_id, entity), vec in zip(flat, vecs):
        normalised = _normalise_name(entity["canonical_name"])

        entity_id = entity_id_by_name.get(normalised)
        if entity_id is None:
            existing = conn.execute(
                "SELECT id::text FROM entities WHERE canonical_name = %s LIMIT 1",
                (normalised,),
            ).fetchone()

            if existing:
                entity_id = existing[0]
            else:
                entity_id = str(uuid.uuid4())
                embedding_str = f"[{','.join(str(v) for v in vec)}]" if vec is not None else None
                conn.execute(
                    """INSERT INTO entities (id, canonical_name, entity_type, aliases, source_id, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s::vector)
                       ON CONFLICT DO NOTHING""",
                    (
                        entity_id,
                        normalised,
                        entity["entity_type"],
                        entity.get("aliases", []),
                        source_id,
                        embedding_str,
                    ),
                )
            entity_id_by_name[normalised] = entity_id

        entity_ids.append(entity_id)
        nodes_by_id.setdefault(
            entity_id, {"canonical_name": normalised, "entity_type": entity["entity_type"]}
        )

        edge_key = (chunk_id, entity_id)
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            edges.append({"chunk_id": chunk_id, "entity_id": entity_id, "confidence": 1.0})

    if driver is not None:
        with driver.session() as session:
            session.run(
                """
                UNWIND $nodes AS node
                MERGE (e:Entity {entity_id: node.entity_id})
                SET e.canonical_name = node.canonical_name, e.entity_type = node.entity_type
                """,
                nodes=[
                    {"entity_id": entity_id, **record}
                    for entity_id, record in nodes_by_id.items()
                ],
            )
            session.run(
                """
                UNWIND $edges AS edge
                MATCH (c:Chunk {chunk_id: edge.chunk_id}), (e:Entity {entity_id: edge.entity_id})
                MERGE (c)-[:MENTIONS {confidence: edge.confidence}]->(e)
                """,
                edges=edges,
            )

    return entity_ids


def _extract_chunk_entities_parallel(
    chunk_rows: list[tuple[str, str]],
    session_id: str | None = None,
) -> tuple[list[tuple[str, list[dict]]], list[str]]:
    """Phase A: fan out `extract_entities` across a thread pool.

    `session_id` is the one OpenCode session shared by every chunk of the source.

    Returns `(succeeded, failed_chunk_ids)`. A chunk-level LLM failure (any
    exception from `extract_entities`, e.g. HTTP/timeout/parse errors — NOT
    the "no API key" graceful path, which returns `[]` and never raises) is
    caught per-chunk, recorded into `failed_chunk_ids`, and does not abort
    the rest of the batch (R7/KTD6) — distinct from a chunk that genuinely
    has zero entities. Ordered result collection (a pre-sized list indexed
    by `future_indexes`) preserves chunk order regardless of completion
    order.
    """
    if not chunk_rows:
        return [], []

    def _extract(row: tuple[str, str]) -> tuple[str, list[dict]]:
        chunk_id, content = row
        return chunk_id, extract_entities(content, session_id=session_id)

    max_workers = min(settings.GRAPH_EXTRACTION_CONCURRENCY, len(chunk_rows))

    results: list[tuple[str, list[dict]] | None] = [None] * len(chunk_rows)
    failed_chunk_ids: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_indexes = {
            executor.submit(_extract, row): index
            for index, row in enumerate(chunk_rows)
        }
        for future in as_completed(future_indexes):
            index = future_indexes[future]
            chunk_id = chunk_rows[index][0]
            try:
                result = future.result()
            except Exception as exc:
                failed_chunk_ids.append(chunk_id)
                log.warning(
                    "graph_extraction_chunk_failed",
                    extra={"chunk_id": chunk_id, "error": str(exc)},
                )
                continue
            results[index] = result

    return [result for result in results if result is not None], failed_chunk_ids


def extract_and_store_graph(
    conn,
    driver,
    source_id: str,
    job_id: str,
    chunk_rows: list[tuple[str, str]],
) -> dict:
    """Phase A-B entity extraction/storage orchestration for one source.

    A: parallel `extract_entities` fan-out with per-chunk failure recording
       (KTD6/R7), all sharing one OpenCode session id minted per source.
    B: one serial batch storage pass (`store_entities_and_edges`) — Postgres
       exact-name dedup stays a serial per-name lookup (R6), Memgraph writes
       batch into two `UNWIND` calls for the whole source.
    """
    total_chunks = len(chunk_rows)

    # KTD6b: missing API key short-circuits before the fan-out entirely —
    # this is the existing graceful dev/CI/smoke-test path, not a failure,
    # so it must never feed the failure-rate gate below.
    if not settings.OPENCODE_API_KEY:
        return {
            "chunks_processed": 0,
            "entities_stored": 0,
            "failed_chunks": [],
            "skipped": True,
            "skip_reason": "OPENCODE_API_KEY not configured",
        }

    extracted_rows, failed_chunks = _extract_chunk_entities_parallel(
        chunk_rows, session_id=opencode.new_session_id("entities")
    )

    if total_chunks and (len(failed_chunks) / total_chunks) > settings.STAGE_FAILURE_RATE_THRESHOLD:
        raise RuntimeError(
            f"graph_extraction failure rate {len(failed_chunks)}/{total_chunks} "
            f"exceeds STAGE_FAILURE_RATE_THRESHOLD="
            f"{settings.STAGE_FAILURE_RATE_THRESHOLD}"
        )

    entity_ids = store_entities_and_edges(conn, driver, source_id, extracted_rows)
    conn.commit()

    return {
        "chunks_processed": len(extracted_rows),
        "entities_stored": len(entity_ids),
        "failed_chunks": failed_chunks,
    }
