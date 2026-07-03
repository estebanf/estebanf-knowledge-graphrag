import json
import math
import re
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from rag import prompts
from rag.config import settings
from rag.db import set_hnsw_ef_search
from rag.embedding import get_embeddings

log = logging.getLogger(__name__)
ProgressCallback = Callable[[str, dict], None]

_OPENCODE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
_MODEL = "deepseek-v4-flash"
_MAX_CHUNK_CHARS = 4000


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def _embedding_literal(embedding) -> str:
    if isinstance(embedding, str):
        return embedding
    return "[" + ",".join(str(v) for v in embedding) + "]"


def _normalized_insights(raw_insights: list[dict]) -> list[dict]:
    normalized = []
    for raw in raw_insights:
        if not isinstance(raw, dict):
            continue
        insight = str(raw.get("insight") or "").strip()
        if not insight:
            continue
        raw_topics = raw.get("topics", [])
        topics = [
            str(topic).strip()
            for topic in raw_topics
            if str(topic).strip()
        ] if isinstance(raw_topics, list) else []
        normalized.append({"insight": insight, "topics": topics})
    return normalized


def extract_insights_from_chunk(content: str) -> list[dict]:
    """Call the LLM to extract insights from one chunk's content.

    Contract (R7): a missing API key is the documented graceful dev/CI/smoke
    path and returns `[]` rather than raising. Any other failure (HTTP error,
    malformed response, timeout, ...) is NOT swallowed here — it raises, so
    the parallel wrapper (`_extract_chunk_insights_parallel`) can distinguish
    "chunk genuinely produced zero insights" from "chunk-level LLM failure"
    and record the latter in `failed_chunks` instead of silently losing it.
    """
    if not settings.OPENCODE_API_KEY:
        return []
    prompt = prompts.INSIGHT_EXTRACTION.format(chunk=content[:_MAX_CHUNK_CHARS])
    resp = httpx.post(
        _OPENCODE_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.OPENCODE_API_KEY}",
        },
        json={"model": _MODEL, "messages": [{"role": "user", "content": prompt}]},
        timeout=(10, 120),
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return json.loads(_strip_fences(raw)).get("insights", [])


def upsert_insight(conn, content: str, embedding: list[float]) -> tuple[str, bool]:
    """Dedup an insight against the corpus and insert it if it's new.

    Dedup lookup mirrors `retrieval.dense_retrieve`'s prefilter + rerank
    pattern: a binary-quantized HNSW prefilter (`insights_embedding_binary_hnsw_idx`)
    narrows the corpus to `INSIGHT_PREFILTER_CANDIDATES` candidates, then
    full-precision `embedding <=>` cosine distance reranks that pool. This
    keeps the lookup index-assisted regardless of corpus size — the raw
    4096-dim vector column has no usable index for `ORDER BY embedding <=>`
    (pgvector caps native HNSW/IVFFlat at 2000 dims), so unprefiltered
    ordering forces a full sequential scan.
    """
    emb_str = _embedding_literal(embedding)
    prefilter_count = settings.INSIGHT_PREFILTER_CANDIDATES
    set_hnsw_ef_search(conn, prefilter_count)
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH insight_prefilter AS MATERIALIZED (
                SELECT id
                FROM insights
                WHERE embedding IS NOT NULL
                ORDER BY binary_quantize(embedding)::bit(4096) <~> binary_quantize(%s::vector)::bit(4096)
                LIMIT %s
            )
            SELECT i.id, 1 - (i.embedding <=> %s::vector) AS sim
            FROM insight_prefilter p
            JOIN insights i ON i.id = p.id
            ORDER BY i.embedding <=> %s::vector
            LIMIT 1
            """,
            (emb_str, prefilter_count, emb_str, emb_str),
        )
        row = cur.fetchone()
        if row and row[1] >= settings.INSIGHT_DEDUP_COSINE_THRESHOLD:
            return str(row[0]), False
        cur.execute(
            "INSERT INTO insights (content, embedding) VALUES (%s, %s::vector) RETURNING id",
            (content, emb_str),
        )
        new_id = cur.fetchone()[0]
    return str(new_id), True


def link_chunk_insight(conn, chunk_id: str, insight_id: str, topics: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunk_insights (chunk_id, insight_id, topics)
            VALUES (%s, %s, %s)
            ON CONFLICT (chunk_id, insight_id) DO UPDATE SET topics = EXCLUDED.topics
            """,
            (chunk_id, insight_id, topics),
        )


def link_chunk_insights_batch(
    conn, pairs: list[tuple[str, str, list[str]]]
) -> None:
    """Bulk variant of `link_chunk_insight` for a whole source batch (Phase C).

    Uses `executemany` (a sequence of individual upsert statements) rather
    than a single multi-row `VALUES (...) ON CONFLICT` — that shape would
    error ("ON CONFLICT DO UPDATE command cannot affect row a second time")
    if the same (chunk_id, insight_id) pair appears twice in one batch, which
    can happen when two raw insights from the same chunk collapse onto the
    same within-batch survivor (AE2-adjacent case).
    """
    if not pairs:
        return
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunk_insights (chunk_id, insight_id, topics)
            VALUES (%s, %s, %s)
            ON CONFLICT (chunk_id, insight_id) DO UPDATE SET topics = EXCLUDED.topics
            """,
            pairs,
        )


def store_insights_in_graph_batch(
    driver, chunk_insight_pairs: list[tuple[str, str, str, list[str]]]
) -> None:
    """Batched replacement for the old per-insight `store_insight_in_graph`
    (Phase E). `chunk_insight_pairs` is a list of
    (chunk_id, insight_id, content, topics) tuples for the whole source.

    Writes two UNWIND batches instead of one MERGE pair per chunk-insight
    pair: one for `Insight` node MERGEs (deduped by insight id — a reused
    insight linked from multiple chunks in this batch is merged once), one
    for `CONTAINS` edge MERGEs (deduped by (chunk_id, insight_id) pair).
    """
    if not driver or not chunk_insight_pairs:
        return

    nodes_by_id: dict[str, str] = {}
    for _chunk_id, insight_id, content, _topics in chunk_insight_pairs:
        nodes_by_id.setdefault(insight_id, content)

    seen_edges: set[tuple[str, str]] = set()
    edges = []
    for chunk_id, insight_id, _content, topics in chunk_insight_pairs:
        key = (chunk_id, insight_id)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({"chunk_id": chunk_id, "insight_id": insight_id, "topics": topics})

    with driver.session() as session:
        session.run(
            """
            UNWIND $nodes AS node
            MERGE (i:Insight {insight_id: node.insight_id})
            SET i.content = node.content
            """,
            nodes=[
                {"insight_id": insight_id, "content": content}
                for insight_id, content in nodes_by_id.items()
            ],
        )
        session.run(
            """
            UNWIND $edges AS edge
            MATCH (c:Chunk {chunk_id: edge.chunk_id}), (i:Insight {insight_id: edge.insight_id})
            MERGE (c)-[:CONTAINS {topics: edge.topics}]->(i)
            """,
            edges=edges,
        )


_LINK_OVERFETCH_MULTIPLIER = 4  # KTD2: over-fetch before the same-source exclusion


def _forward_top_k_neighbors(
    cur, insight_id: str, emb_str: str, source_id: str, candidate_count: int, k: int
) -> list[tuple[str, float, object]]:
    """Top-K prefiltered-and-reranked neighbors for a newly-inserted insight,
    excluding candidates that share `source_id` (the batch's own source) —
    mirrors the legacy forward-pass exclusion at insight_extraction.py:140-146.
    """
    cur.execute(
        """
        WITH insight_prefilter AS MATERIALIZED (
            SELECT id
            FROM insights
            WHERE embedding IS NOT NULL
              AND id != %s
            ORDER BY binary_quantize(embedding)::bit(4096) <~> binary_quantize(%s::vector)::bit(4096)
            LIMIT %s
        )
        SELECT i.id, 1 - (i.embedding <=> %s::vector) AS sim, i.embedding
        FROM insight_prefilter p
        JOIN insights i ON i.id = p.id
        WHERE NOT EXISTS (
            SELECT 1
            FROM chunk_insights ci
            JOIN chunks c ON c.id = ci.chunk_id
            WHERE ci.insight_id = i.id
              AND c.source_id = %s
        )
        ORDER BY i.embedding <=> %s::vector
        LIMIT %s
        """,
        (insight_id, emb_str, candidate_count, emb_str, source_id, emb_str, k),
    )
    return cur.fetchall()


def _reverse_top_k_neighbor_ids(
    cur, candidate_id: str, emb_str: str, candidate_count: int, k: int
) -> set[str]:
    """Top-K prefiltered-and-reranked neighbor ids for a forward-pass
    candidate, excluding any insight that shares a source with `candidate_id`
    via the dynamic self-join — mirrors the legacy reverse-pass exclusion at
    insight_extraction.py:164-172 (the candidate may belong to a source other
    than the batch's own, so this cannot reuse the fixed-source_id filter).
    """
    cur.execute(
        """
        WITH insight_prefilter AS MATERIALIZED (
            SELECT id
            FROM insights
            WHERE embedding IS NOT NULL
              AND id != %s
            ORDER BY binary_quantize(embedding)::bit(4096) <~> binary_quantize(%s::vector)::bit(4096)
            LIMIT %s
        )
        SELECT i.id
        FROM insight_prefilter p
        JOIN insights i ON i.id = p.id
        WHERE NOT EXISTS (
            SELECT 1
            FROM chunk_insights candidate_ci
            JOIN chunks candidate_chunk ON candidate_chunk.id = candidate_ci.chunk_id
            JOIN chunk_insights other_ci ON other_ci.insight_id = i.id
            JOIN chunks other_chunk ON other_chunk.id = other_ci.chunk_id
            WHERE candidate_ci.insight_id = %s
              AND candidate_chunk.source_id = other_chunk.source_id
        )
        ORDER BY i.embedding <=> %s::vector
        LIMIT %s
        """,
        (candidate_id, emb_str, candidate_count, candidate_id, emb_str, k),
    )
    return {str(r[0]) for r in cur.fetchall()}


def link_related_insights(
    conn, driver, source_id: str, new_insights: list[tuple[str, list[float]]]
) -> int:
    """Compute mutual top-K `RELATED_TO` edges for a batch of newly-inserted
    insights from one source, writing all qualifying edges in a single
    Memgraph `UNWIND` call.

    Reuses U1's binary-prefilter + full-precision-rerank pattern
    (`upsert_insight`): `set_hnsw_ef_search` primes the HNSW candidate pool,
    a `MATERIALIZED` CTE narrows the corpus via
    `insights_embedding_binary_hnsw_idx`, and the join back to `insights`
    reranks by full-precision `embedding <=>` cosine distance. Candidates are
    over-fetched (`_LINK_OVERFETCH_MULTIPLIER`x `INSIGHT_LINK_TOP_K`) before
    the same-source exclusion is applied as a post-rerank filter, so
    excluding same-source candidates doesn't starve the final top-K.

    A pair (A, B) is linked only if each is in the other's top-K
    (mutual-top-K, matching the legacy per-insight loop's semantics) — this
    is computed by intersecting a forward pass (new insights -> candidates)
    with a reverse pass (candidates -> their own top-K) in Python, rather
    than nested per-neighbor queries.

    Returns the number of `RELATED_TO` edges written (each edge is written
    as a bidirectional pair via a single MERGE per direction in the batch
    UNWIND, counted once here).
    """
    if not new_insights:
        return 0

    k = settings.INSIGHT_LINK_TOP_K
    candidate_count = k * _LINK_OVERFETCH_MULTIPLIER
    set_hnsw_ef_search(conn, candidate_count)

    embeddings_by_id: dict[str, object] = {
        insight_id: embedding for insight_id, embedding in new_insights
    }
    forward_neighbors: dict[str, list[tuple[str, float]]] = {}

    with conn.cursor() as cur:
        for insight_id, embedding in new_insights:
            emb_str = _embedding_literal(embedding)
            rows = _forward_top_k_neighbors(
                cur, insight_id, emb_str, source_id, candidate_count, k
            )
            neighbors = []
            for cand_id, sim, cand_emb_raw in rows:
                cand_id = str(cand_id)
                neighbors.append((cand_id, float(sim)))
                if cand_emb_raw is not None and cand_id not in embeddings_by_id:
                    embeddings_by_id[cand_id] = cand_emb_raw
            forward_neighbors[insight_id] = neighbors

        # Ordered dedup (not a set) so candidate reverse-pass queries run in
        # a deterministic, discovery order — same-source exclusion and query
        # results don't depend on it, but it keeps behavior reproducible.
        candidate_ids = list(dict.fromkeys(
            cand_id
            for neighbors in forward_neighbors.values()
            for cand_id, _sim in neighbors
        ))

        reverse_neighbor_ids: dict[str, set[str]] = {}
        for cand_id in candidate_ids:
            cand_emb_raw = embeddings_by_id.get(cand_id)
            if cand_emb_raw is None:
                continue
            reverse_neighbor_ids[cand_id] = _reverse_top_k_neighbor_ids(
                cur, cand_id, _embedding_literal(cand_emb_raw), candidate_count, k
            )

    edges = []
    seen_pairs: set[frozenset] = set()
    for a_id, neighbors in forward_neighbors.items():
        for b_id, sim in neighbors:
            if a_id == b_id:
                continue
            if a_id not in reverse_neighbor_ids.get(b_id, set()):
                continue
            pair_key = frozenset((a_id, b_id))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            edges.append({"a_id": a_id, "b_id": b_id, "sim": sim})

    if not edges:
        return 0

    with driver.session() as session:
        session.run(
            """
            UNWIND $edges AS edge
            MATCH (a:Insight {insight_id: edge.a_id}), (b:Insight {insight_id: edge.b_id})
            MERGE (a)-[:RELATED_TO {similarity: edge.sim}]->(b)
            MERGE (b)-[:RELATED_TO {similarity: edge.sim}]->(a)
            """,
            edges=edges,
        )

    return len(edges)


def _extract_chunk_insights_parallel(
    chunk_rows: list[tuple[str, str]],
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[tuple[str, str, list[dict]]], list[str]]:
    """Phase A: fan out `extract_insights_from_chunk` across a thread pool.

    Returns `(succeeded, failed_chunk_ids)`. A chunk-level LLM failure (any
    exception from `extract_insights_from_chunk`, e.g. HTTP/timeout/parse
    errors — NOT the "no API key" graceful path, which returns `[]` and
    never raises) is caught per-chunk, recorded into `failed_chunk_ids`, and
    does not abort the rest of the batch (R7/KTD6) — distinct from a chunk
    that genuinely extracted zero insights.
    """
    if not chunk_rows:
        return [], []

    def _extract(row: tuple[str, str]) -> tuple[str, str, list[dict]]:
        chunk_id, content = row
        raw_insights = extract_insights_from_chunk(content)
        return chunk_id, content, _normalized_insights(raw_insights)

    max_workers = min(settings.INSIGHT_EXTRACTION_CONCURRENCY, len(chunk_rows))
    if progress_callback:
        progress_callback(
            "extract_start",
            {"total": len(chunk_rows), "concurrency": max_workers},
        )

    results: list[tuple[str, str, list[dict]] | None] = [None] * len(chunk_rows)
    failed_chunk_ids: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_indexes = {
            executor.submit(_extract, row): index
            for index, row in enumerate(chunk_rows)
        }
        for completed_count, future in enumerate(as_completed(future_indexes), start=1):
            index = future_indexes[future]
            chunk_id = chunk_rows[index][0]
            try:
                result = future.result()
            except Exception as exc:
                failed_chunk_ids.append(chunk_id)
                log.warning(
                    "insight_extraction_chunk_failed",
                    extra={"chunk_id": chunk_id, "error": str(exc)},
                )
                if progress_callback:
                    progress_callback(
                        "extract_error",
                        {
                            "position": completed_count,
                            "total": len(chunk_rows),
                            "chunk_id": chunk_id,
                            "error": str(exc),
                        },
                    )
                continue

            results[index] = result
            if progress_callback:
                progress_callback(
                    "extract_chunk",
                    {
                        "position": completed_count,
                        "total": len(chunk_rows),
                        "chunk_id": result[0],
                        "insights": len(result[2]),
                    },
                )

    if progress_callback:
        progress_callback("extract_done", {"total": len(chunk_rows)})
    return [result for result in results if result is not None], failed_chunk_ids


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def extract_and_store_insights(
    conn,
    driver,
    source_id: str,
    chunk_rows: list[tuple[str, str]],
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Phased A-E insight storage orchestration for one source's chunks.

    A: parallel extraction with per-chunk failure recording (KTD6/R7).
    B: one batched `get_embeddings` call over every extracted insight text.
    C: within-batch cosine dedup (first occurrence survives), then corpus
       dedup via `upsert_insight`; `chunk_insights` rows written in bulk.
    D: one `link_related_insights` call for the whole batch's new insights.
    E: one UNWIND batch for Insight nodes, one for CONTAINS edges.
    """
    total_chunks = len(chunk_rows)

    # KTD6b: missing API key short-circuits before the fan-out entirely —
    # this is the existing graceful dev/CI/smoke-test path, not a failure,
    # so it must never feed the failure-rate gate below.
    if not settings.OPENCODE_API_KEY:
        if progress_callback:
            progress_callback("store_start", {"total": 0})
            progress_callback("store_done", {"total": 0})
        return {
            "chunks_processed": 0,
            "insights_extracted": 0,
            "insights_reused": 0,
            "failed_chunks": [],
            "related_edges": 0,
            "skipped": True,
            "skip_reason": "OPENCODE_API_KEY not configured",
        }

    extracted_rows, failed_chunks = _extract_chunk_insights_parallel(
        chunk_rows, progress_callback
    )

    if total_chunks and (len(failed_chunks) / total_chunks) > settings.STAGE_FAILURE_RATE_THRESHOLD:
        raise RuntimeError(
            f"insight_extraction failure rate {len(failed_chunks)}/{total_chunks} "
            f"exceeds STAGE_FAILURE_RATE_THRESHOLD="
            f"{settings.STAGE_FAILURE_RATE_THRESHOLD}"
        )

    if progress_callback:
        progress_callback("store_start", {"total": len(extracted_rows)})

    chunks_processed = len(extracted_rows)

    # Flatten to one (chunk_id, raw_insight) item per extracted insight so
    # Phase B can embed everything in a single batched call.
    items: list[tuple[str, dict]] = [
        (chunk_id, raw)
        for chunk_id, _content, raw_insights in extracted_rows
        for raw in raw_insights
    ]

    if not items:
        if progress_callback:
            progress_callback("store_done", {"total": len(extracted_rows)})
        return {
            "chunks_processed": chunks_processed,
            "insights_extracted": 0,
            "insights_reused": 0,
            "failed_chunks": failed_chunks,
            "related_edges": 0,
        }

    texts = [raw["insight"] for _chunk_id, raw in items]
    embeddings = get_embeddings(texts)

    # Phase C, part 1: within-batch pairwise dedup. First occurrence of a
    # near-duplicate (cosine >= INSIGHT_DEDUP_COSINE_THRESHOLD) survives;
    # later ones collapse onto it (AE2).
    survivor_indexes: list[int] = []
    survivor_of: dict[int, int] = {}
    threshold = settings.INSIGHT_DEDUP_COSINE_THRESHOLD
    for idx, emb in enumerate(embeddings):
        matched_survivor = None
        for s_idx in survivor_indexes:
            if _cosine_similarity(emb, embeddings[s_idx]) >= threshold:
                matched_survivor = s_idx
                break
        if matched_survivor is None:
            survivor_indexes.append(idx)
            survivor_of[idx] = idx
        else:
            survivor_of[idx] = matched_survivor

    # Phase C, part 2: corpus dedup for each within-batch survivor.
    insight_id_of_survivor: dict[int, str] = {}
    new_insights_for_linking: list[tuple[str, list[float]]] = []
    insights_extracted = 0
    insights_reused = 0
    for s_idx in survivor_indexes:
        _chunk_id, raw = items[s_idx]
        insight_id, is_new = upsert_insight(conn, raw["insight"], embeddings[s_idx])
        insight_id_of_survivor[s_idx] = insight_id
        if is_new:
            insights_extracted += 1
            new_insights_for_linking.append((insight_id, embeddings[s_idx]))
        else:
            insights_reused += 1

    # Phase C, part 3: bulk chunk_insights links — every (chunk_id, insight)
    # pair, including within-batch duplicates resolved onto their survivor's
    # id, so both chunks end up linked to the one surviving row (AE2).
    chunk_insight_pairs: list[tuple[str, str, list[str]]] = []
    graph_pairs: list[tuple[str, str, str, list[str]]] = []
    for idx, (chunk_id, raw) in enumerate(items):
        insight_id = insight_id_of_survivor[survivor_of[idx]]
        topics = raw.get("topics", [])
        chunk_insight_pairs.append((chunk_id, insight_id, topics))
        graph_pairs.append((chunk_id, insight_id, raw["insight"], topics))

    link_chunk_insights_batch(conn, chunk_insight_pairs)
    conn.commit()

    # Phase D: one linking pass for the whole batch's new insights.
    related_edges = link_related_insights(conn, driver, source_id, new_insights_for_linking)
    conn.commit()

    # Phase E: batched Memgraph writes for the whole batch.
    store_insights_in_graph_batch(driver, graph_pairs)

    if progress_callback:
        progress_callback("store_done", {"total": len(extracted_rows)})

    return {
        "chunks_processed": chunks_processed,
        "insights_extracted": insights_extracted,
        "insights_reused": insights_reused,
        "failed_chunks": failed_chunks,
        "related_edges": related_edges,
    }

