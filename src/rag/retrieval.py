import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
import tiktoken

from rag import prompts
from rag.config import settings
from rag.db import get_connection
from rag.embedding import get_embeddings
from rag.graph_db import get_graph_driver


@dataclass
class RetrievalCandidate:
    chunk_id: str
    chunk: str
    source_id: str
    source_path: str
    source_metadata: dict
    score: float


@dataclass
class InsightCandidate:
    insight_id: str
    insight: str
    score: float


@dataclass
class InsightSourceRef:
    source_id: str
    source_path: str
    source_metadata: dict


@dataclass
class InsightSearchResult:
    score: float
    insight: str
    insight_id: str
    topics: list[str]
    sources: list[InsightSourceRef]


@dataclass
class HybridSearchResults:
    chunks: list[RetrievalCandidate]
    insights: list[InsightSearchResult]


@dataclass
class EntityCandidate:
    entity_id: str
    name: str
    entity_type: str


@dataclass
class SecondHopEntityCandidate:
    chunk_id: str
    name: str
    entity_type: str
    chunk: str


@dataclass
class TraceLogger:
    enabled: bool
    printer: Optional[Callable[[str], None]] = None

    def emit(self, message: str) -> None:
        if self.enabled and self.printer:
            self.printer(message)


@dataclass
class RetrievalParams:
    seed_count: int
    result_count: int
    rrf_k: int
    entity_confidence_threshold: float
    first_hop_similarity_threshold: float
    second_hop_similarity_threshold: float


_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
_ENC = tiktoken.get_encoding("cl100k_base")


def _require_api_key() -> None:
    if not settings.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is required for retrieval")


def _openrouter_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }


def _strip_code_fences(content: str) -> str:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def _token_count(text: str) -> int:
    return len(_ENC.encode(text))


def normalize_query_variants(raw: dict) -> dict[str, object]:
    seen: set[str] = set()
    variants: dict[str, object] = {}

    for key in ("original", "hyde", "expanded", "step_back"):
        value = raw.get(key)
        if not isinstance(value, str):
            continue
        normalized = _normalize_text(value)
        if not normalized or normalized in seen:
            continue
        variants[key] = value.strip()
        seen.add(normalized)

    decomposed: list[str] = []
    for value in raw.get("decomposed", []) or []:
        if not isinstance(value, str):
            continue
        normalized = _normalize_text(value)
        if not normalized or normalized in seen:
            continue
        decomposed.append(value.strip())
        seen.add(normalized)

    if decomposed:
        variants["decomposed"] = decomposed

    if "original" not in variants and isinstance(raw.get("original"), str):
        variants["original"] = raw["original"].strip()
    return variants


def weighted_reciprocal_rank_fusion(
    candidate_lists: dict[str, list[RetrievalCandidate]],
    *,
    rrf_k: int,
    weights: dict[str, float],
    score_floor: float,
) -> list[RetrievalCandidate]:
    fused_scores: dict[str, float] = {}
    representatives: dict[str, RetrievalCandidate] = {}

    for list_name, candidates in candidate_lists.items():
        weight = weights.get(list_name, 1.0)
        for rank, candidate in enumerate(candidates, start=1):
            fused_scores[candidate.chunk_id] = fused_scores.get(candidate.chunk_id, 0.0) + (
                weight / (rrf_k + rank)
            )
            representatives.setdefault(candidate.chunk_id, candidate)

    fused: list[RetrievalCandidate] = []
    for chunk_id, score in fused_scores.items():
        if score < score_floor:
            continue
        candidate = representatives[chunk_id]
        fused.append(
            RetrievalCandidate(
                chunk_id=candidate.chunk_id,
                chunk=candidate.chunk,
                source_id=candidate.source_id,
                source_path=candidate.source_path,
                source_metadata=candidate.source_metadata,
                score=score,
            )
        )

    fused.sort(key=lambda item: item.score, reverse=True)
    return fused


def aggregate_root_score(
    *,
    root_score: float,
    first_hop_scores: list[float],
    second_hop_scores: list[float],
    root_weight: float,
    first_hop_weight: float,
    second_hop_weight: float,
    multi_path_bonus: float,
) -> float:
    return (
        root_weight * root_score
        + first_hop_weight * (max(first_hop_scores) if first_hop_scores else 0.0)
        + second_hop_weight * (max(second_hop_scores) if second_hop_scores else 0.0)
        + multi_path_bonus
    )


def _parse_json_response(content: str) -> dict:
    return json.loads(_strip_code_fences(content))


def _chat_json(model: str, prompt: str, *, timeout: int = 60) -> dict:
    _require_api_key()
    response = requests.post(
        _CHAT_COMPLETIONS_URL,
        headers=_openrouter_headers(),
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return _parse_json_response(response.json()["choices"][0]["message"]["content"])


def generate_query_variants(query: str, trace_logger: Optional[TraceLogger] = None) -> dict[str, object]:
    prompt = prompts.QUERY_VARIANTS.format(
        max_decomposed=settings.RETRIEVAL_MAX_DECOMPOSED_QUERIES,
        query=query,
    )
    raw = _chat_json(settings.MODEL_RETRIEVAL_QUERY_VARIANTS, prompt, timeout=90)
    raw["original"] = query
    variants = normalize_query_variants(raw)
    if trace_logger:
        trace_logger.emit(f"generated query variants: {json.dumps(variants, ensure_ascii=True)}")
    return variants


def _build_chunk_filter_sql(
    source_ids: list[str],
    filters: dict[str, str],
) -> tuple[str, list[object]]:
    conditions = [
        "c.deleted_at IS NULL",
        "s.deleted_at IS NULL",
        "j.status = 'completed'",
        "c.embedding IS NOT NULL",
    ]
    params: list[object] = []
    if source_ids:
        conditions.append("c.source_id = ANY(%s::uuid[])")
        params.append(source_ids)
    for key, value in filters.items():
        conditions.append("s.metadata ->> %s = %s")
        params.extend([key, value])
    return " AND ".join(conditions), params


def _vector_literal(vector: list[float]) -> str:
    return f"[{','.join(str(v) for v in vector)}]"


def _row_to_candidate(row) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=str(row[0]),
        chunk=row[1],
        source_id=str(row[2]),
        source_path=row[3] or "",
        source_metadata=row[4] or {},
        score=float(row[5]),
    )


def dense_retrieve(
    conn,
    query_text: str,
    *,
    source_ids: list[str],
    filters: dict[str, str],
    top_n: int,
    vector: Optional[list[float]] = None,
) -> list[RetrievalCandidate]:
    if vector is None:
        vector = get_embeddings([query_text])[0]
    where_sql, params = _build_chunk_filter_sql(source_ids, filters)
    vector_param = _vector_literal(vector)
    rows = conn.execute(
        f"""
        SELECT c.id, c.content, s.id, s.storage_path, s.metadata,
               (1 - (c.embedding <=> %s::vector)) AS score
        FROM chunks c
        JOIN sources s ON s.id = c.source_id
        JOIN jobs j ON j.id = c.job_id
        WHERE {where_sql}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (vector_param, *params, vector_param, top_n),
    ).fetchall()
    return [_row_to_candidate(row) for row in rows]


def sparse_retrieve(
    conn,
    query_text: str,
    *,
    source_ids: list[str],
    filters: dict[str, str],
    top_n: int,
) -> list[RetrievalCandidate]:
    where_sql, params = _build_chunk_filter_sql(source_ids, filters)
    config = settings.RETRIEVAL_TEXT_SEARCH_CONFIG
    if config == "english":
        rows = conn.execute(
            f"""
            WITH query AS (
                SELECT websearch_to_tsquery('english', %s) AS tsq
            )
            SELECT c.id, c.content, s.id, s.storage_path, s.metadata,
                   ts_rank_cd(
                       to_tsvector('english', coalesce(c.content, '')),
                       query.tsq
                   ) AS score
            FROM chunks c
            JOIN sources s ON s.id = c.source_id
            JOIN jobs j ON j.id = c.job_id
            CROSS JOIN query
            WHERE {where_sql}
              AND to_tsvector('english', coalesce(c.content, '')) @@ query.tsq
            ORDER BY score DESC
            LIMIT %s
            """,
            (query_text, *params, top_n),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            WITH query AS (
                SELECT websearch_to_tsquery(%s, %s) AS tsq
            )
            SELECT c.id, c.content, s.id, s.storage_path, s.metadata,
                   ts_rank_cd(
                       to_tsvector(%s, coalesce(c.content, '')),
                       query.tsq
                   ) AS score
            FROM chunks c
            JOIN sources s ON s.id = c.source_id
            JOIN jobs j ON j.id = c.job_id
            CROSS JOIN query
            WHERE {where_sql}
              AND to_tsvector(%s, coalesce(c.content, '')) @@ query.tsq
            ORDER BY score DESC
            LIMIT %s
            """,
            (config, query_text, config, *params, config, top_n),
        ).fetchall()
    return [_row_to_candidate(row) for row in rows]


def hybrid_search(
    query: str,
    *,
    limit: int,
    min_score: float,
) -> HybridSearchResults:
    vector = get_embeddings([query])[0]
    top_n = max(limit * 3, 20)
    with get_connection() as conn:
        dense = dense_retrieve(conn, query, source_ids=[], filters={}, top_n=top_n, vector=vector)
        sparse = sparse_retrieve(conn, query, source_ids=[], filters={}, top_n=top_n)
        insights = insight_hybrid_search(query, vector=vector, limit=limit, min_score=min_score, conn=conn)

    # RRF for deduplication and ordering; scores are unintuitive (~0.01-0.04)
    # so we replace them with the original cosine similarity (dense preferred over sparse)
    dense_scores = {c.chunk_id: c.score for c in dense}
    sparse_scores = {c.chunk_id: c.score for c in sparse}

    fused = weighted_reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse},
        rrf_k=settings.RETRIEVAL_RRF_K,
        weights={"dense": 1.0, "sparse": 1.0},
        score_floor=0.0,
    )

    results: list[RetrievalCandidate] = []
    for candidate in fused:
        score = dense_scores.get(candidate.chunk_id) or sparse_scores.get(candidate.chunk_id, 0.0)
        if score < min_score:
            continue
        results.append(
            RetrievalCandidate(
                chunk_id=candidate.chunk_id,
                chunk=candidate.chunk,
                source_id=candidate.source_id,
                source_path=candidate.source_path,
                source_metadata=candidate.source_metadata,
                score=score,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    results = results[:limit]
    with get_connection() as conn:
        expanded = _expand_chunk_texts(conn, [result.chunk_id for result in results])
    chunks = [
        RetrievalCandidate(
            chunk_id=result.chunk_id,
            chunk=expanded.get(result.chunk_id, result.chunk),
            source_id=result.source_id,
            source_path=result.source_path,
            source_metadata=result.source_metadata,
            score=result.score,
        )
        for result in results
    ]

    return HybridSearchResults(chunks=chunks, insights=insights)


def _insight_weighted_reciprocal_rank_fusion(
    candidate_lists: dict[str, list[InsightCandidate]],
    *,
    rrf_k: int,
    weights: dict[str, float],
    score_floor: float,
) -> list[InsightCandidate]:
    fused_scores: dict[str, float] = {}
    representatives: dict[str, InsightCandidate] = {}

    for list_name, candidates in candidate_lists.items():
        weight = weights.get(list_name, 1.0)
        for rank, candidate in enumerate(candidates, start=1):
            fused_scores[candidate.insight_id] = fused_scores.get(candidate.insight_id, 0.0) + (
                weight / (rrf_k + rank)
            )
            representatives.setdefault(candidate.insight_id, candidate)

    fused: list[InsightCandidate] = []
    for insight_id, score in fused_scores.items():
        if score < score_floor:
            continue
        candidate = representatives[insight_id]
        fused.append(
            InsightCandidate(
                insight_id=candidate.insight_id,
                insight=candidate.insight,
                score=score,
            )
        )

    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


def _insight_rows_to_candidates(rows) -> list[InsightCandidate]:
    return [
        InsightCandidate(
            insight_id=str(row[0]),
            insight=row[1],
            score=float(row[2]),
        )
        for row in rows
    ]


def insight_dense_retrieve(
    conn,
    vector: list[float],
    top_n: int,
) -> list[InsightCandidate]:
    vector_param = _vector_literal(vector)
    rows = conn.execute(
        f"""
        SELECT i.id, i.content,
               (1 - (i.embedding <=> %s::vector)) AS score
        FROM insights i
        WHERE i.embedding IS NOT NULL
        ORDER BY i.embedding <=> %s::vector
        LIMIT %s
        """,
        (vector_param, vector_param, top_n),
    ).fetchall()
    return _insight_rows_to_candidates(rows)


def insight_sparse_retrieve(
    conn,
    query_text: str,
    top_n: int,
) -> list[InsightCandidate]:
    config = settings.RETRIEVAL_TEXT_SEARCH_CONFIG
    if config == "english":
        rows = conn.execute(
            """
            WITH query AS (
                SELECT websearch_to_tsquery('english', %s) AS tsq
            )
            SELECT i.id, i.content,
                   ts_rank_cd(
                       to_tsvector('english', coalesce(i.content, '')),
                       query.tsq
                   ) AS score
            FROM insights i
            CROSS JOIN query
            WHERE to_tsvector('english', coalesce(i.content, '')) @@ query.tsq
            ORDER BY score DESC
            LIMIT %s
            """,
            (query_text, top_n),
        ).fetchall()
    else:
        rows = conn.execute(
            f"""
            WITH query AS (
                SELECT websearch_to_tsquery(%s, %s) AS tsq
            )
            SELECT i.id, i.content,
                   ts_rank_cd(
                       to_tsvector(%s, coalesce(i.content, '')),
                       query.tsq
                   ) AS score
            FROM insights i
            CROSS JOIN query
            WHERE to_tsvector(%s, coalesce(i.content, '')) @@ query.tsq
            ORDER BY score DESC
            LIMIT %s
            """,
            (config, query_text, config, config, top_n),
        ).fetchall()
    return _insight_rows_to_candidates(rows)


def _fetch_insight_sources_and_topics(
    conn,
    insight_ids: list[str],
) -> dict[str, tuple[list[str], list[InsightSourceRef]]]:
    if not insight_ids:
        return {}
    rows = conn.execute(
        """
        SELECT ci.insight_id, ci.topics, s.id, s.storage_path, s.metadata
        FROM chunk_insights ci
        JOIN chunks c ON c.id = ci.chunk_id
        JOIN sources s ON s.id = c.source_id
        JOIN jobs j ON j.id = c.job_id
        WHERE ci.insight_id = ANY(%s::uuid[])
          AND c.deleted_at IS NULL
          AND s.deleted_at IS NULL
          AND j.status = 'completed'
        """,
        (insight_ids,),
    ).fetchall()

    topics_map: dict[str, set[str]] = {}
    sources_map: dict[str, dict[str, InsightSourceRef]] = {}
    for row in rows:
        iid = str(row[0])
        row_topics = row[1] or []
        source_id = str(row[2])
        source_path = row[3] or ""
        source_metadata = row[4] or {}

        if iid not in topics_map:
            topics_map[iid] = set()
        for t in row_topics:
            topics_map[iid].add(t)

        if iid not in sources_map:
            sources_map[iid] = {}
        key = source_id
        if key not in sources_map[iid]:
            sources_map[iid][key] = InsightSourceRef(
                source_id=source_id,
                source_path=source_path,
                source_metadata=source_metadata,
            )

    result: dict[str, tuple[list[str], list[InsightSourceRef]]] = {}
    for iid in insight_ids:
        topics = sorted(topics_map.get(iid, set()))
        sources = list(sources_map.get(iid, {}).values())
        result[iid] = (topics, sources)
    return result


def insight_hybrid_search(
    query_text: str,
    *,
    vector: list[float],
    limit: int,
    min_score: float,
    conn,
) -> list[InsightSearchResult]:
    top_n = max(limit * 3, 20)
    dense = insight_dense_retrieve(conn, vector, top_n)
    sparse = insight_sparse_retrieve(conn, query_text, top_n)

    dense_scores = {c.insight_id: c.score for c in dense}
    sparse_scores = {c.insight_id: c.score for c in sparse}

    fused = _insight_weighted_reciprocal_rank_fusion(
        {"dense": dense, "sparse": sparse},
        rrf_k=settings.RETRIEVAL_RRF_K,
        weights={"dense": 1.0, "sparse": 1.0},
        score_floor=0.0,
    )

    results: list[InsightCandidate] = []
    for candidate in fused:
        score = dense_scores.get(candidate.insight_id) or sparse_scores.get(candidate.insight_id, 0.0)
        if score < min_score:
            continue
        results.append(
            InsightCandidate(
                insight_id=candidate.insight_id,
                insight=candidate.insight,
                score=score,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    results = results[:limit]

    if not results:
        return []

    detail = _fetch_insight_sources_and_topics(conn, [r.insight_id for r in results])
    return [
        InsightSearchResult(
            score=r.score,
            insight=r.insight,
            insight_id=r.insight_id,
            topics=detail.get(r.insight_id, ([], []))[0],
            sources=detail.get(r.insight_id, ([], []))[1],
        )
        for r in results
    ]


def _trace_candidates(
    trace_logger: Optional[TraceLogger],
    label: str,
    candidates: list[RetrievalCandidate],
) -> None:
    if not trace_logger:
        return
    top = candidates[: settings.RETRIEVAL_TRACE_MAX_CANDIDATES]
    preview = [
        {"chunk_id": c.chunk_id, "score": round(c.score, 4), "source_id": c.source_id}
        for c in top
    ]
    trace_logger.emit(f"{label}: {json.dumps(preview, ensure_ascii=True)}")


def run_first_stage_retrieval(
    *,
    conn=None,  # unused — each variant thread opens its own connection
    query: str,
    variants: dict[str, object],
    source_ids: list[str],
    filters: dict[str, str],
    rrf_k: int,
    trace_logger: Optional[TraceLogger] = None,
) -> list[RetrievalCandidate]:
    # Build the ordered list of (name, text, dense_only) specs
    variant_specs: list[tuple[str, str, bool]] = []
    variant_specs.append(("original", str(variants["original"]), False))
    if isinstance(variants.get("expanded"), str):
        variant_specs.append(("expanded", variants["expanded"], False))
    if isinstance(variants.get("step_back"), str):
        variant_specs.append(("step_back", variants["step_back"], False))
    if isinstance(variants.get("hyde"), str):
        variant_specs.append(("hyde", variants["hyde"], True))
    for index, subquery in enumerate(variants.get("decomposed", []) or []):
        variant_specs.append((f"decomposed_{index}", subquery, False))

    # Batch-embed all variant texts in a single API call
    all_texts = [text for _, text, _ in variant_specs]
    all_vectors = get_embeddings(all_texts)
    text_to_vector: dict[str, list[float]] = dict(zip(all_texts, all_vectors))

    candidate_lists: dict[str, list[RetrievalCandidate]] = {}
    weights: dict[str, float] = {}
    results_lock = threading.Lock()

    def _run_dense(name: str, text: str) -> None:
        with get_connection() as thread_conn:
            candidates = dense_retrieve(
                thread_conn,
                text,
                source_ids=source_ids,
                filters=filters,
                top_n=settings.RETRIEVAL_FIRST_STAGE_TOP_N,
                vector=text_to_vector[text],
            )
        _trace_candidates(trace_logger, f"dense hits for {name}", candidates)
        with results_lock:
            candidate_lists[f"dense:{name}"] = candidates
            weights[f"dense:{name}"] = settings.retrieval_variant_weight(name)

    def _run_sparse(name: str, text: str) -> None:
        with get_connection() as thread_conn:
            candidates = sparse_retrieve(
                thread_conn,
                text,
                source_ids=source_ids,
                filters=filters,
                top_n=settings.RETRIEVAL_FIRST_STAGE_TOP_N,
            )
        _trace_candidates(trace_logger, f"sparse hits for {name}", candidates)
        with results_lock:
            candidate_lists[f"sparse:{name}"] = candidates
            weights[f"sparse:{name}"] = settings.retrieval_variant_weight(name)

    futures = []
    with ThreadPoolExecutor(max_workers=len(variant_specs) * 2) as executor:
        for name, text, dense_only in variant_specs:
            futures.append(executor.submit(_run_dense, name, text))
            if not dense_only:
                futures.append(executor.submit(_run_sparse, name, text))

    for future in futures:
        future.result()  # re-raise any thread exception

    fused = weighted_reciprocal_rank_fusion(
        candidate_lists,
        rrf_k=rrf_k,
        weights=weights,
        score_floor=settings.RETRIEVAL_RRF_SCORE_FLOOR,
    )
    fused = fused[: settings.RETRIEVAL_FUSED_CANDIDATE_COUNT]
    _trace_candidates(trace_logger, "fused candidates", fused)
    return fused


def rerank_documents(
    query: str,
    documents: list[str],
    *,
    top_n: Optional[int] = None,
) -> list[dict]:
    _require_api_key()
    if not documents:
        return []
    response = requests.post(
        _RERANK_URL,
        headers=_openrouter_headers(),
        json={
            "model": settings.MODEL_RETRIEVAL_RERANKER,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def rerank_candidates(
    query: str,
    candidates: list[RetrievalCandidate],
    top_n: int,
    trace_logger: Optional[TraceLogger] = None,
) -> list[RetrievalCandidate]:
    results = rerank_documents(query, [candidate.chunk for candidate in candidates], top_n=top_n)
    reranked: list[RetrievalCandidate] = []
    for item in results:
        candidate = candidates[item["index"]]
        reranked.append(
            RetrievalCandidate(
                chunk_id=candidate.chunk_id,
                chunk=candidate.chunk,
                source_id=candidate.source_id,
                source_path=candidate.source_path,
                source_metadata=candidate.source_metadata,
                score=float(item["relevance_score"]),
            )
        )
    _trace_candidates(trace_logger, "reranked candidates", reranked)
    return reranked


def _select_entity_names(
    query: str,
    seed: RetrievalCandidate,
    entities: list[EntityCandidate],
    trace_logger: Optional[TraceLogger] = None,
) -> list[str]:
    if not entities:
        return []
    prompt = prompts.ENTITY_SELECTION.format(
        max_entities=settings.RETRIEVAL_ENTITY_SELECTION_COUNT,
        query=query,
        seed_chunk=seed.chunk[:1500],
        entities=json.dumps([entity.__dict__ for entity in entities], ensure_ascii=True),
    )
    try:
        response = _chat_json(settings.MODEL_RETRIEVAL_GRAPH, prompt, timeout=60)
        names = [value for value in response.get("selected_entities", []) if isinstance(value, str)]
    except Exception:
        names = [entity.name for entity in entities[: settings.RETRIEVAL_ENTITY_SELECTION_COUNT]]
    if trace_logger:
        trace_logger.emit(f"selected entities for seed {seed.chunk_id}: {json.dumps(names, ensure_ascii=True)}")
    return names[: settings.RETRIEVAL_ENTITY_SELECTION_COUNT]


def _generate_entity_query(
    query: str,
    seed: RetrievalCandidate,
    entity_name: str,
    *,
    entity_context: Optional[dict] = None,
) -> str:
    prompt = prompts.ENTITY_QUERY_GENERATION.format(
        query=query,
        seed_chunk=seed.chunk[:1200],
        entity_name=entity_name,
    )
    if entity_context:
        prompt += f"\nPath context:\n{json.dumps(entity_context, ensure_ascii=True)}\n"

    try:
        response = _chat_json(settings.MODEL_RETRIEVAL_GRAPH, prompt, timeout=60)
        value = response.get("query")
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass
    return f"{query} {entity_name}".strip()


def _load_seed_entities(driver, chunk_id: str) -> list[EntityCandidate]:
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id})-[:MENTIONS]->(e:Entity)
            RETURN e.entity_id AS entity_id, e.canonical_name AS name, e.entity_type AS entity_type
            """,
            chunk_id=chunk_id,
        )
        return [
            EntityCandidate(
                entity_id=record["entity_id"],
                name=record["name"],
                entity_type=record["entity_type"] or "",
            )
            for record in rows
        ]


def _load_entities_for_chunks(
    driver,
    chunk_ids: list[str],
) -> dict[str, list[EntityCandidate]]:
    with driver.session() as session:
        rows = session.run(
            """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})-[:MENTIONS]->(e:Entity)
            RETURN c.chunk_id AS chunk_id,
                   e.entity_id AS entity_id,
                   e.canonical_name AS name,
                   e.entity_type AS entity_type
            """,
            chunk_ids=chunk_ids,
        )
        entities_by_chunk: dict[str, dict[tuple[str, str], EntityCandidate]] = {}
        for record in rows:
            chunk_id = record["chunk_id"]
            chunk_entities = entities_by_chunk.setdefault(chunk_id, {})
            key = (record["name"] or "", record["entity_type"] or "")
            chunk_entities.setdefault(
                key,
                EntityCandidate(
                    entity_id=record["entity_id"],
                    name=record["name"],
                    entity_type=record["entity_type"] or "",
                ),
            )
        return {
            chunk_id: list(chunk_entities.values())
            for chunk_id, chunk_entities in entities_by_chunk.items()
        }


def _load_chunk_ids_for_entity(driver, entity_name: str, entity_type: str) -> list[str]:
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.canonical_name = $entity_name
              AND e.entity_type = $entity_type
            RETURN DISTINCT c.chunk_id AS chunk_id
            """,
            entity_name=entity_name,
            entity_type=entity_type,
        )
        return [record["chunk_id"] for record in rows]


def _fetch_chunk_candidates_by_ids(
    conn,
    chunk_ids: list[str],
    query_text: str,
    *,
    source_ids: list[str],
    filters: dict[str, str],
    limit: int,
    vector: Optional[list[float]] = None,
) -> list[RetrievalCandidate]:
    if not chunk_ids:
        return []
    if vector is None:
        vector = get_embeddings([query_text])[0]
    vector_param = _vector_literal(vector)
    where_sql, params = _build_chunk_filter_sql(source_ids, filters)
    rows = conn.execute(
        f"""
        SELECT c.id, c.content, s.id, s.storage_path, s.metadata,
               (1 - (c.embedding <=> %s::vector)) AS score
        FROM chunks c
        JOIN sources s ON s.id = c.source_id
        JOIN jobs j ON j.id = c.job_id
        WHERE {where_sql}
          AND c.id = ANY(%s::uuid[])
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (vector_param, *params, chunk_ids, vector_param, limit),
    ).fetchall()
    return [_row_to_candidate(row) for row in rows]


def _get_seed_chunk_context(conn, chunk_id: str) -> tuple[str, int] | None:
    row = conn.execute(
        """
        SELECT source_id, chunk_index
        FROM chunks
        WHERE id = %s
        """,
        (chunk_id,),
    ).fetchone()
    if not row:
        return None
    return str(row[0]), int(row[1])


def _fetch_same_source_neighbor_candidates(
    conn,
    seed: RetrievalCandidate,
    query_text: str,
    *,
    source_ids: list[str],
    filters: dict[str, str],
    limit: int,
    vector: Optional[list[float]] = None,
) -> list[RetrievalCandidate]:
    context = _get_seed_chunk_context(conn, seed.chunk_id)
    if not context:
        return []
    seed_source_id, seed_chunk_index = context
    if source_ids and seed_source_id not in source_ids:
        return []

    if vector is None:
        vector = get_embeddings([query_text])[0]
    vector_param = _vector_literal(vector)
    where_sql, params = _build_chunk_filter_sql(source_ids or [seed_source_id], filters)
    rows = conn.execute(
        f"""
        SELECT c.id, c.content, s.id, s.storage_path, s.metadata,
               (1 - (c.embedding <=> %s::vector)) AS score
        FROM chunks c
        JOIN sources s ON s.id = c.source_id
        JOIN jobs j ON j.id = c.job_id
        WHERE {where_sql}
          AND c.source_id = %s::uuid
          AND c.id <> %s::uuid
          AND c.chunk_index BETWEEN %s AND %s
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
        """,
        (
            vector_param,
            *params,
            seed_source_id,
            seed.chunk_id,
            max(0, seed_chunk_index - settings.RETRIEVAL_SAME_SOURCE_NEIGHBOR_WINDOW),
            seed_chunk_index + settings.RETRIEVAL_SAME_SOURCE_NEIGHBOR_WINDOW,
            vector_param,
            limit,
        ),
    ).fetchall()
    return [_row_to_candidate(row) for row in rows]


def _select_second_hop_entities_from_chunks(
    query: str,
    seed: RetrievalCandidate,
    entity_name: str,
    chunk_entity_map: dict[str, dict],
    trace_logger: Optional[TraceLogger] = None,
) -> list[tuple[RetrievalCandidate, EntityCandidate]]:
    if not chunk_entity_map:
        return []
    candidates: list[SecondHopEntityCandidate] = []
    for chunk_id, payload in chunk_entity_map.items():
        chunk = payload["chunk"]
        for entity in payload["entities"]:
            if entity.name == entity_name:
                continue
            candidates.append(
                SecondHopEntityCandidate(
                    chunk_id=chunk_id,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    chunk=chunk.chunk,
                )
            )

    deduped_candidates: list[SecondHopEntityCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        key = (candidate.chunk_id, candidate.name, candidate.entity_type)
        if key in seen:
            continue
        seen.add(key)
        deduped_candidates.append(candidate)
    if not deduped_candidates:
        return []

    prompt = prompts.SECOND_HOP_ENTITY_SELECTION.format(
        max_entities=settings.RETRIEVAL_SECOND_HOP_SELECTION_COUNT,
        query=query,
        seed_chunk=seed.chunk[:1200],
        entity_name=entity_name,
        candidates=json.dumps([candidate.__dict__ for candidate in deduped_candidates], ensure_ascii=True),
    )
    try:
        response = _chat_json(settings.MODEL_RETRIEVAL_GRAPH, prompt, timeout=60)
        names = [value for value in response.get("selected_entities", []) if isinstance(value, str)]
    except Exception:
        names = [candidate.name for candidate in deduped_candidates[: settings.RETRIEVAL_SECOND_HOP_SELECTION_COUNT]]

    selected: list[tuple[RetrievalCandidate, EntityCandidate]] = []
    for candidate in deduped_candidates:
        if candidate.name not in names:
            continue
        payload = chunk_entity_map[candidate.chunk_id]
        matching_entity = next(
            (
                entity
                for entity in payload["entities"]
                if entity.name == candidate.name and entity.entity_type == candidate.entity_type
            ),
            None,
        )
        if matching_entity is None:
            continue
        selected.append((payload["chunk"], matching_entity))
        if len(selected) >= settings.RETRIEVAL_SECOND_HOP_SELECTION_COUNT:
            break
    if trace_logger:
        trace_logger.emit(
            f"selected second-hop entities for {entity_name}: "
            f"{json.dumps([entity.name for _, entity in selected], ensure_ascii=True)}"
        )
    return selected


def _consume_llm_budget(budget: dict, lock: Optional[threading.Lock]) -> bool:
    """Atomically check and consume one LLM budget slot. Returns False if exhausted."""
    max_calls = settings.RETRIEVAL_MAX_GRAPH_LLM_CALLS
    if lock is not None:
        with lock:
            if budget["llm_calls"] >= max_calls:
                return False
            budget["llm_calls"] += 1
            return True
    if budget["llm_calls"] >= max_calls:
        return False
    budget["llm_calls"] += 1
    return True


def expand_seed_candidate(
    seed: RetrievalCandidate,
    query: str,
    source_ids: list[str],
    filters: dict[str, str],
    entity_confidence_threshold: float,
    first_hop_similarity_threshold: float,
    second_hop_similarity_threshold: float,
    *,
    conn,
    driver,
    trace_logger: Optional[TraceLogger] = None,
    budget: Optional[dict[str, float]] = None,
    budget_lock: Optional[threading.Lock] = None,
) -> dict:
    if budget is None:
        budget = {"llm_calls": 0, "query_started_at": time.monotonic()}
    seed_started_at = time.monotonic()

    root_result = {
        "score": seed.score,
        "chunk": seed.chunk,
        "chunk_id": seed.chunk_id,
        "source_id": seed.source_id,
        "source_path": seed.source_path,
        "source_metadata": seed.source_metadata,
        "related": [],
        "_root_score": seed.score,
        "_first_hop_scores": [],
        "_second_hop_scores": [],
        "_multi_path_bonus": 0.0,
    }

    def _time_exhausted() -> bool:
        return (time.monotonic() - seed_started_at) * 1000 >= settings.RETRIEVAL_MAX_GRAPH_EXPANSION_MS_PER_SEED

    if not _consume_llm_budget(budget, budget_lock):
        if trace_logger:
            trace_logger.emit(f"graph budget exhausted before expanding seed {seed.chunk_id}")
        return root_result
    if _time_exhausted():
        if trace_logger:
            trace_logger.emit(f"graph time budget exhausted before expanding seed {seed.chunk_id}")
        return root_result

    entities = _load_seed_entities(driver, seed.chunk_id)
    if trace_logger:
        trace_logger.emit(
            f"loaded {len(entities)} entities for seed {seed.chunk_id}: "
            f"{json.dumps([entity.name for entity in entities[:settings.RETRIEVAL_TRACE_MAX_ENTITIES]], ensure_ascii=True)}"
        )
    selected_names = _select_entity_names(query, seed, entities, trace_logger=trace_logger)
    selected_entities = [entity for entity in entities if entity.name in selected_names]

    for entity in selected_entities[: settings.RETRIEVAL_ENTITY_SELECTION_COUNT]:
        if not _consume_llm_budget(budget, budget_lock) or _time_exhausted():
            if trace_logger:
                trace_logger.emit(f"stopping lower-priority branches for seed {seed.chunk_id}")
            break

        first_hop_query = _generate_entity_query(query, seed, entity.name)
        first_hop_vector = get_embeddings([first_hop_query])[0]
        if trace_logger:
            trace_logger.emit(f"first-hop query for {entity.name}: {first_hop_query}")
        first_hop_candidates = _fetch_chunk_candidates_by_ids(
            conn,
            _load_chunk_ids_for_entity(driver, entity.name, entity.entity_type),
            first_hop_query,
            source_ids=source_ids,
            filters=filters,
            limit=settings.RETRIEVAL_FIRST_HOP_CHUNK_COUNT,
            vector=first_hop_vector,
        )
        first_hop_candidates = [
            candidate
            for candidate in first_hop_candidates
            if candidate.score >= first_hop_similarity_threshold and candidate.chunk_id != seed.chunk_id
        ][: settings.RETRIEVAL_FIRST_HOP_CHUNK_COUNT]
        if not first_hop_candidates:
            if trace_logger:
                trace_logger.emit(f"falling back to same-source neighbors for entity {entity.name}")
            first_hop_candidates = [
                candidate
                for candidate in _fetch_same_source_neighbor_candidates(
                    conn,
                    seed,
                    first_hop_query,
                    source_ids=source_ids,
                    filters=filters,
                limit=settings.RETRIEVAL_SAME_SOURCE_NEIGHBOR_COUNT,
                vector=first_hop_vector,
            )
                if candidate.score >= first_hop_similarity_threshold and candidate.chunk_id != seed.chunk_id
            ][: settings.RETRIEVAL_SAME_SOURCE_NEIGHBOR_COUNT]
        root_result["_first_hop_scores"].extend(candidate.score for candidate in first_hop_candidates)

        related = {
            "entity": entity.name,
            "chunks": [candidate.__dict__ for candidate in first_hop_candidates],
            "second_level_related": [],
        }

        first_hop_entity_map = _load_entities_for_chunks(
            driver,
            [candidate.chunk_id for candidate in first_hop_candidates],
        )
        candidate_entity_map = {
            candidate.chunk_id: {
                "chunk": candidate,
                "entities": [
                    chunk_entity
                    for chunk_entity in first_hop_entity_map.get(candidate.chunk_id, [])
                    if chunk_entity.name != entity.name
                ],
            }
            for candidate in first_hop_candidates
        }
        candidate_entity_map = {
            chunk_id: payload
            for chunk_id, payload in candidate_entity_map.items()
            if payload["entities"]
        }
        if candidate_entity_map and _consume_llm_budget(budget, budget_lock):
            selected_second_hops = _select_second_hop_entities_from_chunks(
                query,
                seed,
                entity.name,
                candidate_entity_map,
                trace_logger=trace_logger,
            )
        else:
            selected_second_hops = []

        for first_hop_chunk, second_hop in selected_second_hops:
            if not _consume_llm_budget(budget, budget_lock):
                break
            second_hop_query = _generate_entity_query(
                query,
                seed,
                second_hop.name,
                entity_context={
                    "entity1": entity.name,
                    "entity2": second_hop.name,
                    "first_hop_chunk": first_hop_chunk.chunk[:1200],
                },
            )
            second_hop_vector = get_embeddings([second_hop_query])[0]
            if trace_logger:
                trace_logger.emit(
                    f"second-hop query for {entity.name} via {first_hop_chunk.chunk_id} -> "
                    f"{second_hop.name}: {second_hop_query}"
                )
            second_hop_chunks = _fetch_chunk_candidates_by_ids(
                conn,
                _load_chunk_ids_for_entity(driver, second_hop.name, second_hop.entity_type),
                second_hop_query,
                source_ids=source_ids,
                filters=filters,
                limit=settings.RETRIEVAL_SECOND_HOP_CHUNK_COUNT,
                vector=second_hop_vector,
            )
            second_hop_chunks = [
                candidate
                for candidate in second_hop_chunks
                if candidate.score >= second_hop_similarity_threshold and candidate.chunk_id != seed.chunk_id
            ][: settings.RETRIEVAL_SECOND_HOP_CHUNK_COUNT]
            root_result["_second_hop_scores"].extend(candidate.score for candidate in second_hop_chunks)
            if first_hop_candidates and second_hop_chunks:
                root_result["_multi_path_bonus"] = settings.RETRIEVAL_MULTI_PATH_BONUS
            related["second_level_related"].append(
                {
                    "entity": second_hop.name,
                    "relationship": {
                        "label": "CO_MENTIONED_IN_FIRST_HOP_CHUNK",
                        "metadata": {
                            "entity1": entity.name,
                            "first_hop_chunk_id": first_hop_chunk.chunk_id,
                        },
                    },
                    "chunks": [candidate.__dict__ for candidate in second_hop_chunks],
                }
            )

        root_result["related"].append(related)

    return root_result


def finalize_root_results(
    query: str,
    root_results: list[dict],
    result_count: int,
    trace_logger: Optional[TraceLogger] = None,
) -> list[dict]:
    def _dedupe_chunks(chunks: list[dict]) -> list[dict]:
        deduped: dict[str, dict] = {}
        for chunk in chunks:
            deduped.setdefault(chunk["chunk_id"], chunk)
        return list(deduped.values())

    # Deduplicate within each root first
    for root in root_results:
        for related in root["related"]:
            related["chunks"] = _dedupe_chunks(related["chunks"])
            for second_level in related["second_level_related"]:
                second_level["chunks"] = _dedupe_chunks(second_level["chunks"])

    # Collect all unique related chunks across all roots for a single reranker call
    all_related_by_id: dict[str, dict] = {}
    for root in root_results:
        for related in root["related"]:
            for chunk in related["chunks"]:
                all_related_by_id.setdefault(chunk["chunk_id"], chunk)
            for second_level in related["second_level_related"]:
                for chunk in second_level["chunks"]:
                    all_related_by_id.setdefault(chunk["chunk_id"], chunk)

    if all_related_by_id:
        unique_chunks = list(all_related_by_id.values())
        reranked_all = rerank_documents(
            query,
            [chunk["chunk"] for chunk in unique_chunks],
            top_n=len(unique_chunks),
        )
        global_rerank_scores: dict[str, float] = {
            unique_chunks[item["index"]]["chunk_id"]: float(item["relevance_score"])
            for item in reranked_all
        }
    else:
        global_rerank_scores = {}

    final_results: list[dict] = []
    for root in root_results:
        if global_rerank_scores:
            first_hop_scores: list[float] = []
            second_hop_scores: list[float] = []
            for related in root["related"]:
                first_hop_scores.extend(
                    global_rerank_scores.get(chunk["chunk_id"], chunk["score"])
                    for chunk in related["chunks"]
                )
                for second_level in related["second_level_related"]:
                    second_hop_scores.extend(
                        global_rerank_scores.get(chunk["chunk_id"], chunk["score"])
                        for chunk in second_level["chunks"]
                    )
        else:
            first_hop_scores = root["_first_hop_scores"]
            second_hop_scores = root["_second_hop_scores"]

        root["_final_score"] = aggregate_root_score(
            root_score=root["_root_score"],
            first_hop_scores=first_hop_scores,
            second_hop_scores=second_hop_scores,
            root_weight=settings.RETRIEVAL_FINAL_ROOT_WEIGHT,
            first_hop_weight=settings.RETRIEVAL_FINAL_FIRST_HOP_WEIGHT,
            second_hop_weight=settings.RETRIEVAL_FINAL_SECOND_HOP_WEIGHT,
            multi_path_bonus=root["_multi_path_bonus"],
        )
        root["score"] = root["_final_score"]
        if trace_logger:
            trace_logger.emit(
                f"final score for root {root['chunk_id']}: "
                f"{round(root['_final_score'], 4)} "
                f"(root={round(root['_root_score'], 4)}, "
                f"first={round(max(first_hop_scores) if first_hop_scores else 0.0, 4)}, "
                f"second={round(max(second_hop_scores) if second_hop_scores else 0.0, 4)}, "
                f"bonus={round(root['_multi_path_bonus'], 4)})"
            )
        final_results.append(root)

    deduped: dict[str, dict] = {}
    for root in sorted(final_results, key=lambda item: item["_final_score"], reverse=True):
        deduped.setdefault(root["chunk_id"], root)

    ordered = list(deduped.values())[:result_count]
    for root in ordered:
        root.pop("_root_score", None)
        root.pop("_first_hop_scores", None)
        root.pop("_second_hop_scores", None)
        root.pop("_multi_path_bonus", None)
        root.pop("_final_score", None)
    return ordered


def _expand_chunk_texts(conn, chunk_ids: list[str]) -> dict[str, str]:
    unique_chunk_ids = list(dict.fromkeys(chunk_ids))
    if not unique_chunk_ids:
        return {}
    rows = conn.execute(
        """
        WITH selected AS (
            SELECT id, source_id, chunk_index, content
            FROM chunks
            WHERE id = ANY(%s::uuid[])
        )
        SELECT selected.id::text AS center_id,
               neighbor.id::text AS chunk_id,
               neighbor.source_id::text,
               neighbor.chunk_index,
               neighbor.content
        FROM selected
        JOIN chunks neighbor
          ON neighbor.source_id = selected.source_id
         AND neighbor.deleted_at IS NULL
         AND neighbor.chunk_index BETWEEN selected.chunk_index - %s AND selected.chunk_index + %s
        ORDER BY selected.id, neighbor.chunk_index
        """,
        (
            unique_chunk_ids,
            settings.RETRIEVAL_SAME_SOURCE_NEIGHBOR_WINDOW,
            settings.RETRIEVAL_SAME_SOURCE_NEIGHBOR_WINDOW,
        ),
    ).fetchall()

    grouped: dict[str, dict[str, object]] = {}
    for center_id, chunk_id, source_id, chunk_index, content in rows:
        payload = grouped.setdefault(
            str(center_id),
            {"center": None, "neighbors": []},
        )
        item = {
            "chunk_id": str(chunk_id),
            "source_id": str(source_id),
            "chunk_index": int(chunk_index),
            "content": content or "",
        }
        if str(chunk_id) == str(center_id):
            payload["center"] = item
        payload["neighbors"].append(item)

    expanded_by_id: dict[str, str] = {}
    min_tokens = settings.RETRIEVAL_EXPANSION_MIN_TOKENS
    max_tokens = settings.RETRIEVAL_EXPANSION_MAX_TOKENS

    for center_id, payload in grouped.items():
        center = payload["center"]
        if not center:
            continue
        neighbors = payload["neighbors"]
        by_index = {item["chunk_index"]: item for item in neighbors}
        assembled = center["content"]
        if _token_count(assembled) >= min_tokens:
            expanded_by_id[center_id] = assembled
            continue

        center_index = center["chunk_index"]
        max_index = max(by_index)
        min_index = min(by_index)

        for index in range(center_index + 1, max_index + 1):
            candidate = by_index.get(index)
            if not candidate:
                continue
            maybe = f"{assembled}\n\n{candidate['content']}".strip()
            if _token_count(maybe) > max_tokens:
                break
            assembled = maybe
            if _token_count(assembled) >= min_tokens:
                break

        if _token_count(assembled) < min_tokens:
            for index in range(center_index - 1, min_index - 1, -1):
                candidate = by_index.get(index)
                if not candidate:
                    continue
                maybe = f"{candidate['content']}\n\n{assembled}".strip()
                if _token_count(maybe) > max_tokens:
                    break
                assembled = maybe
                if _token_count(assembled) >= min_tokens:
                    break

        expanded_by_id[center_id] = assembled

    return expanded_by_id


def _apply_expanded_chunk_text(retrieval_results: list[dict], expanded_by_id: dict[str, str]) -> None:
    for root in retrieval_results:
        root["chunk"] = expanded_by_id.get(root["chunk_id"], root["chunk"])
        for related in root["related"]:
            for chunk in related["chunks"]:
                chunk["chunk"] = expanded_by_id.get(chunk["chunk_id"], chunk["chunk"])
            for second_level in related["second_level_related"]:
                for chunk in second_level["chunks"]:
                    chunk["chunk"] = expanded_by_id.get(chunk["chunk_id"], chunk["chunk"])


def _expand_neighbor_contexts(conn, retrieval_results: list[dict]) -> None:
    chunk_ids: set[str] = set()
    for root in retrieval_results:
        chunk_ids.add(root["chunk_id"])
        for related in root["related"]:
            for chunk in related["chunks"]:
                chunk_ids.add(chunk["chunk_id"])
            for second_level in related["second_level_related"]:
                for chunk in second_level["chunks"]:
                    chunk_ids.add(chunk["chunk_id"])
    expanded_by_id = _expand_chunk_texts(conn, list(chunk_ids))
    _apply_expanded_chunk_text(retrieval_results, expanded_by_id)


def _resolved_params(
    *,
    seed_count: Optional[int],
    result_count: Optional[int],
    rrf_k: Optional[int],
    entity_confidence_threshold: Optional[float],
    first_hop_similarity_threshold: Optional[float],
    second_hop_similarity_threshold: Optional[float],
) -> RetrievalParams:
    return RetrievalParams(
        seed_count=seed_count or settings.RETRIEVAL_SEED_COUNT,
        result_count=result_count or settings.RETRIEVAL_RESULT_COUNT,
        rrf_k=rrf_k or settings.RETRIEVAL_RRF_K,
        entity_confidence_threshold=(
            entity_confidence_threshold or settings.RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD
        ),
        first_hop_similarity_threshold=(
            first_hop_similarity_threshold or settings.RETRIEVAL_FIRST_HOP_SIMILARITY_THRESHOLD
        ),
        second_hop_similarity_threshold=(
            second_hop_similarity_threshold or settings.RETRIEVAL_SECOND_HOP_SIMILARITY_THRESHOLD
        ),
    )


def retrieve(
    *,
    query: str,
    source_ids: list[str],
    filters: dict[str, str],
    seed_count: Optional[int],
    result_count: Optional[int],
    rrf_k: Optional[int],
    entity_confidence_threshold: Optional[float],
    first_hop_similarity_threshold: Optional[float],
    second_hop_similarity_threshold: Optional[float],
    trace: bool,
    trace_printer: Optional[Callable[[str], None]],
) -> dict:
    _require_api_key()
    trace_logger = TraceLogger(enabled=trace, printer=trace_printer)
    params = _resolved_params(
        seed_count=seed_count,
        result_count=result_count,
        rrf_k=rrf_k,
        entity_confidence_threshold=entity_confidence_threshold,
        first_hop_similarity_threshold=first_hop_similarity_threshold,
        second_hop_similarity_threshold=second_hop_similarity_threshold,
    )
    trace_logger.emit(
        "starting retrieval with params: "
        + json.dumps(
            {
                "query": query,
                "source_ids": source_ids,
                "filters": filters,
                "seed_count": params.seed_count,
                "result_count": params.result_count,
                "rrf_k": params.rrf_k,
            },
            ensure_ascii=True,
        )
    )

    variants = generate_query_variants(query, trace_logger=trace_logger)
    # first-stage retrieval opens its own per-thread connections internally
    with get_graph_driver() as driver:
        fused_candidates = run_first_stage_retrieval(
            conn=None,  # unused — each thread opens its own connection
            query=query,
            variants=variants,
            source_ids=source_ids,
            filters=filters,
            rrf_k=params.rrf_k,
            trace_logger=trace_logger,
        )
        seed_candidates = rerank_candidates(
            query,
            fused_candidates,
            top_n=params.seed_count,
            trace_logger=trace_logger,
        )[: params.seed_count]
        trace_logger.emit(f"selected {len(seed_candidates)} seed chunks")

        budget: dict = {"llm_calls": 0, "started_at": time.monotonic()}
        budget_lock = threading.Lock()

        def _expand_one(seed: RetrievalCandidate) -> dict:
            with get_connection() as seed_conn:
                return expand_seed_candidate(
                    seed,
                    query,
                    source_ids,
                    filters,
                    params.entity_confidence_threshold,
                    params.first_hop_similarity_threshold,
                    params.second_hop_similarity_threshold,
                    conn=seed_conn,
                    driver=driver,
                    trace_logger=trace_logger,
                    budget=budget,
                    budget_lock=budget_lock,
                )

        with ThreadPoolExecutor(max_workers=len(seed_candidates)) as pool:
            root_results = list(pool.map(_expand_one, seed_candidates))

        final_results = finalize_root_results(
            query,
            root_results,
            params.result_count,
            trace_logger=trace_logger,
        )
        with get_connection() as conn:
            _expand_neighbor_contexts(conn, final_results)
    return {"retrieval_results": final_results}


def resolve_retrieval_scope(
    *,
    query: str,
    source_ids: list[str],
    filters: dict[str, str],
    seed_count: Optional[int],
    result_count: Optional[int],
    rrf_k: Optional[int],
    entity_confidence_threshold: Optional[float],
    first_hop_similarity_threshold: Optional[float],
    second_hop_similarity_threshold: Optional[float],
    trace: bool,
    trace_printer: Optional[Callable[[str], None]],
) -> list[str]:
    _require_api_key()
    trace_logger = TraceLogger(enabled=trace, printer=trace_printer)
    params = _resolved_params(
        seed_count=seed_count,
        result_count=result_count,
        rrf_k=rrf_k,
        entity_confidence_threshold=entity_confidence_threshold,
        first_hop_similarity_threshold=first_hop_similarity_threshold,
        second_hop_similarity_threshold=second_hop_similarity_threshold,
    )

    variants = generate_query_variants(query, trace_logger=trace_logger)
    fused_candidates = run_first_stage_retrieval(
        conn=None,
        query=query,
        variants=variants,
        source_ids=source_ids,
        filters=filters,
        rrf_k=params.rrf_k,
        trace_logger=trace_logger,
    )
    seed_candidates = rerank_candidates(
        query,
        fused_candidates,
        top_n=params.seed_count,
        trace_logger=trace_logger,
    )[: params.seed_count]

    resolved: list[str] = []
    seen: set[str] = set()
    for seed in seed_candidates:
        if seed.source_id in seen:
            continue
        seen.add(seed.source_id)
        resolved.append(seed.source_id)
    return resolved
