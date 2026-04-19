import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

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
class EntityCandidate:
    entity_id: str
    name: str
    entity_type: str


@dataclass
class SecondHopEntityCandidate:
    entity_id: str
    name: str
    entity_type: str
    relationship_label: str
    relationship_metadata: dict


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
    prompt = f"""You are generating bounded retrieval query variants.
Return ONLY a JSON object with keys: original, hyde, expanded, step_back, decomposed.
- original must be the exact input query
- hyde must be a short hypothetical answer passage for dense retrieval
- expanded must add synonyms, aliases, abbreviations, and related terms
- step_back must be a more general background query
- decomposed must contain at most {settings.RETRIEVAL_MAX_DECOMPOSED_QUERIES} focused sub-queries
- avoid near-duplicate variants

Query:
{query}
"""
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
) -> list[RetrievalCandidate]:
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
    rows = conn.execute(
        f"""
        SELECT c.id, c.content, s.id, s.storage_path, s.metadata,
               ts_rank_cd(
                   to_tsvector(%s, coalesce(c.content, '')),
                   websearch_to_tsquery(%s, %s)
               ) AS score
        FROM chunks c
        JOIN sources s ON s.id = c.source_id
        JOIN jobs j ON j.id = c.job_id
        WHERE {where_sql}
          AND to_tsvector(%s, coalesce(c.content, '')) @@ websearch_to_tsquery(%s, %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (config, config, query_text, *params, config, config, query_text, top_n),
    ).fetchall()
    return [_row_to_candidate(row) for row in rows]


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
    conn,
    query: str,
    variants: dict[str, object],
    source_ids: list[str],
    filters: dict[str, str],
    rrf_k: int,
    trace_logger: Optional[TraceLogger] = None,
) -> list[RetrievalCandidate]:
    candidate_lists: dict[str, list[RetrievalCandidate]] = {}
    weights: dict[str, float] = {}

    def _add_variant(name: str, variant_text: str, *, dense_only: bool = False) -> None:
        dense_key = f"dense:{name}"
        dense_candidates = dense_retrieve(
            conn,
            variant_text,
            source_ids=source_ids,
            filters=filters,
            top_n=settings.RETRIEVAL_FIRST_STAGE_TOP_N,
        )
        candidate_lists[dense_key] = dense_candidates
        weights[dense_key] = settings.retrieval_variant_weight(name)
        _trace_candidates(trace_logger, f"dense hits for {name}", dense_candidates)

        if dense_only:
            return

        sparse_key = f"sparse:{name}"
        sparse_candidates = sparse_retrieve(
            conn,
            variant_text,
            source_ids=source_ids,
            filters=filters,
            top_n=settings.RETRIEVAL_FIRST_STAGE_TOP_N,
        )
        candidate_lists[sparse_key] = sparse_candidates
        weights[sparse_key] = settings.retrieval_variant_weight(name)
        _trace_candidates(trace_logger, f"sparse hits for {name}", sparse_candidates)

    _add_variant("original", str(variants["original"]))
    if isinstance(variants.get("expanded"), str):
        _add_variant("expanded", variants["expanded"])
    if isinstance(variants.get("step_back"), str):
        _add_variant("step_back", variants["step_back"])
    if isinstance(variants.get("hyde"), str):
        _add_variant("hyde", variants["hyde"], dense_only=True)
    for index, subquery in enumerate(variants.get("decomposed", []) or []):
        _add_variant(f"decomposed_{index}", subquery)

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
    prompt = f"""You are selecting graph entities to expand for retrieval.
Return ONLY a JSON object with key selected_entities containing up to {settings.RETRIEVAL_ENTITY_SELECTION_COUNT} entity names.

User query:
{query}

Seed chunk:
{seed.chunk[:1500]}

Entities:
{json.dumps([entity.__dict__ for entity in entities], ensure_ascii=True)}
"""
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
    relationship_label: Optional[str] = None,
    relationship_metadata: Optional[dict] = None,
) -> str:
    prompt = f"""You are generating a retrieval sub-query.
Return ONLY a JSON object with key query.

Original user query:
{query}

Seed chunk:
{seed.chunk[:1200]}

Entity:
{entity_name}
"""
    if relationship_label:
        prompt += f"\nRelationship label:\n{relationship_label}\n"
    if relationship_metadata:
        prompt += f"\nRelationship metadata:\n{json.dumps(relationship_metadata, ensure_ascii=True)}\n"

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
            MATCH (c:Chunk {chunk_id: $chunk_id})-[r]-(e:Entity)
            WHERE type(r) IN ['MENTIONS', 'MENTIONED_IN']
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


def _load_second_hop_entities(
    driver,
    entity_id: str,
    confidence_threshold: float,
) -> list[SecondHopEntityCandidate]:
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (e1:Entity {entity_id: $entity_id})-[r:RELATED_TO]->(e2:Entity)
            WHERE coalesce(r.confidence, 0.0) >= $confidence_threshold
            RETURN e2.entity_id AS entity_id,
                   e2.canonical_name AS name,
                   e2.entity_type AS entity_type,
                   r.type AS relationship_label,
                   properties(r) AS relationship_metadata
            """,
            entity_id=entity_id,
            confidence_threshold=confidence_threshold,
        )
        return [
            SecondHopEntityCandidate(
                entity_id=record["entity_id"],
                name=record["name"],
                entity_type=record["entity_type"] or "",
                relationship_label=record["relationship_label"] or "RELATED_TO",
                relationship_metadata=record["relationship_metadata"] or {},
            )
            for record in rows
        ]


def _load_chunk_ids_for_entity(driver, entity_id: str) -> list[str]:
    with driver.session() as session:
        rows = session.run(
            """
            MATCH (c:Chunk)-[r]-(e:Entity {entity_id: $entity_id})
            WHERE type(r) IN ['MENTIONS', 'MENTIONED_IN']
            RETURN c.chunk_id AS chunk_id
            """,
            entity_id=entity_id,
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
) -> list[RetrievalCandidate]:
    if not chunk_ids:
        return []
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
) -> list[RetrievalCandidate]:
    context = _get_seed_chunk_context(conn, seed.chunk_id)
    if not context:
        return []
    seed_source_id, seed_chunk_index = context
    if source_ids and seed_source_id not in source_ids:
        return []

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


def _select_second_hop_entities(
    query: str,
    seed: RetrievalCandidate,
    entity_name: str,
    candidates: list[SecondHopEntityCandidate],
    trace_logger: Optional[TraceLogger] = None,
) -> list[SecondHopEntityCandidate]:
    if not candidates:
        return []
    prompt = f"""You are selecting second-hop graph entities to expand for retrieval.
Return ONLY a JSON object with key selected_entities containing up to {settings.RETRIEVAL_SECOND_HOP_SELECTION_COUNT} entity names.

User query:
{query}

Seed chunk:
{seed.chunk[:1200]}

Current entity:
{entity_name}

Candidates:
{json.dumps([candidate.__dict__ for candidate in candidates], ensure_ascii=True)}
"""
    try:
        response = _chat_json(settings.MODEL_RETRIEVAL_GRAPH, prompt, timeout=60)
        names = [value for value in response.get("selected_entities", []) if isinstance(value, str)]
    except Exception:
        names = [candidate.name for candidate in candidates[: settings.RETRIEVAL_SECOND_HOP_SELECTION_COUNT]]

    selected = [candidate for candidate in candidates if candidate.name in names]
    selected = selected[: settings.RETRIEVAL_SECOND_HOP_SELECTION_COUNT]
    if trace_logger:
        trace_logger.emit(
            f"selected second-hop entities for {entity_name}: "
            f"{json.dumps([candidate.name for candidate in selected], ensure_ascii=True)}"
        )
    return selected


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

    if budget["llm_calls"] >= settings.RETRIEVAL_MAX_GRAPH_LLM_CALLS:
        if trace_logger:
            trace_logger.emit(f"graph budget exhausted before expanding seed {seed.chunk_id}")
        return root_result
    if (time.monotonic() - seed_started_at) * 1000 >= settings.RETRIEVAL_MAX_GRAPH_EXPANSION_MS_PER_SEED:
        if trace_logger:
            trace_logger.emit(f"graph time budget exhausted before expanding seed {seed.chunk_id}")
        return root_result

    entities = _load_seed_entities(driver, seed.chunk_id)
    if trace_logger:
        trace_logger.emit(
            f"loaded {len(entities)} entities for seed {seed.chunk_id}: "
            f"{json.dumps([entity.name for entity in entities[:settings.RETRIEVAL_TRACE_MAX_ENTITIES]], ensure_ascii=True)}"
        )
    budget["llm_calls"] += 1
    selected_names = _select_entity_names(query, seed, entities, trace_logger=trace_logger)
    selected_entities = [entity for entity in entities if entity.name in selected_names]

    for entity in selected_entities[: settings.RETRIEVAL_ENTITY_SELECTION_COUNT]:
        budget_hit = budget["llm_calls"] >= settings.RETRIEVAL_MAX_GRAPH_LLM_CALLS
        time_hit = (time.monotonic() - seed_started_at) * 1000 >= settings.RETRIEVAL_MAX_GRAPH_EXPANSION_MS_PER_SEED
        if budget_hit or time_hit:
            if trace_logger:
                trace_logger.emit(f"stopping lower-priority branches for seed {seed.chunk_id}")
            break

        budget["llm_calls"] += 1
        first_hop_query = _generate_entity_query(query, seed, entity.name)
        if trace_logger:
            trace_logger.emit(f"first-hop query for {entity.name}: {first_hop_query}")
        first_hop_candidates = _fetch_chunk_candidates_by_ids(
            conn,
            _load_chunk_ids_for_entity(driver, entity.entity_id),
            first_hop_query,
            source_ids=source_ids,
            filters=filters,
            limit=settings.RETRIEVAL_FIRST_HOP_CHUNK_COUNT,
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
                )
                if candidate.score >= first_hop_similarity_threshold and candidate.chunk_id != seed.chunk_id
            ][: settings.RETRIEVAL_SAME_SOURCE_NEIGHBOR_COUNT]
        root_result["_first_hop_scores"].extend(candidate.score for candidate in first_hop_candidates)

        related = {
            "entity": entity.name,
            "chunks": [candidate.__dict__ for candidate in first_hop_candidates],
            "second_level_related": [],
        }

        second_hop_candidates = _load_second_hop_entities(driver, entity.entity_id, entity_confidence_threshold)
        if second_hop_candidates:
            budget["llm_calls"] += 1
            selected_second_hops = _select_second_hop_entities(
                query,
                seed,
                entity.name,
                second_hop_candidates,
                trace_logger=trace_logger,
            )
        else:
            selected_second_hops = []

        for second_hop in selected_second_hops:
            if budget["llm_calls"] >= settings.RETRIEVAL_MAX_GRAPH_LLM_CALLS:
                break
            budget["llm_calls"] += 1
            second_hop_query = _generate_entity_query(
                query,
                seed,
                second_hop.name,
                relationship_label=second_hop.relationship_label,
                relationship_metadata=second_hop.relationship_metadata,
            )
            if trace_logger:
                trace_logger.emit(
                    f"second-hop query for {entity.name} -> {second_hop.relationship_label} -> "
                    f"{second_hop.name}: {second_hop_query}"
                )
            second_hop_chunks = _fetch_chunk_candidates_by_ids(
                conn,
                _load_chunk_ids_for_entity(driver, second_hop.entity_id),
                second_hop_query,
                source_ids=source_ids,
                filters=filters,
                limit=settings.RETRIEVAL_SECOND_HOP_CHUNK_COUNT,
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
                        "label": second_hop.relationship_label,
                        "metadata": second_hop.relationship_metadata,
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

    final_results: list[dict] = []
    for root in root_results:
        for related in root["related"]:
            related["chunks"] = _dedupe_chunks(related["chunks"])
            for second_level in related["second_level_related"]:
                second_level["chunks"] = _dedupe_chunks(second_level["chunks"])

        related_chunks: list[dict] = []
        for related in root["related"]:
            related_chunks.extend(related["chunks"])
            for second_level in related["second_level_related"]:
                related_chunks.extend(second_level["chunks"])

        if related_chunks:
            reranked_related = rerank_documents(
                query,
                [chunk["chunk"] for chunk in related_chunks],
                top_n=len(related_chunks),
            )
            reranked_scores = {
                related_chunks[item["index"]]["chunk_id"]: float(item["relevance_score"])
                for item in reranked_related
            }
            first_hop_scores: list[float] = []
            second_hop_scores: list[float] = []
            for related in root["related"]:
                first_hop_scores.extend(
                    reranked_scores.get(chunk["chunk_id"], chunk["score"])
                    for chunk in related["chunks"]
                )
                for second_level in related["second_level_related"]:
                    second_hop_scores.extend(
                        reranked_scores.get(chunk["chunk_id"], chunk["score"])
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
    with get_connection() as conn, get_graph_driver() as driver:
        fused_candidates = run_first_stage_retrieval(
            conn=conn,
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

        budget = {"llm_calls": 0, "started_at": time.monotonic()}
        root_results = [
            expand_seed_candidate(
                seed,
                query,
                source_ids,
                filters,
                params.entity_confidence_threshold,
                params.first_hop_similarity_threshold,
                params.second_hop_similarity_threshold,
                conn=conn,
                driver=driver,
                trace_logger=trace_logger,
                budget=budget,
            )
            for seed in seed_candidates
        ]
        final_results = finalize_root_results(
            query,
            root_results,
            params.result_count,
            trace_logger=trace_logger,
        )
    return {"retrieval_results": final_results}
