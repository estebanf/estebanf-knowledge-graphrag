from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import igraph
import leidenalg
import requests

from rag.config import settings
from rag.db import get_connection
from rag.graph_db import get_graph_driver
from rag.retrieval import hybrid_search, retrieve


@dataclass
class EntityNode:
    entity_id: str
    canonical_name: str
    entity_type: str
    source_ids: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)


@dataclass
class ChunkResult:
    chunk_id: str
    source_id: str
    source_name: str
    entity_overlap_count: int
    score: float
    content: str


@dataclass
class ContributingSource:
    source_id: str
    source_name: str


@dataclass
class Community:
    community_id: str
    is_cross_source: bool
    entity_count: int
    entities: list[dict]
    contributing_sources: list[ContributingSource]
    chunks: list[ChunkResult]
    summary: str = ""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _resolve_scope(
    scope_mode: str,
    source_ids: list[str],
    criteria: list[str],
    filters: dict[str, str],
    search_options: dict,
    retrieve_options: dict,
) -> list[str]:
    if scope_mode == "ids":
        return list(dict.fromkeys(source_ids))

    if scope_mode == "search":
        limit = search_options.get("limit", 10)
        min_score = search_options.get("min_score", 0.0)
        seen: set[str] = set()
        for criterion in criteria:
            for r in hybrid_search(criterion, limit=limit, min_score=min_score):
                seen.add(r.source_id)
        return list(seen)

    if scope_mode == "retrieve":
        seen = set()
        for criterion in criteria:
            result = retrieve(
                query=criterion,
                source_ids=[],
                filters=filters,
                seed_count=retrieve_options.get("seed_count"),
                result_count=retrieve_options.get("result_count"),
                rrf_k=retrieve_options.get("rrf_k"),
                entity_confidence_threshold=retrieve_options.get("entity_confidence_threshold"),
                first_hop_similarity_threshold=retrieve_options.get("first_hop_similarity_threshold"),
                second_hop_similarity_threshold=retrieve_options.get("second_hop_similarity_threshold"),
                trace=retrieve_options.get("trace", False),
                trace_printer=None,
            )
            for item in result.get("retrieval_results", []):
                seen.add(item["source_id"])
                for rel in item.get("related", []):
                    for chunk in rel.get("chunks", []):
                        seen.add(chunk["source_id"])
        return list(seen)

    raise ValueError(f"Unknown scope_mode: {scope_mode!r}")


def _load_graph_data(
    source_ids: list[str],
) -> tuple[dict[str, EntityNode], dict[str, str], list[str]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, source_id FROM chunks WHERE source_id = ANY(%s) AND deleted_at IS NULL",
            (source_ids,),
        ).fetchall()
    chunk_to_source: dict[str, str] = {str(row[0]): str(row[1]) for row in rows}

    if not chunk_to_source:
        return {}, {}, list(source_ids)

    chunk_ids = list(chunk_to_source.keys())
    with get_graph_driver() as driver:
        with driver.session() as session:
            records = session.run(
                """
                UNWIND $chunk_ids AS cid
                MATCH (c:Chunk {chunk_id: cid})-[:MENTIONS]->(e:Entity)
                RETURN cid, e.entity_id AS entity_id,
                       e.canonical_name AS canonical_name,
                       e.entity_type AS entity_type
                """,
                chunk_ids=chunk_ids,
            ).data()

    entities: dict[str, EntityNode] = {}
    connected_sources: set[str] = set()

    for rec in records:
        eid = rec["entity_id"]
        cid = rec["cid"]
        sid = chunk_to_source.get(cid, "")
        if eid not in entities:
            entities[eid] = EntityNode(
                entity_id=eid,
                canonical_name=rec["canonical_name"],
                entity_type=rec["entity_type"],
            )
        entities[eid].chunk_ids.add(cid)
        entities[eid].source_ids.add(sid)
        connected_sources.add(sid)

    excluded = [sid for sid in source_ids if sid not in connected_sources]
    return entities, chunk_to_source, excluded


def _load_entity_embeddings(entity_ids: list[str]) -> dict[str, list[float]]:
    if not entity_ids:
        return {}
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, embedding FROM entities WHERE id = ANY(%s) AND embedding IS NOT NULL",
            (entity_ids,),
        ).fetchall()
    result: dict[str, list[float]] = {}
    for row in rows:
        emb = row[1]
        if emb is None:
            continue
        if isinstance(emb, str):
            emb = [float(x) for x in emb.strip("[]").split(",")]
        result[str(row[0])] = list(emb)
    return result


_MAX_COOC_ENTITIES_PER_SOURCE = 100
_MAX_ENTITIES_FOR_SEMANTIC = 500


def _build_igraph(
    entities: dict[str, EntityNode],
    chunk_to_source: dict[str, str],
    semantic_threshold: float,
    source_cooc_weight: float,
) -> igraph.Graph:
    entity_ids = list(entities.keys())
    idx = {eid: i for i, eid in enumerate(entity_ids)}
    g = igraph.Graph(n=len(entity_ids), directed=False)
    g.vs["entity_id"] = entity_ids

    chunk_entities: dict[str, set[str]] = defaultdict(set)
    for eid, node in entities.items():
        for cid in node.chunk_ids:
            chunk_entities[cid].add(eid)

    entity_degree = {eid: len(node.chunk_ids) for eid, node in entities.items()}
    edge_weights: dict[tuple[int, int], float] = {}

    for cid, chunk_eid_set in chunk_entities.items():
        chunk_eid_list = list(chunk_eid_set)
        for i in range(len(chunk_eid_list)):
            for j in range(i + 1, len(chunk_eid_list)):
                a, b = chunk_eid_list[i], chunk_eid_list[j]
                da, db = entity_degree[a], entity_degree[b]
                if da == 0 or db == 0:
                    continue
                base = 1.0 / math.sqrt(da * db)
                cross = 0 if entities[a].source_ids & entities[b].source_ids else 1
                w = base * (1 + 0.5 * cross)
                key = (min(idx[a], idx[b]), max(idx[a], idx[b]))
                edge_weights[key] = edge_weights.get(key, 0.0) + w

    source_entities: dict[str, set[str]] = defaultdict(set)
    for eid, node in entities.items():
        for sid in node.source_ids:
            source_entities[sid].add(eid)

    # Skip source co-occurrence for large sources — chunk co-occurrence edges already
    # connect most entity pairs; the weak fallback edge is only useful for small sources.
    for src_eids in source_entities.values():
        src_list = list(src_eids)
        if len(src_list) > _MAX_COOC_ENTITIES_PER_SOURCE:
            continue
        for i in range(len(src_list)):
            for j in range(i + 1, len(src_list)):
                a, b = src_list[i], src_list[j]
                key = (min(idx[a], idx[b]), max(idx[a], idx[b]))
                if key not in edge_weights:
                    edge_weights[key] = source_cooc_weight

    # Skip O(N²) semantic cross-source computation when entity count is too large.
    if len(entity_ids) <= _MAX_ENTITIES_FOR_SEMANTIC:
        source_entity_sets: dict[str, set[str]] = defaultdict(set)
        for eid, node in entities.items():
            for sid in node.source_ids:
                source_entity_sets[sid].add(eid)

        source_list = [(sid, list(eids)) for sid, eids in source_entity_sets.items()]
        embeddings: dict[str, list[float]] = {}
        for si in range(len(source_list)):
            for sj in range(si + 1, len(source_list)):
                _, eids_a = source_list[si]
                _, eids_b = source_list[sj]
                for a_id in eids_a:
                    for b_id in eids_b:
                        vi, vj = idx[a_id], idx[b_id]
                        key = (min(vi, vj), max(vi, vj))
                        if key in edge_weights:
                            continue
                        if not embeddings:
                            embeddings = _load_entity_embeddings(entity_ids)
                        emb_a = embeddings.get(a_id)
                        emb_b = embeddings.get(b_id)
                        if emb_a is None or emb_b is None:
                            continue
                        sim = _cosine_similarity(emb_a, emb_b)
                        if sim < semantic_threshold:
                            continue
                        edge_weights[key] = sim * 0.5

    if edge_weights:
        edges = list(edge_weights.keys())
        g.add_edges(edges)
        g.es["weight"] = [edge_weights[e] for e in edges]

    return g


def _run_leiden(g: igraph.Graph, min_community_size: int) -> list[list[str]]:
    if g.vcount() == 0:
        return []

    weights = "weight" if g.ecount() > 0 else None
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights=weights)

    return [
        [g.vs[i]["entity_id"] for i in membership]
        for membership in partition
        if len(membership) >= min_community_size
    ]


def _load_source_names(source_ids: list[str]) -> dict[str, str]:
    if not source_ids:
        return {}
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name FROM sources WHERE id = ANY(%s)",
            (source_ids,),
        ).fetchall()
    return {str(row[0]): (row[1] or "") for row in rows}


def _score_and_select_chunks(
    community_entity_ids: list[str],
    entities: dict[str, EntityNode],
    chunk_to_source: dict[str, str],
    source_names: dict[str, str],
    cutoff: float,
    top_k: int,
) -> list[ChunkResult]:
    entity_count = len(community_entity_ids)
    if entity_count == 0:
        return []

    chunk_entity_hits: dict[str, set[str]] = defaultdict(set)
    for eid in community_entity_ids:
        if eid in entities:
            for cid in entities[eid].chunk_ids:
                chunk_entity_hits[cid].add(eid)

    chunk_ids_needed = list(chunk_entity_hits.keys())
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, content FROM chunks WHERE id = ANY(%s) AND deleted_at IS NULL",
            (chunk_ids_needed,),
        ).fetchall()
    chunk_content: dict[str, str] = {str(row[0]): row[1] for row in rows}

    scored: list[ChunkResult] = []
    for cid, hit_entities in chunk_entity_hits.items():
        overlap = len(hit_entities)
        score = (overlap ** 2) / entity_count
        if score < cutoff:
            continue
        sid = chunk_to_source.get(cid, "")
        scored.append(ChunkResult(
            chunk_id=cid,
            source_id=sid,
            source_name=source_names.get(sid, ""),
            entity_overlap_count=overlap,
            score=score,
            content=chunk_content.get(cid, ""),
        ))

    unique_sources = {c.source_id for c in scored}
    if len(unique_sources) <= 1:
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    by_source: dict[str, list[ChunkResult]] = defaultdict(list)
    for c in scored:
        by_source[c.source_id].append(c)
    for src_list in by_source.values():
        src_list.sort(key=lambda c: c.score, reverse=True)

    source_order = sorted(by_source.keys(), key=lambda s: by_source[s][0].score, reverse=True)
    pointers = {s: 0 for s in source_order}
    result: list[ChunkResult] = []
    while len(result) < top_k:
        added_any = False
        for sid in source_order:
            p = pointers[sid]
            if p < len(by_source[sid]):
                result.append(by_source[sid][p])
                pointers[sid] += 1
                added_any = True
                if len(result) >= top_k:
                    break
        if not added_any:
            break
    return result


def _summarize_community(community: Community, model: str) -> str:
    if not settings.OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is required for summarization")

    chunk_texts = "\n\n---\n\n".join(
        f"[Source: {c.source_name}]\n{c.content}" for c in community.chunks
    )
    prompt = settings.COMMUNITY_SUMMARIZATION_PROMPT or (
        "Craft a compelling narrative summarizing these chunks of related information"
    )
    full_prompt = f"{prompt}\n\n<chunks>\n{chunk_texts}\n</chunks>"

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": [{"role": "user", "content": full_prompt}]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def detect_communities(
    *,
    scope_mode: str,
    source_ids: list[str],
    criteria: list[str],
    filters: dict[str, str],
    search_options: dict,
    retrieve_options: dict,
    semantic_threshold: Optional[float] = None,
    source_cooc_weight: Optional[float] = None,
    cutoff: Optional[float] = None,
    min_community_size: Optional[int] = None,
    top_k_chunks: Optional[int] = None,
    summarize_model: Optional[str] = None,
) -> dict:
    sem_threshold = semantic_threshold if semantic_threshold is not None else settings.COMMUNITY_SEMANTIC_THRESHOLD
    cooc_weight = source_cooc_weight if source_cooc_weight is not None else settings.COMMUNITY_SOURCE_COOC_WEIGHT
    _cutoff = cutoff if cutoff is not None else settings.COMMUNITY_CUTOFF
    min_size = min_community_size if min_community_size is not None else settings.COMMUNITY_MIN_COMMUNITY_SIZE
    top_k = top_k_chunks if top_k_chunks is not None else settings.COMMUNITY_TOP_K_CHUNKS

    resolved_ids = _resolve_scope(scope_mode, source_ids, criteria, filters, search_options, retrieve_options)
    entities, chunk_to_source, excluded_ids = _load_graph_data(resolved_ids)
    source_names = _load_source_names(resolved_ids)
    g = _build_igraph(entities, chunk_to_source, sem_threshold, cooc_weight)
    community_lists = _run_leiden(g, min_size)

    communities: list[Community] = []
    for i, eid_list in enumerate(community_lists):
        all_sources: set[str] = set()
        for eid in eid_list:
            if eid in entities:
                all_sources.update(entities[eid].source_ids)

        chunks = _score_and_select_chunks(eid_list, entities, chunk_to_source, source_names, _cutoff, top_k)
        community = Community(
            community_id=str(i),
            is_cross_source=len(all_sources) > 1,
            entity_count=len(eid_list),
            entities=[
                {
                    "entity_id": entities[eid].entity_id,
                    "canonical_name": entities[eid].canonical_name,
                    "entity_type": entities[eid].entity_type,
                }
                for eid in eid_list if eid in entities
            ],
            contributing_sources=[
                ContributingSource(source_id=sid, source_name=source_names.get(sid, ""))
                for sid in sorted(all_sources)
            ],
            chunks=chunks,
        )
        communities.append(community)

    communities.sort(key=lambda c: (not c.is_cross_source, -c.entity_count))

    if summarize_model:
        for community in communities:
            community.summary = _summarize_community(community, summarize_model)

    return {
        "metadata": {
            "scope_mode": scope_mode,
            "source_count": len(resolved_ids),
            "sources_excluded": [
                {"source_id": sid, "reason": "no entity connections"}
                for sid in excluded_ids
            ],
            "parameters": {
                "semantic_threshold": sem_threshold,
                "source_cooc_weight": cooc_weight,
                "cutoff": _cutoff,
                "min_community_size": min_size,
                "top_k_chunks": top_k,
            },
        },
        "communities": [
            {
                "community_id": c.community_id,
                "is_cross_source": c.is_cross_source,
                "entity_count": c.entity_count,
                "entities": c.entities,
                "contributing_sources": [
                    {"source_id": cs.source_id, "source_name": cs.source_name}
                    for cs in c.contributing_sources
                ],
                "chunks": [
                    {
                        "chunk_id": ch.chunk_id,
                        "source_id": ch.source_id,
                        "source_name": ch.source_name,
                        "entity_overlap_count": ch.entity_overlap_count,
                        "score": ch.score,
                        "content": ch.content,
                    }
                    for ch in c.chunks
                ],
                "summary": c.summary,
            }
            for c in communities
        ],
    }
