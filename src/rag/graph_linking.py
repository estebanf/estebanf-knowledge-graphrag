from rag.config import settings
from rag.embedding import get_embeddings


def embed_entity_names(names: list[str]) -> list[list[float]]:
    if not names:
        return []
    return get_embeddings(names)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_dedup_candidates(conn, source_id: str) -> list[tuple[str, str, float]]:
    """Return (source_entity_id, existing_entity_id, similarity) pairs above threshold."""
    source_rows = conn.execute(
        "SELECT id, canonical_name FROM entities WHERE source_id = %s",
        (source_id,),
    ).fetchall()
    if not source_rows:
        return []

    existing_rows = conn.execute(
        "SELECT id, canonical_name FROM entities WHERE source_id != %s AND source_id IS NOT NULL",
        (source_id,),
    ).fetchall()
    if not existing_rows:
        return []

    source_vecs = embed_entity_names([r[1] for r in source_rows])
    existing_vecs = embed_entity_names([r[1] for r in existing_rows])

    candidates = []
    for i, (src_id, _) in enumerate(source_rows):
        for j, (ex_id, _) in enumerate(existing_rows):
            sim = _cosine_sim(source_vecs[i], existing_vecs[j])
            if sim >= settings.ENTITY_DEDUP_COSINE_THRESHOLD:
                candidates.append((src_id, ex_id, sim))
    return candidates


def merge_entities(conn, driver, canonical_id: str, duplicate_id: str) -> None:
    """Re-link duplicate entity's Memgraph edges to canonical, then delete duplicate."""
    dup_row = conn.execute(
        "SELECT aliases FROM entities WHERE id = %s", (duplicate_id,)
    ).fetchone()
    canon_row = conn.execute(
        "SELECT aliases FROM entities WHERE id = %s", (canonical_id,)
    ).fetchone()
    if not dup_row or not canon_row:
        return

    merged_aliases = list(set((canon_row[0] or []) + (dup_row[0] or [])))
    conn.execute(
        "UPDATE entities SET aliases = %s WHERE id = %s",
        (merged_aliases, canonical_id),
    )
    conn.execute("DELETE FROM entities WHERE id = %s", (duplicate_id,))

    with driver.session() as session:
        session.run(
            "MATCH (dup:Entity {entity_id: $dup_id})-[r]->(n) "
            "MATCH (canon:Entity {entity_id: $canon_id}) "
            "MERGE (canon)-[r2:RELATED_TO {type: r.type, confidence: r.confidence, chunk_id: r.chunk_id}]->(n)",
            dup_id=duplicate_id,
            canon_id=canonical_id,
        )
        session.run(
            "MATCH (dup:Entity {entity_id: $dup_id}) DETACH DELETE dup",
            dup_id=duplicate_id,
        )


def create_mentioned_in_edges(conn, driver, source_id: str) -> None:
    """Create (Entity)-[:MENTIONED_IN]->(Chunk) edges for all chunks in source."""
    rows = conn.execute(
        """
        SELECT e.id, c.id
        FROM entities e
        JOIN chunks c ON c.source_id = e.source_id
        WHERE e.source_id = %s AND c.deleted_at IS NULL
        """,
        (source_id,),
    ).fetchall()

    with driver.session() as session:
        for entity_id, chunk_id in rows:
            session.run(
                "MATCH (e:Entity {entity_id: $entity_id}), (c:Chunk {chunk_id: $chunk_id}) "
                "MERGE (e)-[:MENTIONED_IN]->(c)",
                entity_id=str(entity_id),
                chunk_id=str(chunk_id),
            )


def link_graph(conn, driver, source_id: str, job_id: str) -> None:
    candidates = find_dedup_candidates(conn, source_id)
    for src_id, ex_id, _ in candidates:
        merge_entities(conn, driver, ex_id, src_id)
    create_mentioned_in_edges(conn, driver, source_id)
