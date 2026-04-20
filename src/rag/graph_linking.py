from rag.config import settings


def find_dedup_candidates(conn, source_id: str) -> list[tuple[str, str, float]]:
    """Return (source_entity_id, existing_entity_id, similarity) pairs above threshold."""
    rows = conn.execute(
        """
        SELECT src_id, ex_id, similarity FROM (
            SELECT src.id AS src_id, ex.id AS ex_id,
                   1 - (src.embedding <=> ex.embedding) AS similarity
            FROM entities src
            JOIN entities ex
              ON ex.source_id != src.source_id
             AND ex.source_id IS NOT NULL
             AND ex.embedding IS NOT NULL
            WHERE src.source_id = %s
              AND src.embedding IS NOT NULL
        ) sub
        WHERE similarity >= %s
        ORDER BY similarity DESC
        """,
        (source_id, settings.ENTITY_DEDUP_COSINE_THRESHOLD),
    ).fetchall()
    return [(str(r[0]), str(r[1]), float(r[2])) for r in rows]


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
    """Create (Entity)-[:MENTIONED_IN]->(Chunk) edges using existing MENTIONS edges."""
    chunk_ids = [
        str(r[0])
        for r in conn.execute(
            "SELECT id FROM chunks WHERE source_id = %s AND deleted_at IS NULL",
            (source_id,),
        ).fetchall()
    ]
    if not chunk_ids:
        return

    with driver.session() as session:
        session.run(
            "UNWIND $chunk_ids AS cid "
            "MATCH (c:Chunk {chunk_id: cid})-[:MENTIONS]->(e:Entity) "
            "MERGE (e)-[:MENTIONED_IN]->(c)",
            chunk_ids=chunk_ids,
        )


def link_graph(conn, driver, source_id: str, job_id: str) -> None:
    candidates = find_dedup_candidates(conn, source_id)
    for src_id, ex_id, _ in candidates:
        merge_entities(conn, driver, ex_id, src_id)
    create_mentioned_in_edges(conn, driver, source_id)
