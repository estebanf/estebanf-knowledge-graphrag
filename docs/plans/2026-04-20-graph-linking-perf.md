# Graph Linking Performance & Resilience

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate the OpenRouter timeout in `graph_linking` by persisting entity embeddings and pushing similarity computation to Postgres, and fix the N×M Memgraph write storm in `create_mentioned_in_edges`.

**Architecture:** Add an `embedding vector(4096)` column to `entities`; store it during `graph_extraction` so embeddings are never re-computed at link time. Replace the Python cosine similarity loop with a single pgvector `<=>` SQL query. Replace the cross-product Memgraph loop with a single UNWIND Cypher that traverses existing `MENTIONS` edges.

**Tech Stack:** Python, psycopg (postgres driver already in use), pgvector `<=>` cosine distance operator, Memgraph Cypher `UNWIND`.

---

### Task 1: Migration — add `embedding` column to `entities`

**Files:**
- Create: `scripts/migrate/005_entity_embeddings.sql`

**Step 1: Write the migration file**

```sql
ALTER TABLE entities ADD COLUMN IF NOT EXISTS embedding vector(4096);
```

**Step 2: Apply the migration**

```bash
docker compose exec -T postgres psql -U rag -d rag < scripts/migrate/005_entity_embeddings.sql
```

Expected: `ALTER TABLE`

**Step 3: Verify**

```bash
docker compose exec -T postgres psql -U rag -d rag -c "\d entities"
```

Expected: `embedding | vector(4096)` row appears in output.

**Step 4: Commit**

```bash
git add scripts/migrate/005_entity_embeddings.sql
git commit -m "feat: add embedding column to entities for dedup caching"
```

---

### Task 2: Store embeddings when entities are created (`graph_extraction.py`)

**Files:**
- Modify: `src/rag/graph_extraction.py`
- Test: `tests/test_graph_extraction.py`

**Background:** `store_entities_and_edges` currently inserts entities by name only. We need to batch-embed the entity names and store the vector in the same INSERT.

**Step 1: Write the failing test**

Add to `tests/test_graph_extraction.py`:

```python
def test_store_entities_and_edges_stores_embedding():
    conn = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=MagicMock())
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    entities = [{"canonical_name": "Acme", "entity_type": "ORGANIZATION", "aliases": []}]
    fake_vec = [0.1] * 4096

    with patch("rag.graph_extraction.get_embeddings", return_value=[fake_vec]) as mock_embed:
        from rag.graph_extraction import store_entities_and_edges
        store_entities_and_edges(conn, driver, "chunk-uuid", "source-uuid", entities, [])

    mock_embed.assert_called_once_with(["Acme"])
    insert_sql, insert_params = conn.execute.call_args_list[0][0]
    assert "embedding" in insert_sql
    assert f"[{','.join(str(v) for v in fake_vec)}]" in insert_params
```

**Step 2: Run to verify it fails**

```bash
pytest tests/test_graph_extraction.py::test_store_entities_and_edges_stores_embedding -v
```

Expected: FAIL — `get_embeddings` not called, `embedding` not in INSERT.

**Step 3: Implement**

At top of `src/rag/graph_extraction.py`, add import:
```python
from rag.embedding import get_embeddings
```

Replace `store_entities_and_edges` with:

```python
def store_entities_and_edges(
    conn,
    driver,
    chunk_id: str,
    source_id: str,
    entities: list[dict],
    relationships: list[dict],
) -> list[str]:
    entity_ids: list[str] = []
    name_to_id: dict[str, str] = {}

    if not entities:
        return entity_ids

    names = [e["canonical_name"] for e in entities]
    try:
        vecs = get_embeddings(names)
    except Exception:
        vecs = [None] * len(entities)

    with driver.session() as session:
        for entity, vec in zip(entities, vecs):
            entity_id = str(uuid.uuid4())
            entity_ids.append(entity_id)
            name_to_id[entity["canonical_name"]] = entity_id

            embedding_str = f"[{','.join(str(v) for v in vec)}]" if vec is not None else None

            conn.execute(
                """INSERT INTO entities (id, canonical_name, entity_type, aliases, source_id, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT DO NOTHING""",
                (
                    entity_id,
                    entity["canonical_name"],
                    entity["entity_type"],
                    entity.get("aliases", []),
                    source_id,
                    embedding_str,
                ),
            )

            session.run(
                "MERGE (e:Entity {entity_id: $entity_id}) "
                "SET e.canonical_name = $canonical_name, e.entity_type = $entity_type",
                entity_id=entity_id,
                canonical_name=entity["canonical_name"],
                entity_type=entity["entity_type"],
            )
            session.run(
                "MATCH (c:Chunk {chunk_id: $chunk_id}), (e:Entity {entity_id: $entity_id}) "
                "MERGE (c)-[:MENTIONS {confidence: $confidence}]->(e)",
                chunk_id=chunk_id,
                entity_id=entity_id,
                confidence=1.0,
            )

        for rel in relationships:
            e1_id = name_to_id.get(rel.get("source", ""))
            e2_id = name_to_id.get(rel.get("target", ""))
            if not e1_id or not e2_id:
                continue
            session.run(
                "MATCH (e1:Entity {entity_id: $e1_id}), (e2:Entity {entity_id: $e2_id}) "
                "MERGE (e1)-[:RELATED_TO {type: $type, confidence: $confidence, chunk_id: $chunk_id}]->(e2)",
                e1_id=e1_id,
                e2_id=e2_id,
                type=rel["type"],
                confidence=rel["confidence"],
                chunk_id=chunk_id,
            )

    return entity_ids
```

**Step 4: Run new test + full suite**

```bash
pytest tests/test_graph_extraction.py -v
```

Expected: all pass. If `test_store_entities_and_edges_inserts_to_postgres` fails because it doesn't mock `get_embeddings`, add `patch("rag.graph_extraction.get_embeddings", return_value=[[0.1]*4096])` to that test.

**Step 5: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: all pass.

**Step 6: Commit**

```bash
git add src/rag/graph_extraction.py tests/test_graph_extraction.py
git commit -m "feat: store entity embeddings at extraction time"
```

---

### Task 3: Fix `find_dedup_candidates` — push similarity to pgvector

**Files:**
- Modify: `src/rag/graph_linking.py`
- Test: `tests/test_graph_linking.py`

**Background:** Replace the Python cosine similarity loop (which required re-embedding ALL existing entities) with a single SQL query using pgvector's `<=>` cosine distance operator. Entities without embeddings (pre-migration rows) are skipped via `IS NOT NULL` filter. Remove `embed_entity_names`, `_cosine_sim`, and the `get_embeddings` import — they are no longer needed.

**Step 1: Write the failing test**

Replace `test_find_dedup_candidates_returns_high_similarity_pairs` and add a new one in `tests/test_graph_linking.py`:

```python
def test_find_dedup_candidates_uses_sql_similarity():
    """find_dedup_candidates must query DB for similarity, not call embedding API."""
    conn = MagicMock()
    # SQL returns (src_id, ex_id, similarity) rows already above threshold
    conn.execute.return_value.fetchall.return_value = [
        ("src-id-1", "ex-id-1", 0.97),
    ]

    from rag.graph_linking import find_dedup_candidates
    pairs = find_dedup_candidates(conn, "source-uuid")

    assert len(pairs) == 1
    assert pairs[0] == ("src-id-1", "ex-id-1", 0.97)
    # Must NOT call get_embeddings
    with patch("rag.graph_linking.get_embeddings") as mock_embed:
        find_dedup_candidates(conn, "source-uuid")
    mock_embed.assert_not_called()


def test_find_dedup_candidates_returns_empty_when_no_matches():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []

    from rag.graph_linking import find_dedup_candidates
    result = find_dedup_candidates(conn, "source-uuid")
    assert result == []
```

**Step 2: Run to verify they fail**

```bash
pytest tests/test_graph_linking.py::test_find_dedup_candidates_uses_sql_similarity -v
```

Expected: FAIL.

**Step 3: Implement**

Replace the entire contents of `src/rag/graph_linking.py` with:

```python
from rag.config import settings


def find_dedup_candidates(conn, source_id: str) -> list[tuple[str, str, float]]:
    """Return (source_entity_id, existing_entity_id, similarity) pairs above threshold."""
    rows = conn.execute(
        """
        SELECT src.id, ex.id, 1 - (src.embedding <=> ex.embedding) AS similarity
        FROM entities src
        JOIN entities ex
          ON ex.source_id != src.source_id
         AND ex.source_id IS NOT NULL
         AND ex.embedding IS NOT NULL
        WHERE src.source_id = %s
          AND src.embedding IS NOT NULL
          AND 1 - (src.embedding <=> ex.embedding) >= %s
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
```

**Step 4: Update the now-outdated tests**

In `tests/test_graph_linking.py`, remove or replace these tests that tested the old implementation:
- `test_embed_entity_names_delegates_to_get_embeddings` — remove (function deleted)
- `test_embed_entity_names_empty_returns_empty` — remove (function deleted)
- `test_find_dedup_candidates_returns_high_similarity_pairs` — remove (replaced by new test above)
- `test_find_dedup_candidates_no_existing_returns_empty` — remove (replaced by new test above)
- `test_create_mentioned_in_edges_creates_memgraph_edges` — update (see below)

Replace `test_create_mentioned_in_edges_creates_memgraph_edges` with:

```python
def test_create_mentioned_in_edges_uses_single_unwind_query():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("chunk-1",), ("chunk-2",),
    ]
    session_mock = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session_mock)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)

    from rag.graph_linking import create_mentioned_in_edges
    create_mentioned_in_edges(conn, driver, "source-uuid")

    # exactly one Cypher call with UNWIND
    assert session_mock.run.call_count == 1
    cypher, kwargs = session_mock.run.call_args[0][0], session_mock.run.call_args[1]
    assert "UNWIND" in cypher
    assert "MENTIONED_IN" in cypher
    assert kwargs["chunk_ids"] == ["chunk-1", "chunk-2"]


def test_create_mentioned_in_edges_no_chunks_skips_memgraph():
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    driver = MagicMock()

    from rag.graph_linking import create_mentioned_in_edges
    create_mentioned_in_edges(conn, driver, "source-uuid")

    driver.session.assert_not_called()
```

**Step 5: Run tests**

```bash
pytest tests/test_graph_linking.py -v
```

Expected: all pass.

**Step 6: Run full suite**

```bash
pytest tests/ -x -q
```

Expected: all pass.

**Step 7: Commit**

```bash
git add src/rag/graph_linking.py tests/test_graph_linking.py
git commit -m "perf: push entity dedup similarity to pgvector, batch MENTIONED_IN edges"
```

---

### Task 4: Update CLAUDE.md implementation status

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update the implementation status section**

In the "What's implemented" list under the graph pipeline bullet, append:

> Entity embeddings stored at extraction time; dedup uses pgvector `<=>` in SQL (no re-embedding at link time); `MENTIONED_IN` edges created via single `UNWIND` Cypher.

**Step 2: Add migration to the apply-migrations section**

Add to the migration list:
```bash
psql $POSTGRES_URL -f scripts/migrate/005_entity_embeddings.sql
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for graph linking perf improvements"
```
