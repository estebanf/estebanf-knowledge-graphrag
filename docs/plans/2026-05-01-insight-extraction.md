# Insight Extraction Feature Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `insight_extraction` pipeline stage that extracts, deduplicates, stores, embeds, and graph-links insights per chunk, plus two new `rag sources` subcommands and a backfill remediation script.

**Architecture:** A new `src/rag/insight_extraction.py` module handles the LLM call (OpenCode API, `deepseek-v4-flash`), pgvector-based deduplication, Postgres storage (`insights` + `chunk_insights` tables), and Memgraph node/edge creation (`Insight` nodes, `CONTAINS` edges from Chunk, `RELATED_TO` edges between insights via mutual top-K). The stage is appended as the 8th and final stage in `STAGE_ORDER`.

**Tech Stack:** Python, psycopg2/psycopg3, pgvector (cosine `<=>` operator), Memgraph (Cypher), httpx, Typer, Pydantic BaseSettings

---

## Context

The ingestion pipeline currently extracts entities and relationships but no higher-level insights per chunk. This feature adds:
1. LLM-driven insight extraction per chunk using the OpenCode API
2. Deduplication so semantically identical insights across chunks share one DB/graph node
3. Mutual top-K `RELATED_TO` edges between insight nodes to enable insight-level graph traversal
4. Two new CLI operations: `rag sources insights <source-id>` and `rag sources last <k>`
5. A remediation script to backfill insights for already-ingested sources

---

## Critical Files

| Role | Path |
|---|---|
| New module (core) | `src/rag/insight_extraction.py` |
| Pipeline orchestration | `src/rag/ingestion.py` |
| CLI | `src/rag/cli.py` |
| Config | `src/rag/config.py` |
| Prompts | `src/rag/prompts/__init__.py` |
| DB migration | `scripts/migrate/007_insights.sql` |
| Remediation script | `scripts/remediate_insights.py` |
| Memgraph init | `scripts/init/memgraph/` (check if exists, else add constraint via migration) |
| Tests | `tests/test_insight_extraction.py` (new), `tests/test_config.py`, `tests/test_prompts.py`, `tests/test_cli_sources.py` |
| Docs | `README.md`, `AGENTS.md` |

---

## Task 1: Database Migration

**Files:**
- Create: `scripts/migrate/007_insights.sql`

**Step 1:** Write migration

```sql
BEGIN;

CREATE TABLE IF NOT EXISTS insights (
  id          uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  content     text NOT NULL,
  embedding   vector(4096),
  created_at  timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunk_insights (
  chunk_id    uuid NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  insight_id  uuid NOT NULL REFERENCES insights(id) ON DELETE CASCADE,
  topics      text[] NOT NULL DEFAULT '{}',
  PRIMARY KEY (chunk_id, insight_id)
);

CREATE INDEX IF NOT EXISTS insights_embedding_hnsw_idx
  ON insights USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunk_insights_chunk_id_idx ON chunk_insights(chunk_id);
CREATE INDEX IF NOT EXISTS chunk_insights_insight_id_idx ON chunk_insights(insight_id);

COMMIT;
```

**Step 2:** Apply migration

```bash
docker compose exec -T postgres psql -U rag -d rag -f scripts/migrate/007_insights.sql
```

Expected: `CREATE TABLE`, `CREATE INDEX` (no errors)

**Step 3:** Verify

```bash
docker compose exec -T postgres psql -U rag -d rag -c "\d insights" -c "\d chunk_insights"
```

**Step 4:** Commit

```bash
git add scripts/migrate/007_insights.sql
git commit -m "feat: add insights and chunk_insights tables with HNSW index"
```

---

## Task 2: Config Additions

**Files:**
- Modify: `src/rag/config.py`

**Step 1:** Write failing test in `tests/test_config.py`

```python
def test_insight_config_defaults():
    from rag.config import Settings
    s = Settings(_env_file=None)
    assert s.OPENCODE_API_KEY == ""
    assert s.INSIGHT_DEDUP_COSINE_THRESHOLD == 0.95
    assert s.INSIGHT_LINK_TOP_K == 10
```

**Step 2:** Run test to confirm it fails

```bash
pytest tests/test_config.py::test_insight_config_defaults -v
```

Expected: `FAIL` — `Settings` has no `OPENCODE_API_KEY`

**Step 3:** Add to `src/rag/config.py` in the insight extraction section (after `ENTITY_DEDUP_COSINE_THRESHOLD`):

```python
# Insight extraction
OPENCODE_API_KEY: str = ""
INSIGHT_DEDUP_COSINE_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = 0.95
INSIGHT_LINK_TOP_K: Annotated[int, Field(gt=0)] = 10
```

**Step 4:** Run test to confirm it passes

```bash
pytest tests/test_config.py::test_insight_config_defaults -v
```

Expected: `PASS`

**Step 5:** Add vars to `.env.example` in the insight extraction section:

```
OPENCODE_API_KEY=
INSIGHT_DEDUP_COSINE_THRESHOLD=0.95
INSIGHT_LINK_TOP_K=10
```

**Step 6:** Commit

```bash
git add src/rag/config.py .env.example tests/test_config.py
git commit -m "feat: add insight extraction config vars"
```

---

## Task 3: Insight Extraction Prompt

**Files:**
- Modify: `src/rag/prompts/__init__.py`

**Step 1:** Write failing test in `tests/test_prompts.py`

```python
def test_insight_extraction_prompt_placeholders():
    from rag.prompts import INSIGHT_EXTRACTION
    assert "{chunk}" in INSIGHT_EXTRACTION
    assert "insights" in INSIGHT_EXTRACTION
    assert "topics" in INSIGHT_EXTRACTION.lower()
```

**Step 2:** Run test to confirm it fails

```bash
pytest tests/test_prompts.py::test_insight_extraction_prompt_placeholders -v
```

Expected: `FAIL`

**Step 3:** Add `INSIGHT_EXTRACTION` constant to `src/rag/prompts/__init__.py`

Add the full prompt exactly as specified by the user, with `{chunk}` as the only format placeholder. The topics table is hardcoded in the prompt text. The closing instruction must include the JSON schema: `{"insights": [{"insight":"","topics":[""]}]}`

```python
INSIGHT_EXTRACTION = """\
From the chunk below, extract insights that are specific, meaningful, and useful. Prioritize information that reveals causes, consequences, patterns, tradeoffs, risks, opportunities, decisions, assumptions, changes, tensions, or implications. Exclude generic summaries, obvious statements, repeated points, and details that do not affect understanding or action.

Focus only on insights that are related to one or more of these topics:
AI Adoption, AI Opportunity Strategy, AI Governance, AI Readiness, AI Use Case Prioritization, Workflow Automation, Workflow Intelligence, Human-in-the-Loop Design, Productized Services, Offer Design, Solution Architecture, Commercial Solutioning, Delivery Governance, Operating Models, B2B Transformation, Vertical Solutions, Cross-Functional Alignment, Technical Leadership, Executive Communication, Business Outcomes, Product Management, Product Strategy, Product Discovery, Product Roadmapping, Product Prioritization, Product Requirements, Product-Market Fit, Product Positioning, Product Packaging, Productized Services, Platform Strategy, Workflow Product Design, AI Product Management, AI Feature Strategy, Customer Research, Voice of Customer, Jobs-to-Be-Done, MVP Definition, Experimentation, Product Analytics, Go-to-Market Alignment, Buyer Enablement, Sales Enablement, Customer Onboarding, Customer Success Strategy, Product Operations, Stakeholder Alignment, Delivery Readiness, Acceptance Criteria, Product Governance, Outcome Management

Return your findings in a JSON document that follows this structure:
{{"insights": [{{"insight":"","topics":[""]}}]}}

Return {{"insights": []}} if no relevant insights are found.

Chunk
```
{chunk}
```"""
```

**Step 4:** Run test to confirm it passes

```bash
pytest tests/test_prompts.py::test_insight_extraction_prompt_placeholders -v
```

Expected: `PASS`

**Step 5:** Commit

```bash
git add src/rag/prompts/__init__.py tests/test_prompts.py
git commit -m "feat: add INSIGHT_EXTRACTION prompt constant"
```

---

## Task 4: New Module `src/rag/insight_extraction.py`

**Files:**
- Create: `src/rag/insight_extraction.py`
- Create: `tests/test_insight_extraction.py`

This task has multiple sub-steps. Write tests first, then implement.

### 4a: OpenCode API call

**Step 1:** Write failing test

```python
# tests/test_insight_extraction.py
import json
from unittest.mock import MagicMock, patch

def test_extract_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "")
    from rag.insight_extraction import extract_insights_from_chunk
    assert extract_insights_from_chunk("some text") == []

def test_extract_returns_empty_on_api_error(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "test-key")
    with patch("rag.insight_extraction.httpx.post", side_effect=Exception("connection error")):
        from rag.insight_extraction import extract_insights_from_chunk
        assert extract_insights_from_chunk("some text") == []

def test_extract_parses_valid_response(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.settings.OPENCODE_API_KEY", "test-key")
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": '{"insights": [{"insight": "AI reduces costs", "topics": ["AI Adoption"]}]}'}}]
    }
    mock_resp.raise_for_status = MagicMock()
    with patch("rag.insight_extraction.httpx.post", return_value=mock_resp):
        from rag.insight_extraction import extract_insights_from_chunk
        result = extract_insights_from_chunk("AI reduces operational costs significantly.")
    assert len(result) == 1
    assert result[0]["insight"] == "AI reduces costs"
    assert result[0]["topics"] == ["AI Adoption"]
```

**Step 2:** Run to confirm fails

```bash
pytest tests/test_insight_extraction.py -k "api" -v
```

**Step 3:** Implement `extract_insights_from_chunk` in `src/rag/insight_extraction.py`

```python
import json
import re
import logging
import httpx
from rag.config import settings
from rag import prompts

log = logging.getLogger(__name__)

_OPENCODE_URL = "https://opencode.ai/zen/go/v1/chat/completions"
_MODEL = "deepseek-v4-flash"
_MAX_CHUNK_CHARS = 4000


def _strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def extract_insights_from_chunk(content: str) -> list[dict]:
    if not settings.OPENCODE_API_KEY:
        return []
    try:
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
    except Exception as exc:
        log.warning("insight_extraction_failed", error=str(exc))
        return []
```

**Step 4:** Run tests to confirm they pass

```bash
pytest tests/test_insight_extraction.py -k "api or extract" -v
```

### 4b: Deduplication — `upsert_insight`

**Step 1:** Write failing tests

```python
def test_upsert_insight_reuses_existing():
    from rag.insight_extraction import upsert_insight
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = ("existing-uuid", 0.97)
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    insight_id, is_new = upsert_insight(conn, "some insight", [0.1] * 4096)
    assert insight_id == "existing-uuid"
    assert is_new is False

def test_upsert_insight_creates_new_when_below_threshold():
    from rag.insight_extraction import upsert_insight
    conn = MagicMock()
    cursor = MagicMock()
    # First fetchone: low-similarity existing; second fetchone: the RETURNING id
    cursor.fetchone.side_effect = [("existing-uuid", 0.80), ("new-uuid",)]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    insight_id, is_new = upsert_insight(conn, "different insight", [0.9] * 4096)
    assert is_new is True
```

**Step 2:** Run to confirm fails

```bash
pytest tests/test_insight_extraction.py -k "upsert" -v
```

**Step 3:** Implement `upsert_insight`

```python
def upsert_insight(conn, content: str, embedding: list[float]) -> tuple[str, bool]:
    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) AS sim
            FROM insights
            ORDER BY embedding <=> %s::vector
            LIMIT 1
            """,
            (emb_str, emb_str),
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
```

**Step 4:** Run tests to confirm they pass

```bash
pytest tests/test_insight_extraction.py -k "upsert" -v
```

### 4c: `link_chunk_insight`

**Step 1:** Write failing test

```python
def test_link_chunk_insight_executes_upsert_sql():
    from rag.insight_extraction import link_chunk_insight
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    link_chunk_insight(conn, "chunk-id", "insight-id", ["AI Adoption"])
    sql_called = cursor.execute.call_args[0][0]
    assert "ON CONFLICT" in sql_called
    assert "chunk_insights" in sql_called
```

**Step 2:** Run to confirm fails. Implement:

```python
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
```

**Step 3:** Run tests to confirm pass

```bash
pytest tests/test_insight_extraction.py -k "link_chunk" -v
```

### 4d: `store_insight_in_graph`

**Step 1:** Write failing test

```python
def test_store_insight_in_graph_merges_node_and_edge():
    from rag.insight_extraction import store_insight_in_graph
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    store_insight_in_graph(driver, "chunk-1", "insight-1", "AI reduces costs", ["AI Adoption"])
    assert session.run.call_count == 2
    first_call_cypher = session.run.call_args_list[0][0][0]
    assert "MERGE" in first_call_cypher and "Insight" in first_call_cypher
    second_call_cypher = session.run.call_args_list[1][0][0]
    assert "CONTAINS" in second_call_cypher
```

**Step 2:** Run to confirm fails. Implement:

```python
def store_insight_in_graph(
    driver, chunk_id: str, insight_id: str, content: str, topics: list[str]
) -> None:
    with driver.session() as session:
        session.run(
            "MERGE (i:Insight {insight_id: $insight_id}) SET i.content = $content",
            insight_id=insight_id,
            content=content,
        )
        session.run(
            """
            MATCH (c:Chunk {chunk_id: $chunk_id}), (i:Insight {insight_id: $insight_id})
            MERGE (c)-[:CONTAINS {topics: $topics}]->(i)
            """,
            chunk_id=chunk_id,
            insight_id=insight_id,
            topics=topics,
        )
```

**Step 3:** Run tests to confirm pass

```bash
pytest tests/test_insight_extraction.py -k "store_insight" -v
```

### 4e: `link_related_insights` (mutual top-K)

**Step 1:** Write failing tests

```python
def test_link_related_insights_creates_mutual_edges():
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    cursor = MagicMock()
    b_emb = [0.2] * 4096
    cursor.fetchall.side_effect = [
        [("b-id", 0.88, b_emb)],            # A's neighbors
        [("a-id", 0.88, [0.1]*4096)],       # B's neighbors — includes A, so mutual
    ]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    link_related_insights(conn, driver, "a-id", [0.1] * 4096)
    assert session.run.call_count == 1
    cypher = session.run.call_args[0][0]
    assert "RELATED_TO" in cypher

def test_link_related_insights_skips_non_mutual():
    from rag.insight_extraction import link_related_insights
    conn = MagicMock()
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__ = lambda s: session
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    cursor = MagicMock()
    cursor.fetchall.side_effect = [
        [("b-id", 0.88, [0.2]*4096)],   # A's neighbors: B
        [("c-id", 0.90, [0.3]*4096)],   # B's neighbors: C (not A) — not mutual
    ]
    conn.cursor.return_value.__enter__ = lambda s: cursor
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    link_related_insights(conn, driver, "a-id", [0.1] * 4096)
    session.run.assert_not_called()
```

**Step 2:** Run to confirm fails. Implement `link_related_insights`:

```python
def link_related_insights(
    conn, driver, insight_id: str, embedding: list[float]
) -> None:
    k = settings.INSIGHT_LINK_TOP_K
    emb_str = "[" + ",".join(str(v) for v in embedding) + "]"

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, 1 - (embedding <=> %s::vector) AS sim, embedding
            FROM insights
            WHERE id != %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (emb_str, insight_id, emb_str, k),
        )
        a_neighbors = cur.fetchall()  # list of (b_id, sim, b_emb)

        for b_id, sim, b_emb_raw in a_neighbors:
            b_emb_str = "[" + ",".join(str(v) for v in b_emb_raw) + "]"
            cur.execute(
                """
                SELECT id FROM insights
                WHERE id != %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (b_id, b_emb_str, k),
            )
            b_neighbor_ids = {str(r[0]) for r in cur.fetchall()}

            if insight_id in b_neighbor_ids:
                with driver.session() as session:
                    session.run(
                        """
                        MATCH (a:Insight {insight_id: $a_id}), (b:Insight {insight_id: $b_id})
                        MERGE (a)-[:RELATED_TO {similarity: $sim}]->(b)
                        MERGE (b)-[:RELATED_TO {similarity: $sim}]->(a)
                        """,
                        a_id=insight_id,
                        b_id=str(b_id),
                        sim=float(sim),
                    )
```

**Step 3:** Run tests to confirm pass

```bash
pytest tests/test_insight_extraction.py -k "related" -v
```

### 4f: Orchestrator `extract_and_store_insights`

**Step 1:** Write failing test

```python
def test_extract_and_store_insights_returns_counts(monkeypatch):
    monkeypatch.setattr("rag.insight_extraction.extract_insights_from_chunk",
                        lambda content: [{"insight": "insight A", "topics": ["AI Adoption"]}])
    monkeypatch.setattr("rag.insight_extraction.upsert_insight",
                        lambda conn, content, emb: ("new-uuid", True))
    monkeypatch.setattr("rag.insight_extraction.link_chunk_insight", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.store_insight_in_graph", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.link_related_insights", lambda *a, **k: None)
    monkeypatch.setattr("rag.insight_extraction.get_embeddings", lambda texts: [[0.1]*4096])

    from rag.insight_extraction import extract_and_store_insights
    conn = MagicMock()
    driver = MagicMock()
    result = extract_and_store_insights(
        conn, driver, "src-id", [("chunk-1", "some content")]
    )
    assert result["chunks_processed"] == 1
    assert result["insights_extracted"] == 1
    assert result["insights_reused"] == 0
```

**Step 2:** Run to confirm fails. Implement:

```python
from rag.embedding import get_embeddings

def extract_and_store_insights(
    conn,
    driver,
    source_id: str,
    chunk_rows: list[tuple[str, str]],
) -> dict:
    chunks_processed = 0
    insights_extracted = 0
    insights_reused = 0

    for chunk_id, content in chunk_rows:
        raw_insights = extract_insights_from_chunk(content)
        if not raw_insights:
            chunks_processed += 1
            conn.commit()
            continue

        texts = [r["insight"] for r in raw_insights]
        embeddings = get_embeddings(texts)

        for raw, emb in zip(raw_insights, embeddings):
            insight_id, is_new = upsert_insight(conn, raw["insight"], emb)
            topics = raw.get("topics", [])
            link_chunk_insight(conn, chunk_id, insight_id, topics)
            store_insight_in_graph(driver, chunk_id, insight_id, raw["insight"], topics)
            if is_new:
                link_related_insights(conn, driver, insight_id, emb)
                insights_extracted += 1
            else:
                insights_reused += 1

        chunks_processed += 1
        conn.commit()

    return {
        "chunks_processed": chunks_processed,
        "insights_extracted": insights_extracted,
        "insights_reused": insights_reused,
    }
```

**Step 3:** Run all insight extraction tests

```bash
pytest tests/test_insight_extraction.py -v
```

Expected: all pass

**Step 4:** Commit

```bash
git add src/rag/insight_extraction.py tests/test_insight_extraction.py
git commit -m "feat: add insight_extraction module with dedup, graph linking, and mutual top-K edges"
```

---

## Task 5: Pipeline Integration

**Files:**
- Modify: `src/rag/ingestion.py`
- Modify: `src/rag/cli.py`

### 5a: Update STAGE_ORDER

**Step 1:** In `src/rag/ingestion.py`, change `STAGE_ORDER`:

```python
STAGE_ORDER = [
    "parsing", "profiling", "chunking", "validation",
    "embedding", "graph_extraction", "graph_linking", "insight_extraction",
]
```

**Step 2:** In `src/rag/cli.py`, find the duplicate `STAGE_ORDER` tuple (used for retry stage validation) and update it to include `"insight_extraction"`.

### 5b: Add insight_extraction stage to `execute_ingestion_pipeline`

**Step 1:** Write failing test

> Note: Look at existing tests in `tests/test_job_lifecycle.py` to understand the fixture pattern before writing this test. The test should mock `extract_and_store_insights` and verify it is called when starting the pipeline from the `"insight_extraction"` stage.

**Step 2:** Add stage block inside `execute_ingestion_pipeline()` in `src/rag/ingestion.py`, after the `graph_linking` block:

```python
# --- Insight Extraction ---
if start_idx <= STAGE_ORDER.index("insight_extraction"):
    _update_stage(conn, job_id, "insight_extraction")
    try:
        from rag.insight_extraction import extract_and_store_insights
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content FROM chunks WHERE job_id = %s AND deleted_at IS NULL ORDER BY chunk_index",
                (job_id,),
            )
            chunk_rows = [(str(r[0]), r[1]) for r in cur.fetchall()]
        result = extract_and_store_insights(conn, driver, source_id, chunk_rows)
    except Exception as exc:
        _fail_stage(conn, job_id, "insight_extraction", exc)
        raise
    _complete_stage(conn, job_id, "insight_extraction", result)
```

### 5c: Add cleanup logic in `cleanup_from_stage()`

In `src/rag/ingestion.py`, in `cleanup_from_stage()`, add handling for restarting from `"insight_extraction"`:

```python
if idx <= STAGE_ORDER.index("insight_extraction"):
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM chunk_insights WHERE chunk_id IN (SELECT id FROM chunks WHERE job_id = %s)",
            (job_id,),
        )
        cur.execute(
            "DELETE FROM insights WHERE id NOT IN (SELECT DISTINCT insight_id FROM chunk_insights)"
        )
    if driver:
        with driver.session() as session:
            session.run(
                "MATCH (i:Insight) WHERE NOT (i)<-[:CONTAINS]-() DETACH DELETE i"
            )
```

### 5d: Update hard-delete / `delete_source_artifacts()`

In `src/rag/ingestion.py`, before deleting chunks for a source, add:

```python
with conn.cursor() as cur:
    cur.execute(
        "DELETE FROM chunk_insights WHERE chunk_id IN (SELECT id FROM chunks WHERE source_id = %s)",
        (source_id,),
    )
    cur.execute(
        "DELETE FROM insights WHERE id NOT IN (SELECT DISTINCT insight_id FROM chunk_insights)"
    )
```

And in the Memgraph cleanup section:
```cypher
MATCH (i:Insight) WHERE NOT (i)<-[:CONTAINS]-() DETACH DELETE i
```

**Step 5:** Run ingestion tests

```bash
pytest -q tests/test_job_lifecycle.py tests/test_ingestion_submit.py tests/test_worker.py
```

Expected: all pass

**Step 6:** Commit

```bash
git add src/rag/ingestion.py src/rag/cli.py
git commit -m "feat: add insight_extraction as 8th pipeline stage with cleanup"
```

---

## Task 6: CLI — `rag sources insights`

**Files:**
- Modify: `src/rag/cli.py`
- Modify: `tests/test_cli_sources.py`

**Step 1:** Write failing test

```python
def test_sources_insights_prints_table():
    from typer.testing import CliRunner
    from rag.cli import app
    from unittest.mock import patch, MagicMock
    import uuid
    runner = CliRunner()
    fake_rows = [
        (uuid.uuid4(), "AI reduces operational costs by 30%", ["AI Adoption", "Business Outcomes"], "2026-01-10 12:00:00"),
    ]
    with patch("rag.cli._get_connection") as mock_conn:
        cur = MagicMock()
        cur.fetchall.return_value = fake_rows
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cur
        result = runner.invoke(app, ["sources", "insights", str(uuid.uuid4())])
    assert result.exit_code == 0
    assert "AI reduces" in result.output

def test_sources_insights_empty():
    from typer.testing import CliRunner
    from rag.cli import app
    from unittest.mock import patch, MagicMock
    runner = CliRunner()
    with patch("rag.cli._get_connection") as mock_conn:
        cur = MagicMock()
        cur.fetchall.return_value = []
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cur
        result = runner.invoke(app, ["sources", "insights", "some-id"])
    assert "No insights" in result.output
```

**Step 2:** Run to confirm fails

```bash
pytest tests/test_cli_sources.py -k "insights" -v
```

**Step 3:** Implement `sources insights` command in `src/rag/cli.py`:

```python
@sources_app.command("insights")
def sources_insights(
    source_id: Annotated[str, typer.Argument(help="Source UUID")],
) -> None:
    """List all insights extracted from chunks of a source."""
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT i.id, i.content, ci.topics, i.created_at
                FROM chunks c
                JOIN chunk_insights ci ON ci.chunk_id = c.id
                JOIN insights i ON i.id = ci.insight_id
                WHERE c.source_id = %s AND c.deleted_at IS NULL
                ORDER BY i.created_at
                """,
                (source_id,),
            )
            rows = cur.fetchall()

    if not rows:
        console.print("[dim]No insights found for this source.[/dim]")
        return

    table = Table(title=f"Insights for {source_id}")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Content")
    table.add_column("Topics")
    table.add_column("Created")
    for r in rows:
        table.add_row(
            str(r[0]),
            r[1][:120] + ("…" if len(r[1]) > 120 else ""),
            ", ".join(r[2]) if r[2] else "",
            str(r[3])[:19],
        )
    console.print(table)
```

**Step 4:** Run tests to confirm pass

```bash
pytest tests/test_cli_sources.py -k "insights" -v
```

**Step 5:** Commit

```bash
git add src/rag/cli.py tests/test_cli_sources.py
git commit -m "feat: add 'rag sources insights' subcommand"
```

---

## Task 7: CLI — `rag sources last`

**Files:**
- Modify: `src/rag/cli.py`
- Modify: `tests/test_cli_sources.py`

**Step 1:** Write failing tests

```python
def test_sources_last_with_integer():
    from typer.testing import CliRunner
    from rag.cli import app
    from unittest.mock import patch, MagicMock
    import uuid
    runner = CliRunner()
    fake_rows = [(uuid.uuid4(),), (uuid.uuid4(),)]
    with patch("rag.cli._get_connection") as mock_conn:
        cur = MagicMock()
        cur.fetchall.return_value = fake_rows
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cur
        result = runner.invoke(app, ["sources", "last", "5"])
    assert result.exit_code == 0
    assert "LIMIT" in str(cur.execute.call_args)

def test_sources_last_with_date_string():
    from typer.testing import CliRunner
    from rag.cli import app
    from unittest.mock import patch, MagicMock
    runner = CliRunner()
    with patch("rag.cli._get_connection") as mock_conn:
        cur = MagicMock()
        cur.fetchall.return_value = []
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cur
        result = runner.invoke(app, ["sources", "last", "2026-01-01"])
    assert result.exit_code == 0
    assert "created_at >=" in str(cur.execute.call_args)
```

**Step 2:** Run to confirm fails

```bash
pytest tests/test_cli_sources.py -k "last" -v
```

**Step 3:** Implement `sources last`:

```python
@sources_app.command("last")
def sources_last(
    k: Annotated[str, typer.Argument(help="Integer (last N sources) or date string (sources since YYYY-MM-DD)")],
) -> None:
    """Return source IDs: last K sources (integer) or sources since a date (date string)."""
    with _get_connection() as conn:
        with conn.cursor() as cur:
            try:
                n = int(k)
                cur.execute(
                    "SELECT id FROM sources WHERE deleted_at IS NULL ORDER BY created_at DESC LIMIT %s",
                    (n,),
                )
            except ValueError:
                cur.execute(
                    "SELECT id FROM sources WHERE deleted_at IS NULL AND created_at >= %s::timestamptz ORDER BY created_at DESC",
                    (k,),
                )
            rows = cur.fetchall()

    if not rows:
        console.print("[dim]No sources found.[/dim]")
        return

    for r in rows:
        console.print(str(r[0]))
```

**Step 4:** Run tests to confirm pass

```bash
pytest tests/test_cli_sources.py -k "last" -v
```

**Step 5:** Commit

```bash
git add src/rag/cli.py tests/test_cli_sources.py
git commit -m "feat: add 'rag sources last' subcommand"
```

---

## Task 8: Remediation Script

**Files:**
- Create: `scripts/remediate_insights.py`

**Step 1:** Check the Memgraph driver import pattern used in `src/rag/ingestion.py` and use the same pattern in the script.

**Step 2:** Write the script

```python
#!/usr/bin/env python3
"""
Backfill insight extraction for sources that have chunks but no chunk_insights rows.
Idempotent: sources with any existing chunk_insight are skipped automatically.

Usage:
    python scripts/remediate_insights.py [--batch-size N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psycopg2
# Use same driver import as ingestion.py — verify before writing
from rag.config import settings
from rag.insight_extraction import extract_and_store_insights

_PENDING_SQL = """
SELECT DISTINCT c.source_id
FROM chunks c
WHERE c.deleted_at IS NULL
  AND NOT EXISTS (
    SELECT 1 FROM chunk_insights ci WHERE ci.chunk_id = c.id
  )
ORDER BY c.source_id
"""

_CHUNK_ROWS_SQL = """
SELECT id::text, content FROM chunks
WHERE source_id = %s AND deleted_at IS NULL
ORDER BY chunk_index
"""


def main():
    parser = argparse.ArgumentParser(description="Backfill insights for already-ingested sources.")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of sources to process per batch.")
    args = parser.parse_args()

    conn = psycopg2.connect(settings.POSTGRES_URL)
    # Use same driver construction as ingestion.py
    driver = ...  # fill in from ingestion.py pattern

    try:
        with conn.cursor() as cur:
            cur.execute(_PENDING_SQL)
            pending = [str(r[0]) for r in cur.fetchall()]

        total = len(pending)
        print(f"Found {total} sources pending insight extraction.")

        if total == 0:
            print("Nothing to do.")
            return

        processed = 0
        errors = 0

        for i in range(0, total, args.batch_size):
            batch = pending[i:i + args.batch_size]
            print(f"\nBatch {i // args.batch_size + 1}: sources {i+1}–{min(i+args.batch_size, total)} of {total}")

            for source_id in batch:
                try:
                    with conn.cursor() as cur:
                        cur.execute(_CHUNK_ROWS_SQL, (source_id,))
                        chunk_rows = [(r[0], r[1]) for r in cur.fetchall()]

                    if not chunk_rows:
                        print(f"  [SKIP] {source_id} — no chunks")
                        continue

                    result = extract_and_store_insights(conn, driver, source_id, chunk_rows)
                    print(
                        f"  [OK] {source_id} — "
                        f"{result['chunks_processed']} chunks, "
                        f"{result['insights_extracted']} new, "
                        f"{result['insights_reused']} reused"
                    )
                    processed += 1
                except Exception as exc:
                    print(f"  [ERROR] {source_id} — {exc}")
                    conn.rollback()
                    errors += 1

        print(f"\nDone. Processed: {processed}, Errors: {errors}")
    finally:
        conn.close()
        driver.close()


if __name__ == "__main__":
    main()
```

**Step 3:** Make executable

```bash
chmod +x scripts/remediate_insights.py
```

**Step 4:** Smoke-test

```bash
python scripts/remediate_insights.py --batch-size 5
```

Expected: `Found 0 sources pending insight extraction. Nothing to do.` (on a clean DB)

**Step 5:** Commit

```bash
git add scripts/remediate_insights.py
git commit -m "feat: add remediate_insights.py backfill script"
```

---

## Task 9: Documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

**Step 1:** Update `README.md`

1. In "Ingestion" stages list, add `8. insight_extraction`
2. In "Environment" section, add **"Insight extraction"** subsection:

| Variable | Default | Purpose |
|---|---|---|
| `OPENCODE_API_KEY` | empty | API key for OpenCode service used in insight extraction. |
| `INSIGHT_DEDUP_COSINE_THRESHOLD` | `0.95` | Minimum cosine similarity to reuse an existing insight instead of creating a new one. |
| `INSIGHT_LINK_TOP_K` | `10` | Number of nearest insight neighbors used for mutual top-K `RELATED_TO` edge creation. |

3. In "Source Operations" section, add:

```bash
# List all insights for a source
venv/bin/rag sources insights <source_id>

# Return last N source IDs (integer) or source IDs since a date
venv/bin/rag sources last 5
venv/bin/rag sources last 2026-01-01
```

4. Add new **"Remediation"** section:

```bash
python scripts/remediate_insights.py --batch-size 10
```

5. In "Migrations and Existing Databases", add `scripts/migrate/007_insights.sql`

**Step 2:** Update `AGENTS.md`

1. In "Code Organization", add:
   - `src/rag/insight_extraction.py`: OpenCode API call, per-chunk insight extraction, pgvector dedup against `insights` table, Memgraph `Insight` node and `CONTAINS`/`RELATED_TO` edge management.

2. In "Current Behavioral Notes", add:
   - Insight extraction uses the OpenCode API (`deepseek-v4-flash`) per chunk. Dedup uses pgvector `<=>` cosine distance; threshold is `INSIGHT_DEDUP_COSINE_THRESHOLD`.
   - Mutual top-K for `RELATED_TO` edges is computed entirely in Postgres via pgvector; Memgraph stores the resulting edges only.
   - `scripts/remediate_insights.py` backfills insights for existing sources; it skips sources that already have `chunk_insights` entries.

3. In "Verification Shortcuts", add:
   - insight extraction: `pytest -q tests/test_insight_extraction.py tests/test_config.py tests/test_prompts.py tests/test_cli_sources.py`

4. In "Data and Schema Notes", add:
   - `insights` and `chunk_insights` tables added in `scripts/migrate/007_insights.sql`. Apply on any DB predating this feature.

**Step 3:** Commit

```bash
git add README.md AGENTS.md
git commit -m "docs: document insight extraction stage, CLI commands, and remediation script"
```

---

## Verification

### Unit tests

```bash
pytest -q tests/test_insight_extraction.py tests/test_config.py tests/test_prompts.py tests/test_cli_sources.py tests/test_job_lifecycle.py
```

### Full suite

```bash
pytest -q
```

### Integration smoke test (requires running Docker + valid API keys)

```bash
# Confirm schema
docker compose exec -T postgres psql -U rag -d rag \
  -c "SELECT table_name FROM information_schema.tables WHERE table_name IN ('insights','chunk_insights');"

# Confirm 8-stage order
python -c "from rag.ingestion import STAGE_ORDER; assert STAGE_ORDER[-1] == 'insight_extraction', STAGE_ORDER"

# Ingest and run worker
rag ingest test_documents/sample.md
rag worker --poll-interval 1 --stuck-minutes 5

# Replace <source-id> with actual ID from ingest output
rag sources insights <source-id>
rag sources last 5
rag sources last 2026-01-01

# Memgraph verification
printf "MATCH (i:Insight) RETURN count(i);\n" | docker compose exec -T memgraph mgconsole
printf "MATCH ()-[r:CONTAINS]->() RETURN count(r);\n" | docker compose exec -T memgraph mgconsole
printf "MATCH ()-[r:RELATED_TO]->(:Insight) RETURN count(r);\n" | docker compose exec -T memgraph mgconsole

# Remediation
python scripts/remediate_insights.py --batch-size 5
```
