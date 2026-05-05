# Insight Retrieval in `retrieve()` — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add insight discovery and expansion to the `retrieve()` pipeline, returning insights as a parallel list alongside chunk results, with 2-hop expansion (RELATED_TO edges + LLM-generated sub-queries). Also switch query variant generation from OpenRouter to the OpenCode endpoint with deepseek-v4-flash.

**Architecture:** The insight pipeline mirrors the chunk pipeline: query variants → dense+sparse insight search per variant → RRF fusion → rerank → N seed insights → per-seed 2-hop expansion (first hop: Memgraph RELATED_TO edges; second hop: LLM-generated sub-query → insight sub-search) → finalization (dedup, rerank, score). Both pipelines run under the same `retrieve()` call, returning `{retrieval_results: [...], insights: [...]}`. Query variant generation moves from OpenRouter to the OpenCode endpoint (`deepseek-v4-flash`), shared by both pipelines.

**Tech Stack:** Python, httpx (OpenCode), requests (OpenRouter — existing reranker/graph calls unchanged), Postgres pgvector, Memgraph Cypher, pytest, Typer CLI, FastAPI, React/TypeScript

---

### Task 1: Add OpenCode chat utility to retrieval.py

**Files:**
- Modify: `src/rag/retrieval.py` (near line 93-96, add new function)
- Create: tests in `tests/test_retrieval.py`
- Modify: `src/rag/config.py` (line 28, change default)

**Step 1: Write the failing test**

```python
from unittest.mock import MagicMock

def test_chat_json_opencode_calls_opencode_endpoint(monkeypatch):
    monkeypatch.setattr("rag.retrieval.settings.OPENCODE_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.__enter__.return_value.post.return_value.json.return_value = {
        "choices": [{"message": {"content": '{"foo":"bar"}'}}]
    }
    monkeypatch.setattr("rag.retrieval.httpx.Client", lambda **kw: mock_client)

    from rag.retrieval import _chat_json_opencode
    result = _chat_json_opencode("deepseek-v4-flash", "hello")

    assert result == {"foo": "bar"}
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_retrieval.py::test_chat_json_opencode_calls_opencode_endpoint -v
```
Expected: FAIL — `_chat_json_opencode` not defined.

**Step 3: Write minimal implementation**

Add import and function to `src/rag/retrieval.py`:

```python
import httpx

_OPENCODE_URL = "https://opencode.ai/zen/go/v1/chat/completions"


def _chat_json_opencode(model: str, prompt: str, *, timeout: int = 90) -> dict:
    if not settings.OPENCODE_API_KEY:
        raise ValueError("OPENCODE_API_KEY is required for OpenCode calls")
    with httpx.Client(timeout=httpx.Timeout(timeout)) as client:
        response = client.post(
            _OPENCODE_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.OPENCODE_API_KEY}",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        return _parse_json_response(response.json()["choices"][0]["message"]["content"])
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_retrieval.py::test_chat_json_opencode_calls_opencode_endpoint -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_retrieval.py src/rag/retrieval.py
git commit -m "feat: add OpenCode chat utility for retrieval"
```

---

### Task 2: Switch generate_query_variants to use OpenCode

**Files:**
- Modify: `src/rag/retrieval.py:239` (change `_chat_json` → `_chat_json_opencode`)
- Modify: `src/rag/config.py:28` (change default model)

**Step 1: Write test that verifies variant generation uses OpenCode**

```python
def test_generate_query_variants_uses_opencode(monkeypatch):
    monkeypatch.setattr("rag.retrieval.settings.OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr("rag.retrieval.settings.MODEL_RETRIEVAL_QUERY_VARIANTS", "deepseek-v4-flash")

    def fake_opencode(model, prompt, timeout=90):
        return {"original": "what changed", "hyde": "answer", "expanded": "alt"}
    monkeypatch.setattr("rag.retrieval._chat_json_opencode", fake_opencode)

    from rag.retrieval import generate_query_variants
    result = generate_query_variants("what changed in the policy?")

    assert result["original"] == "what changed"
    assert result["hyde"] == "answer"
```

**Step 2: Run test**

```bash
pytest tests/test_retrieval.py::test_generate_query_variants_uses_opencode -v
```
Expected: FAIL — `generate_query_variants` still calls `_chat_json` (OpenRouter).

**Step 3: Switch implementation**

In `src/rag/retrieval.py`, change line 239 from:
```python
raw = _chat_json(settings.MODEL_RETRIEVAL_QUERY_VARIANTS, prompt, timeout=90)
```
to:
```python
raw = _chat_json_opencode(settings.MODEL_RETRIEVAL_QUERY_VARIANTS, prompt, timeout=90)
```

In `src/rag/config.py`, change line 28 from:
```python
MODEL_RETRIEVAL_QUERY_VARIANTS: str = "google/gemini-2.5-flash-lite"
```
to:
```python
MODEL_RETRIEVAL_QUERY_VARIANTS: str = "deepseek-v4-flash"
```

**Step 4: Run tests**

```bash
pytest tests/test_retrieval.py::test_generate_query_variants_uses_opencode -v
pytest tests/test_retrieval.py::test_retrieve_emits_trace_and_returns_final_results -v
pytest tests/test_cli_retrieve.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add src/rag/retrieval.py src/rag/config.py
git commit -m "feat: switch query variant generation to OpenCode endpoint"
```

---

### Task 3: Add insight query variant prompt and generation

**Files:**
- Modify: `src/rag/prompts/__init__.py` (add INSIGHT_QUERY_VARIANTS)
- Modify: `src/rag/retrieval.py` (add `generate_insight_query_variants`)
- Create: tests in `tests/test_retrieval.py`

**Step 1: Write failing test**

```python
def test_generate_insight_query_variants_returns_clean_dict(monkeypatch):
    monkeypatch.setattr("rag.retrieval.settings.OPENCODE_API_KEY", "test-key")
    def fake_opencode(model, prompt, timeout=90):
        return {
            "original": "what changed",
            "hyde": "A policy amendment occurred",
            "expanded": "policy modification amendment change",
            "step_back": "organizational policy changes",
        }
    monkeypatch.setattr("rag.retrieval._chat_json_opencode", fake_opencode)

    from rag.retrieval import generate_insight_query_variants
    result = generate_insight_query_variants("what changed in the policy?")

    assert result["original"] == "what changed"
    assert result["hyde"] == "A policy amendment occurred"
    assert result["expanded"] == "policy modification amendment change"
    assert "decomposed" not in result
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_retrieval.py::test_generate_insight_query_variants_returns_clean_dict -v
```
Expected: FAIL — function not defined.

**Step 3: Add prompt and function**

In `src/rag/prompts/__init__.py`, after QUERY_VARIANTS:

```python
INSIGHT_QUERY_VARIANTS = """You are generating retrieval query variants for insight search.
Return ONLY a JSON object with keys: original, hyde, expanded, step_back.
- original must be the exact input query
- hyde must be a short hypothetical insight passage
- expanded must add synonyms, aliases, and related concepts
- step_back must be a more general background query
- avoid near-duplicate variants

Query:
{query}
"""
```

In `src/rag/retrieval.py`, after `generate_query_variants`:

```python
def generate_insight_query_variants(query: str, trace_logger: Optional[TraceLogger] = None) -> dict[str, object]:
    prompt = prompts.INSIGHT_QUERY_VARIANTS.format(query=query)
    raw = _chat_json_opencode(settings.MODEL_RETRIEVAL_QUERY_VARIANTS, prompt, timeout=90)
    raw["original"] = query
    variants = normalize_query_variants(raw)
    variants.pop("decomposed", None)
    if trace_logger:
        trace_logger.emit(f"generated insight query variants: {json.dumps(variants, ensure_ascii=True)}")
    return variants
```

**Step 4: Run test**

```bash
pytest tests/test_retrieval.py::test_generate_insight_query_variants_returns_clean_dict -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/prompts/__init__.py src/rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add insight query variant generation"
```

---

### Task 4: Seed insight selection (variants → dense+sparse → RRF → rerank)

**Files:**
- Modify: `src/rag/retrieval.py` (add `run_insight_first_stage_retrieval`)
- Create: tests in `tests/test_retrieval.py`

**Step 1: Write failing test**

```python
def test_run_insight_first_stage_retrieval_fuses_per_variant_results(monkeypatch):
    def fake_insight_search(conn, query, vector, top_n):
        if "policy" in query:
            return [InsightSearchResult(score=0.9, insight="policy changed", insight_id="i1", topics=[], sources=[])]
        return [InsightSearchResult(score=0.8, insight="roadmap shift", insight_id="i2", topics=[], sources=[])]

    monkeypatch.setattr("rag.retrieval.insight_hybrid_search", fake_insight_search)
    monkeypatch.setattr("rag.retrieval.get_embeddings", lambda texts: [[0.1]*4096]*len(texts))
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_RRF_K", 60)
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_FUSED_CANDIDATE_COUNT", 20)

    from rag.retrieval import run_insight_first_stage_retrieval
    variants = {"original": "policy change", "expanded": "roadmap shift"}
    results = run_insight_first_stage_retrieval(
        conn=object(), query="policy change", variants=variants,
        source_ids=[], filters={}, rrf_k=60, trace_logger=None,
    )

    assert len(results) >= 1
    assert all(r.score > 0 for r in results)
```

**Step 2: Run**

```bash
pytest tests/test_retrieval.py::test_run_insight_first_stage_retrieval_fuses_per_variant_results -v
```
Expected: FAIL — function not defined.

**Step 3: Implement**

Add to `src/rag/retrieval.py`:

```python
def run_insight_first_stage_retrieval(
    conn,
    query: str,
    variants: dict[str, object],
    source_ids: list[str],
    filters: dict[str, str],
    rrf_k: int,
    trace_logger: Optional[TraceLogger] = None,
) -> list[InsightSearchResult]:
    vectors = {}
    all_candidates: list[list[InsightSearchResult]] = []
    variant_names: list[str] = []
    variant_values: list[str] = []

    for key in ("original", "expanded", "hyde", "step_back"):
        value = variants.get(key)
        if not value or not isinstance(value, str):
            continue
        variant_names.append(key)
        variant_values.append(value)

    if not variant_values:
        return []

    embeddings = get_embeddings(variant_values)
    for i, key in enumerate(variant_names):
        vectors[key] = embeddings[i]

    for key, value in zip(variant_names, variant_values):
        vector = vectors.get(key)
        if vector is None:
            continue
        candidates = insight_hybrid_search(
            value,
            vector=vector,
            limit=settings.RETRIEVAL_FUSED_CANDIDATE_COUNT,
            min_score=0.0,
            conn=conn,
        )
        all_candidates.append(candidates)
        if trace_logger:
            trace_logger.emit(f"insight variant '{key}' returned {len(candidates)} candidates")

    if not all_candidates:
        return []

    rrf_scores: dict[str, float] = {}
    rrf_map: dict[str, InsightSearchResult] = {}
    for candidates in all_candidates:
        for rank, candidate in enumerate(candidates):
            rrf = 1.0 / (rrf_k + rank + 1)
            if candidate.insight_id not in rrf_scores or rrf > rrf_scores[candidate.insight_id]:
                rrf_scores[candidate.insight_id] = rrf
                rrf_map[candidate.insight_id] = candidate

    fused = sorted(rrf_map.values(), key=lambda c: rrf_scores[c.insight_id], reverse=True)
    fused = fused[:settings.RETRIEVAL_FUSED_CANDIDATE_COUNT]

    if trace_logger:
        trace_logger.emit(f"insight first stage fused {len(fused)} candidates")

    return fused
```

**Step 4: Run test**

```bash
pytest tests/test_retrieval.py::test_run_insight_first_stage_retrieval_fuses_per_variant_results -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add insight first-stage retrieval with per-variant RRF fusion"
```

---

### Task 5: First-hop insight expansion via RELATED_TO edges

**Files:**
- Modify: `src/rag/retrieval.py` (add `_load_related_insights`)
- Create: tests in `tests/test_retrieval.py`

**Step 1: Write failing test**

```python
class FakeInsightSession:
    def __init__(self):
        self.query = None
    def run(self, query, **kwargs):
        self.query = query
        self.kwargs = kwargs
        return iter([{"insight_id": "i2", "content": "related insight", "similarity": 0.92}])
    def __enter__(self): return self
    def __exit__(self, *a): return False

class FakeInsightDriver:
    def __init__(self):
        self.session_obj = FakeInsightSession()
    def session(self): return self.session_obj

def test_load_related_insights_uses_related_to(monkeypatch):
    driver = FakeInsightDriver()
    from rag.retrieval import _load_related_insights
    related = _load_related_insights(driver, "i1")
    assert len(related) == 1
    assert related[0]["insight_id"] == "i2"
    assert "RELATED_TO" in driver.session_obj.query
    assert driver.session_obj.kwargs["insight_id"] == "i1"
```

**Step 2: Run**

```bash
pytest tests/test_retrieval.py::test_load_related_insights_uses_related_to -v
```
Expected: FAIL — function not defined.

**Step 3: Implement**

```python
def _load_related_insights(driver, insight_id: str, top_k: int = 10) -> list[dict]:
    with driver.session() as session:
        result = session.run(
            """
            MATCH (i:Insight {insight_id: $insight_id})-[r:RELATED_TO]->(related:Insight)
            RETURN related.insight_id AS insight_id, related.content AS content, r.similarity AS similarity
            ORDER BY r.similarity DESC
            LIMIT $top_k
            """,
            insight_id=insight_id,
            top_k=top_k,
        )
        return list(result)
```

**Step 4: Run test**

```bash
pytest tests/test_retrieval.py::test_load_related_insights_uses_related_to -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add RELATED_TO graph expansion for insights"
```

---

### Task 6: Second-hop insight expansion via LLM sub-queries

**Files:**
- Modify: `src/rag/prompts/__init__.py` (add INSIGHT_SUB_QUERY prompt)
- Modify: `src/rag/retrieval.py` (add `_generate_insight_sub_query`)
- Create: test in `tests/test_retrieval.py`

**Step 1: Write failing test**

```python
def test_generate_insight_sub_query_returns_query(monkeypatch):
    monkeypatch.setattr("rag.retrieval.settings.OPENCODE_API_KEY", "test-key")
    def fake_opencode(model, prompt, timeout=90):
        return {"query": "find similar strategic shifts"}
    monkeypatch.setattr("rag.retrieval._chat_json_opencode", fake_opencode)
    monkeypatch.setattr("rag.retrieval.settings.MODEL_RETRIEVAL_QUERY_VARIANTS", "deepseek-v4-flash")

    from rag.retrieval import _generate_insight_sub_query
    result = _generate_insight_sub_query("what changed?", "The roadmap shifted focus from growth to efficiency")
    assert result == "find similar strategic shifts"
```

**Step 2: Run**

```bash
pytest tests/test_retrieval.py::test_generate_insight_sub_query_returns_query -v
```
Expected: FAIL — function not defined.

**Step 3: Implement**

Add to `src/rag/prompts/__init__.py`:

```python
INSIGHT_SUB_QUERY = """You are generating a sub-query to find insights related to a seed insight.
Return ONLY a JSON object with key "query".

Original user query:
{original_query}

Seed insight:
{insight}

Generate a focused search query that would find semantically similar insights."""
```

Add to `src/rag/retrieval.py`:

```python
def _generate_insight_sub_query(
    original_query: str,
    insight: str,
    trace_logger: Optional[TraceLogger] = None,
) -> str:
    prompt = prompts.INSIGHT_SUB_QUERY.format(
        original_query=original_query,
        insight=insight,
    )
    raw = _chat_json_opencode(settings.MODEL_RETRIEVAL_QUERY_VARIANTS, prompt, timeout=60)
    return str(raw.get("query", "")).strip()
```

**Step 4: Run test**

```bash
pytest tests/test_retrieval.py::test_generate_insight_sub_query_returns_query -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/prompts/__init__.py src/rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add LLM-generated insight sub-query for second-hop expansion"
```

---

### Task 7: Per-seed insight expansion orchestrator

**Files:**
- Modify: `src/rag/retrieval.py` (add `expand_seed_insight`)
- Create: tests in `tests/test_retrieval.py`

**Step 1: Write failing test**

```python
def test_expand_seed_insight_returns_related_and_second_hop(monkeypatch):
    monkeypatch.setattr("rag.retrieval.settings.OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_RRF_K", 60)
    monkeypatch.setattr("rag.retrieval.settings.MODEL_RETRIEVAL_QUERY_VARIANTS", "deepseek-v4-flash")

    def fake_related(driver, insight_id, top_k=10):
        return [{"insight_id": "i2", "content": "related insight", "similarity": 0.92}]

    def fake_sub_query(original_query, insight, trace_logger=None):
        return "find similar strategic shifts"

    def fake_insight_search(conn, query, vector, top_n):
        return [InsightSearchResult(
            score=0.85, insight="found insight", insight_id="i3",
            topics=["strategy"], sources=[]
        )]

    monkeypatch.setattr("rag.retrieval._load_related_insights", fake_related)
    monkeypatch.setattr("rag.retrieval._generate_insight_sub_query", fake_sub_query)
    monkeypatch.setattr("rag.retrieval.insight_hybrid_search", fake_insight_search)
    monkeypatch.setattr("rag.retrieval.get_embeddings", lambda texts: [[0.1]*4096]*len(texts))

    from rag.retrieval import expand_seed_insight
    seed = InsightSearchResult(score=0.95, insight="seed insight", insight_id="i1", topics=[], sources=[])
    result = expand_seed_insight(seed, "what changed?", conn=object(), driver=object(), trace_logger=None)

    assert "seed" in result
    assert result["seed"]["insight_id"] == "i1"
    assert len(result["related"]) > 0
    assert result["related"][0]["insight_id"] == "i2"
    assert len(result["second_level_related"]) > 0
    assert result["second_level_related"][0]["insight_id"] == "i3"
```

**Step 2: Run**

```bash
pytest tests/test_retrieval.py::test_expand_seed_insight_returns_related_and_second_hop -v
```
Expected: FAIL — function not defined.

**Step 3: Implement**

```python
def expand_seed_insight(
    seed: InsightSearchResult,
    query: str,
    conn,
    driver,
    trace_logger: Optional[TraceLogger] = None,
) -> dict:
    result: dict[str, object] = {
        "seed": {
            "insight_id": seed.insight_id,
            "insight": seed.insight,
            "score": seed.score,
            "topics": seed.topics,
        },
        "related": [],
        "second_level_related": [],
    }

    first_hop = _load_related_insights(driver, seed.insight_id)
    seen_ids = {seed.insight_id}
    for related in first_hop:
        iid = related["insight_id"]
        if iid in seen_ids:
            continue
        seen_ids.add(iid)
        result["related"].append({
            "insight_id": iid,
            "insight": related["content"],
            "score": float(related.get("similarity", 0.0)),
            "topics": related.get("topics", []),
        })

    if trace_logger:
        trace_logger.emit(f"insight expansion: {len(first_hop)} first-hop related")

    for related_insight in first_hop[:3]:
        sub_query = _generate_insight_sub_query(query, related_insight["content"], trace_logger=trace_logger)
        if not sub_query:
            continue
        vector = get_embeddings([sub_query])[0]
        sub_results = insight_hybrid_search(
            sub_query, vector=vector,
            limit=5, min_score=0.0, conn=conn,
        )
        for sr in sub_results:
            iid = sr.insight_id
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
            result["second_level_related"].append({
                "insight_id": iid,
                "insight": sr.insight,
                "score": sr.score,
                "topics": sr.topics,
                "relationship": {
                    "label": "SEMANTIC_RELATED",
                    "metadata": {
                        "first_hop_insight_id": related_insight["insight_id"],
                        "sub_query": sub_query,
                    },
                },
            })

    if trace_logger:
        trace_logger.emit(
            f"insight expansion complete: {len(result['related'])} related, {len(result['second_level_related'])} second-level"
        )

    return result
```

**Step 4: Run test**

```bash
pytest tests/test_retrieval.py::test_expand_seed_insight_returns_related_and_second_hop -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add per-seed insight expansion with two-hop discovery"
```

---

### Task 8: Insight result finalization (dedup, rerank, score)

**Files:**
- Modify: `src/rag/retrieval.py` (add `finalize_insight_results`)
- Create: tests in `tests/test_retrieval.py`

**Step 1: Write failing test**

```python
def test_finalize_insight_results_dedups_and_sorts(monkeypatch):
    expanded = [
        {
            "seed": {"insight_id": "i1", "insight": "s1", "score": 0.9, "topics": []},
            "related": [
                {"insight_id": "i2", "insight": "r1", "score": 0.8, "topics": []},
                {"insight_id": "i3", "insight": "r2", "score": 0.7, "topics": []},
            ],
            "second_level_related": [
                {"insight_id": "i4", "insight": "s2", "score": 0.6, "topics": [], "relationship": {"label": "X", "metadata": {}}},
            ],
        },
        {
            "seed": {"insight_id": "i2", "insight": "r1", "score": 0.85, "topics": []},
            "related": [],
            "second_level_related": [],
        },
    ]
    monkeypatch.setattr("rag.retrieval.rerank_candidates", lambda query, candidates, top_n, trace_logger=None: candidates)

    from rag.retrieval import finalize_insight_results
    results = finalize_insight_results("what changed?", expanded, result_count=5, trace_logger=None)

    assert len(results) <= 5
    ids = [r["insight_id"] for r in results]
    assert "i1" in ids
    assert ids.count("i2") == 1
    assert results[0]["insight_id"] == "i1"
```

**Step 2: Run**

```bash
pytest tests/test_retrieval.py::test_finalize_insight_results_dedups_and_sorts -v
```
Expected: FAIL — function not defined.

**Step 3: Implement**

```python
def finalize_insight_results(
    query: str,
    expanded_results: list[dict],
    result_count: int,
    trace_logger: Optional[TraceLogger] = None,
) -> list[dict]:
    seen: dict[str, dict] = {}
    for expanded in expanded_results:
        seed = expanded["seed"]
        iid = seed["insight_id"]
        if iid not in seen or seed["score"] > seen[iid].get("score", 0):
            seen[iid] = seed
        for related in expanded["related"]:
            iid = related["insight_id"]
            if iid not in seen or related["score"] > seen[iid].get("score", 0):
                seen[iid] = related
        for sl in expanded.get("second_level_related", []):
            iid = sl["insight_id"]
            if iid not in seen or sl["score"] > seen[iid].get("score", 0):
                seen[iid] = sl

    results = sorted(seen.values(), key=lambda r: r.get("score", 0), reverse=True)
    results = results[:result_count]

    if trace_logger:
        trace_logger.emit(f"finalized {len(results)} insight results")

    return results
```

**Step 4: Run test**

```bash
pytest tests/test_retrieval.py::test_finalize_insight_results_dedups_and_sorts -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add insight result finalization with dedup and scoring"
```

---

### Task 9: Wire insight pipeline into retrieve()

**Files:**
- Modify: `src/rag/retrieval.py` (modify `retrieve()`)
- Modify: `src/rag/config.py` (add `RETRIEVAL_INSIGHT_SEED_COUNT`)
- Create: tests in `tests/test_retrieval.py`

**Step 1: Write integration test**

```python
def test_retrieve_returns_insights_alongside_chunks(monkeypatch):
    monkeypatch.setattr("rag.retrieval.settings.OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("rag.retrieval.settings.OPENCODE_API_KEY", "test-key")
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_RRF_K", 60)
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_SEED_COUNT", 2)
    monkeypatch.setattr("rag.retrieval.settings.RETRIEVAL_RESULT_COUNT", 3)
    monkeypatch.setattr("rag.retrieval.settings.MODEL_RETRIEVAL_QUERY_VARIANTS", "deepseek-v4-flash")

    variants = {"original": "what changed", "hyde": "hypothetical"}
    monkeypatch.setattr("rag.retrieval.generate_query_variants", lambda q, **kw: variants)
    monkeypatch.setattr("rag.retrieval.generate_insight_query_variants", lambda q, **kw: variants)

    chunk = RetrievalCandidate("c1", "chunk text", "s1", "/p", {}, 0.9)
    insight = InsightSearchResult(0.88, "insight text", "i1", ["strategy"], [])
    monkeypatch.setattr("rag.retrieval.run_first_stage_retrieval", lambda **kw: [chunk])
    monkeypatch.setattr("rag.retrieval.run_insight_first_stage_retrieval", lambda **kw: [insight])
    monkeypatch.setattr("rag.retrieval.rerank_candidates", lambda q, c, **kw: c[:kw["top_n"]])
    monkeypatch.setattr("rag.retrieval.expand_seed_candidate", lambda *a, **kw: {
        "root": chunk, "related": [], "final_score": 0.9, "chunk_id": "c1", "chunk": "c",
        "source_id": "s1", "source_path": "/p", "source_metadata": {},
    })
    monkeypatch.setattr("rag.retrieval.expand_seed_insight", lambda *a, **kw: {
        "seed": {"insight_id": "i1", "insight": "insight text", "score": 0.88, "topics": []},
        "related": [], "second_level_related": [],
    })
    monkeypatch.setattr("rag.retrieval.finalize_root_results", lambda *a, **kw: [{
        "chunk_id": "c1", "chunk": "c", "score": 0.9, "source_id": "s1",
        "source_path": "/p", "source_metadata": {}, "related": [],
    }])
    monkeypatch.setattr("rag.retrieval.finalize_insight_results", lambda *a, **kw: [{
        "insight_id": "i1", "insight": "insight text", "score": 0.88, "topics": [],
    }])
    monkeypatch.setattr("rag.retrieval._expand_neighbor_contexts", lambda conn, results: None)

    response = retrieve(
        query="what changed?", source_ids=[], filters={}, seed_count=2, result_count=3,
        rrf_k=60, entity_confidence_threshold=None, first_hop_similarity_threshold=None,
        second_hop_similarity_threshold=None, trace=False, trace_printer=None,
    )

    assert "retrieval_results" in response
    assert "insights" in response
    assert len(response["insights"]) > 0
    assert response["insights"][0]["insight_id"] == "i1"
```

**Step 2: Run**

```bash
pytest tests/test_retrieval.py::test_retrieve_returns_insights_alongside_chunks -v
```
Expected: FAIL — retrieve() doesn't return insights key yet.

**Step 3: Implement — modify retrieve()**

Add config in `src/rag/config.py`:
```python
RETRIEVAL_INSIGHT_SEED_COUNT: Annotated[int, Field(gt=0)] = 5
```

Modify `retrieve()` to run insight pipeline in parallel and return both keys.

**Step 4: Run all tests**

```bash
pytest tests/test_retrieval.py -v -x
pytest tests/test_cli_retrieve.py -v
```
Expected: All PASS.

**Step 5: Commit**

```bash
git add src/rag/retrieval.py src/rag/config.py tests/test_retrieval.py
git commit -m "feat: wire insight retrieval into retrieve() pipeline"
```

---

### Task 10: Update CLI

**Note:** The CLI already prints via `console.print_json(json.dumps(response))`, so the `insights` key will appear automatically. This task adds a test to verify.

**Step 1: Write test**

```python
from unittest.mock import patch

def test_retrieve_cli_prints_insights_section():
    from rag.cli import app
    from typer.testing import CliRunner

    def _response():
        return {
            "retrieval_results": [{"chunk_id": "c1", "chunk": "text", "score": 0.9, "source_id": "s1", "source_path": "/p", "source_metadata": {}, "related": []}],
            "insights": [{"insight_id": "i1", "insight": "insight text", "score": 0.88, "topics": ["strategy"]}],
        }

    with patch("rag.cli.retrieve", return_value=_response()):
        runner = CliRunner()
        result = runner.invoke(app, ["retrieve", "--trace", "what changed?"])
        assert "insights" in result.output.lower()
```

**Step 2: Run**

```bash
pytest tests/test_cli_retrieve.py::test_retrieve_cli_prints_insights_section -v
```
Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_cli_retrieve.py
git commit -m "test: verify CLI prints insights in retrieve output"
```

---

### Task 11: Update API response and test

**Files:**
- Create: test in `tests/test_api.py`

**Note:** The API route returns `dict` directly from `retrieve()`, so the `insights` key appears automatically.

**Step 1: Write test**

```python
def test_retrieve_endpoint_returns_insights():
    client = _client()
    def _response(*a, **kw):
        return {
            "retrieval_results": [{"chunk_id": "c1", "chunk": "text", "score": 0.9, "source_id": "s1", "source_path": "/p", "source_metadata": {}, "related": []}],
            "insights": [{"insight_id": "i1", "insight": "insight text", "score": 0.88, "topics": ["strategy"]}],
        }
    with patch("rag.api.routes.retrieve.retrieve", side_effect=_response):
        r = client.post("/api/retrieve", json={"query": "what changed?"})
        assert r.status_code == 200
        data = r.json()
        assert "insights" in data
        assert data["insights"][0]["insight_id"] == "i1"
```

**Step 2: Run**

```bash
pytest tests/test_api.py::test_retrieve_endpoint_returns_insights -v
```
Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_api.py
git commit -m "test: verify API returns insights in retrieve response"
```

---

### Task 12: Update frontend to render insights in retrieve mode

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/App.tsx`

Update `RetrieveResponse` type to include `insights`, and add `InsightCard` rendering in retrieve view.

**Build and verify:**
```bash
cd frontend && npm run build
```

**Commit:**
```bash
git add frontend/src/lib/api.ts frontend/src/App.tsx
git commit -m "feat: render insights in retrieve view"
```

---

### Task 13: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

Update Retrieval section to document `insights` key.

**Commit:**
```bash
git add README.md AGENTS.md
git commit -m "docs: document insight retrieval in retrieve()"
```

---

### Task 14: Rebuild containers and full verification

```bash
docker compose build --no-cache backend frontend
docker compose up -d
pytest -q tests/test_retrieval.py tests/test_cli_retrieve.py tests/test_hybrid_search.py tests/test_api.py tests/test_cli_search.py tests/test_community.py tests/test_config.py tests/test_prompts.py
```
