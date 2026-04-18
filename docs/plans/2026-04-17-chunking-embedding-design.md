# Adaptive Chunking + Embeddings Design

**Date:** 2026-04-17  
**Branch:** feature02-chunking  
**Status:** Approved

## Context

Phase 1 implemented parsing and metadata extraction. This design covers the next four pipeline stages: profiling, chunking, validation, and embedding. The goal is to split ingested documents into quality-validated chunks and store dense vector embeddings for hybrid retrieval.

The implementation stays synchronous (extending the existing `ingest_file()` flow) and uses LangChain text splitters for chunking strategies.

---

## Pipeline Extension

```
parsing → metadata_extraction → profiling → chunking → validation → embedding → completed
```

Each stage:
- Updates `jobs.status = 'processing:<stage>'` before starting
- Sets `jobs.status = 'failed:<stage>'` on error and halts

---

## Stage 1: Profiling (`src/rag/profiling.py`)

**Purpose:** Determine the document's structural characteristics to inform chunking strategy selection.

**Approach:** Single LLM call to OpenRouter using `MODEL_DOC_PROFILING` (default: `google/gemma-3-4b-it`). Samples the first 3000 characters of the parsed markdown. Returns a `DocumentProfile` dataclass.

**Output fields:**
- `structure_type`: `well-structured | loosely-structured | unstructured`
- `heading_consistency`: `consistent | inconsistent | none`
- `content_density`: `uniform | variable`
- `primary_content_type`: `prose | tabular | mixed | qa_pairs | code | transcript`
- `avg_section_length`: `short | medium | long`
- `has_tables`: bool
- `has_code_blocks`: bool
- `domain`: `legal | financial | technical | general | medical | policy`

**Error behavior:** On any failure, returns a safe default profile (`unstructured`, `general`) so ingestion continues without halting.

**Pattern:** Same `requests.post` pattern as `metadata_extraction.py` with `temperature=0` and JSON-only response.

---

## Stage 2: Chunking (`src/rag/chunking.py`)

**Purpose:** Split the markdown into chunks using a strategy selected deterministically from the document profile.

**Strategy selection matrix:**

| Profile condition | Base strategy |
|---|---|
| `structure_type == well-structured` AND `heading_consistency == consistent` | `MarkdownHeaderTextSplitter` |
| `primary_content_type in {transcript, qa_pairs}` | `RecursiveCharacterTextSplitter` with sentence separators |
| `content_density == uniform` AND `structure_type == unstructured` | `RecursiveCharacterTextSplitter` |
| All other cases | `RecursiveCharacterTextSplitter` (fallback) |

**Modifier layers** (applied on top of base strategy):
- `avg_section_length == long` → Hierarchical parent-child: parent chunks at 2000 tokens, child at 400 tokens, `parent_chunk_id` set on children
- `domain in {legal, financial, medical}` → Proposition chunking: LLM call (`MODEL_PROPOSITION_CHUNKING`, default `qwen/qwen2.5-14b-instruct`) decomposes each chunk into atomic facts; each proposition becomes a child chunk

**Libraries:** `langchain-text-splitters`, `tiktoken` (token counting)  
**Note:** No `SemanticChunker` — avoids a second embedding pass during chunking.

**DB write:** All chunks inserted immediately with `chunking_strategy`, `chunking_config` (JSON with chunk_size, overlap, etc.), `parent_chunk_id`, and `token_count`. `embedding` column left NULL at this stage.

---

## Stage 3: Validation (`src/rag/chunk_validation.py`)

**Purpose:** Sample-based LLM quality check before generating embeddings.

**Sample rates** (from env vars already in `.env.example`):
- Default: `CHUNK_VALIDATION_SAMPLE_RATE=0.10` (10%)
- High-stakes domains (legal/financial/medical): `CHUNK_VALIDATION_SAMPLE_RATE_HIGH_STAKES=0.25`
- First ingestion of a new domain type for this deployment: 100%

**Approach:** Each sampled chunk is sent to OpenRouter (`MODEL_CHUNK_VALIDATION`, default `qwen/qwen2.5-7b-instruct`) with a prompt asking for pass/fail on: completeness, coherence, appropriate length. Returns JSON `{"pass": true}`.

**Failure threshold:** If >20% of sampled chunks fail → job transitions to `failed:validation`. Chunks already written to DB are soft-deleted (set `deleted_at`).

---

## Stage 4: Embedding (`src/rag/embedding.py`)

**Purpose:** Generate dense vector embeddings for all chunks and store them.

**API:** OpenRouter embeddings endpoint (`POST https://openrouter.ai/api/v1/embeddings`), model from `MODEL_EMBEDDING` env var (default: `qwen/qwen3-embedding-8b`).

**Batching:** 32 chunks per request. Results bulk-updated to `chunks.embedding` via `executemany`.

**Schema migration required:** `qwen/qwen3-embedding-8b` outputs 4096 dimensions. Migration script `scripts/migrate/002_update_vector_dimensions.sql` drops and recreates `chunks.embedding` and `entities.embedding` as `vector(4096)` and rebuilds HNSW indexes.

**New env vars:**
- `MODEL_EMBEDDING=qwen/qwen3-embedding-8b`
- `EMBEDDING_DIMENSIONS=4096`

---

## Config Changes (`src/rag/config.py`)

New `Settings` fields:
- `MODEL_DOC_PROFILING: str = "google/gemma-3-4b-it"`
- `MODEL_CHUNK_VALIDATION: str = "qwen/qwen2.5-7b-instruct"`
- `MODEL_PROPOSITION_CHUNKING: str = "qwen/qwen2.5-14b-instruct"`
- `MODEL_EMBEDDING: str = "qwen/qwen3-embedding-8b"`
- `EMBEDDING_DIMENSIONS: int = 4096`
- `CHUNK_VALIDATION_SAMPLE_RATE: float = 0.10`
- `CHUNK_VALIDATION_SAMPLE_RATE_HIGH_STAKES: float = 0.25`

---

## Files to Create

| File | Purpose |
|---|---|
| `src/rag/profiling.py` | LLM document profiling |
| `src/rag/chunking.py` | Adaptive chunking logic |
| `src/rag/chunk_validation.py` | Chunk quality validation |
| `src/rag/embedding.py` | Embedding generation + DB storage |
| `scripts/migrate/002_update_vector_dimensions.sql` | Vector dimension migration |

## Files to Modify

| File | Change |
|---|---|
| `src/rag/config.py` | Add new Settings fields |
| `src/rag/ingestion.py` | Add 4 new pipeline stages |
| `pyproject.toml` | Add `langchain-text-splitters`, `tiktoken` |
| `.env.example` | Document `MODEL_EMBEDDING`, `EMBEDDING_DIMENSIONS` |
| `.env` | Set actual values |
| `CLAUDE.md` | Update implementation status for Phase 2 |

---

## Verification

1. Run migration: `psql $POSTGRES_URL -f scripts/migrate/002_update_vector_dimensions.sql`
2. Install new deps: `pip install -e .`
3. Run `rag ingest test_documents/Play\ 2.md` — job should complete with `status: completed`
4. Query `SELECT count(*), chunking_strategy FROM chunks GROUP BY chunking_strategy` to verify chunks exist
5. Query `SELECT count(*) FROM chunks WHERE embedding IS NOT NULL` to verify embeddings stored
6. Run existing tests: `pytest tests/test_ingestion.py -v` — all 6 should still pass
7. Write new tests: `tests/test_chunking.py`, `tests/test_embedding.py`
