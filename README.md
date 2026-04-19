# Knowledge Graph RAG

Self-hosted ingestion and retrieval over PostgreSQL/pgvector and Memgraph, exposed through a CLI.

Use [docs/prd.md](/Users/estebanf/development/knowledge-graphrag/docs/prd.md) for product requirements. This README is the operator manual for local setup, ingestion, retrieval, and verification.

## Requirements

- Python `3.11+`
- Docker Desktop or compatible Docker engine
- OpenRouter API key with access to:
  - chat-completions models for metadata, profiling, graph extraction, and retrieval query rewriting
  - an embeddings model
  - a reranker model

## Installation

1. Copy the environment template:

```bash
cp .env.example .env
```

2. Fill in at least:

```bash
OPENROUTER_API_KEY=...
POSTGRES_PASSWORD=...
```

3. Start the local services:

```bash
./scripts/start.sh
```

4. Install the package into your active Python environment:

```bash
pip install -e .
```

5. Confirm connectivity:

```bash
venv/bin/rag health
```

## Environment

All model choices and retrieval tunables are env-backed. Nothing in the retrieval pipeline depends on hardcoded model IDs or thresholds.

Important variables:

- Core connections:
  - `POSTGRES_URL`
  - `MEMGRAPH_URL`
  - `STORAGE_BASE_PATH`
  - `OPENROUTER_API_KEY`
- Ingestion models:
  - `MODEL_METADATA_EXTRACTION`
  - `MODEL_DOC_PROFILING`
  - `MODEL_CHUNK_VALIDATION`
  - `MODEL_ENTITY_EXTRACTION`
  - `MODEL_RELATIONSHIP_EXTRACTION`
  - `MODEL_PROPOSITION_CHUNKING`
  - `MODEL_EMBEDDING`
- Retrieval models:
  - `MODEL_RETRIEVAL_QUERY_VARIANTS`
  - `MODEL_RETRIEVAL_GRAPH`
  - `MODEL_RETRIEVAL_RERANKER`
- Retrieval defaults:
  - `RETRIEVAL_RRF_K`
  - `RETRIEVAL_SEED_COUNT`
  - `RETRIEVAL_RESULT_COUNT`
  - `RETRIEVAL_FIRST_STAGE_TOP_N`
  - `RETRIEVAL_FUSED_CANDIDATE_COUNT`
  - `RETRIEVAL_ENTITY_SELECTION_COUNT`
  - `RETRIEVAL_SECOND_HOP_SELECTION_COUNT`
  - `RETRIEVAL_FIRST_HOP_CHUNK_COUNT`
  - `RETRIEVAL_SECOND_HOP_CHUNK_COUNT`
  - `RETRIEVAL_FIRST_HOP_SIMILARITY_THRESHOLD`
  - `RETRIEVAL_SECOND_HOP_SIMILARITY_THRESHOLD`
  - `RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD`
  - `RETRIEVAL_MAX_GRAPH_LLM_CALLS`
  - `RETRIEVAL_MAX_GRAPH_EXPANSION_MS`
  - `RETRIEVAL_MAX_GRAPH_EXPANSION_MS_PER_SEED`
  - `RETRIEVAL_TEXT_SEARCH_CONFIG`
  - `RETRIEVAL_WEIGHT_ORIGINAL`
  - `RETRIEVAL_WEIGHT_DECOMPOSED`
  - `RETRIEVAL_WEIGHT_EXPANDED`
  - `RETRIEVAL_WEIGHT_STEP_BACK`
  - `RETRIEVAL_WEIGHT_HYDE`
  - `RETRIEVAL_FINAL_ROOT_WEIGHT`
  - `RETRIEVAL_FINAL_FIRST_HOP_WEIGHT`
  - `RETRIEVAL_FINAL_SECOND_HOP_WEIGHT`
  - `RETRIEVAL_MULTI_PATH_BONUS`
  - `RETRIEVAL_SAME_SOURCE_NEIGHBOR_WINDOW`
  - `RETRIEVAL_SAME_SOURCE_NEIGHBOR_COUNT`
  - `RETRIEVAL_TRACE_MAX_CANDIDATES`
  - `RETRIEVAL_TRACE_MAX_ENTITIES`

## Services and Data

Start or restart the local database and graph services:

```bash
docker compose up -d
docker compose ps
```

Inspect Postgres directly:

```bash
docker compose exec -T postgres psql -U rag -d rag
```

Inspect Memgraph directly:

```bash
printf "MATCH (n) RETURN count(n);\n" | docker compose exec -T memgraph mgconsole
```

## Ingestion

The CLI submits ingestion jobs. The worker processes them asynchronously.

Submit one file:

```bash
venv/bin/rag ingest test_documents/Play\ 2.md
```

Submit a folder of supported files:

```bash
venv/bin/rag ingest test_documents
```

Attach metadata at ingest time:

```bash
venv/bin/rag ingest test_documents/Play\ 2.md --metadata kind=report --metadata domain=technical
```

Supported file types currently exercised in tests:

- `PDF`
- `DOCX`
- `PPTX`
- `MD`
- `TXT`

Start the worker:

```bash
venv/bin/rag worker --poll-interval 1 --stuck-minutes 30
```

## Job Operations

List recent jobs:

```bash
venv/bin/rag jobs list
venv/bin/rag jobs list --status failed
```

Inspect one job:

```bash
venv/bin/rag jobs status <job_id>
```

Retry a failed job:

```bash
venv/bin/rag jobs retry <job_id>
venv/bin/rag jobs retry <job_id> --from-stage chunking
```

Cancel a queued or running job:

```bash
venv/bin/rag jobs cancel <job_id>
```

## Source Operations

List active sources:

```bash
venv/bin/rag sources list
```

Inspect one source:

```bash
venv/bin/rag sources get <source_id>
```

Soft-delete a source:

```bash
venv/bin/rag sources delete <source_id>
```

Hard-delete a source and its stored file:

```bash
venv/bin/rag sources delete <source_id> --hard
```

## Retrieval

Run retrieval with defaults:

```bash
venv/bin/rag retrieve "What topics are covered in the ingested reports?"
```

Restrict retrieval to specific sources:

```bash
venv/bin/rag retrieve "What changed?" --source-id <source_uuid> --source-id <source_uuid>
```

Restrict retrieval by source metadata:

```bash
venv/bin/rag retrieve "What topics are covered?" --filter kind=report --filter domain=technical
```

Override retrieval parameters:

```bash
venv/bin/rag retrieve "What topics are covered?" \
  --seed-count 3 \
  --result-count 2 \
  --rrf-k 40 \
  --entity-confidence-threshold 0.8 \
  --first-hop-similarity-threshold 0.6 \
  --second-hop-similarity-threshold 0.6
```

Trace retrieval activity:

```bash
venv/bin/rag retrieve "What topics are covered in the ingested reports?" --trace
```

`--trace` prints live activity logs to stdout first, then prints the final JSON response as the last block. If you need machine-only output, do not pass `--trace`.

If graph expansion finds no non-seed chunks for a selected entity, retrieval falls back to bounded same-source neighbor chunks before giving up on that branch. This improves `related` results for structured decks and reports without broadening to unrelated sources.

The final response shape is:

```json
{
  "retrieval_results": [
    {
      "score": 0.0,
      "chunk": "",
      "chunk_id": "",
      "source_id": "",
      "source_path": "",
      "source_metadata": {},
      "related": []
    }
  ]
}
```

## Migrations and Existing Databases

If your local database predates the current code, apply the repo migrations in `scripts/migrate/`.

The migrations that matter most for the current code are:

- `scripts/migrate/001_add_markdown_content.sql`
- `scripts/migrate/002_update_vector_dimensions.sql`
- `scripts/migrate/004_job_improvements.sql`

Example:

```bash
docker compose exec -T postgres psql -U rag -d rag -f scripts/migrate/004_job_improvements.sql
```

## Verification

Targeted retrieval and CLI tests:

```bash
pytest -q tests/test_retrieval.py tests/test_cli_retrieve.py tests/test_config.py
```

Fast ingestion-oriented suite:

```bash
pytest -q tests/test_cli_jobs.py tests/test_job_lifecycle.py tests/test_worker.py tests/test_ingestion_submit.py tests/test_parser.py tests/test_storage.py tests/test_cli_ingest.py tests/test_chunking.py tests/test_observability.py tests/test_cli_health.py tests/test_profiling.py tests/test_chunk_validation.py tests/test_embedding.py
```

Full suite:

```bash
pytest -q
```

Live integration:

```bash
pytest -q tests/test_ingestion.py
venv/bin/rag retrieve "What topics are covered in the ingested reports?" --result-count 1 --seed-count 1 --trace
```

## Troubleshooting

- `rag health` fails:
  - confirm `docker compose ps`
  - confirm `POSTGRES_URL` and `MEMGRAPH_URL`
- Retrieval fails before searching:
  - confirm `OPENROUTER_API_KEY`
  - confirm retrieval model env vars are set
- Retrieval returns roots with empty `related`:
  - some chunks do not currently have outbound `MENTIONS` edges in Memgraph
  - retrieval still returns the reranked root chunk
- Sparse results are often empty:
  - current retrieval uses runtime full-text search over chunk content
  - exact lexical overlap matters more than dense retrieval
- Hard delete breaks:
  - dependent rows must be removed in this order: `entities`, `chunks`, `jobs`, `sources`
