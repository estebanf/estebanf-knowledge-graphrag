# Knowledge Graph RAG

Self-hosted ingestion, search, retrieval, and community analysis over PostgreSQL/pgvector and Memgraph. The project ships with:

- a Python CLI for operators
- a FastAPI backend
- a React frontend served by nginx in Docker

## Requirements

- Python `3.11+`
- Docker Desktop or a compatible Docker engine
- OpenRouter API key with access to:
  - chat-completions models for metadata extraction, profiling, chunk validation, graph extraction, retrieval query rewriting, graph-stage selection, image description, and community summarization
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

3. Start the local stack:

```bash
./scripts/start.sh
```

4. Install the package into your active Python environment:

```bash
pip install -e .
```

If you need local document parsing and ingestion support, install the ingest extra:

```bash
pip install -e .[ingest]
```

5. Confirm connectivity:

```bash
venv/bin/rag health
curl http://localhost:8000/api/health
```

## Environment

All runtime settings are env-backed. The CLI, backend, worker, and Docker services read configuration from `.env`.

### Core services and paths

| Variable | Default | Purpose |
| --- | --- | --- |
| `POSTGRES_PASSWORD` | `changeme` | Password used by the Postgres container. Required for local Docker startup. |
| `POSTGRES_USER` | `rag` | Postgres username created by the container. |
| `POSTGRES_DB` | `rag` | Postgres database name created by the container. |
| `POSTGRES_PORT` | `5432` | Host port mapped to the Postgres container. |
| `POSTGRES_URL` | `postgresql://rag:${POSTGRES_PASSWORD}@localhost:5432/rag` | Connection string used by the Python app outside Docker. |
| `POSTGRES_SHARED_BUFFERS` | `512MB` | Postgres shared buffer tuning for the container. |
| `POSTGRES_WORK_MEM` | `64MB` | Postgres per-operation working memory. |
| `POSTGRES_MAINTENANCE_WORK_MEM` | `256MB` | Postgres maintenance memory for operations like indexing. |
| `POSTGRES_MAX_CONNECTIONS` | `50` | Postgres connection limit in the container. |
| `MEMGRAPH_URL` | `bolt://localhost:7687` | Bolt connection string used by the Python app outside Docker. |
| `MEMGRAPH_BOLT_PORT` | `7687` | Host port mapped to Memgraph Bolt. |
| `MEMGRAPH_LAB_PORT` | `3000` | Host port mapped to Memgraph Lab. |
| `MEMGRAPH_MEMORY_LIMIT` | `2048` | Memgraph container memory limit passed to the server. |
| `STORAGE_BASE_PATH` | `./data/documents` | Base directory where original source files and copied markdown images are stored. |
| `BACKEND_PORT` | `8000` | Host port mapped to the FastAPI backend container. |
| `FRONTEND_PORT` | `80` | Host port mapped to the nginx-served frontend container. |
| `OPENROUTER_API_KEY` | empty | Required for embeddings and all LLM-backed stages. |
| `LOG_LEVEL` | `INFO` | Application log verbosity. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OTLP collector endpoint for telemetry if observability is enabled. |

### Model selection

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL_METADATA_EXTRACTION` | `google/gemma-3-4b-it` | Extracts structured metadata from parsed markdown. |
| `MODEL_DOC_PROFILING` | `google/gemma-3-4b-it` | Profiles a document to guide chunking behavior. |
| `MODEL_CHUNK_VALIDATION` | `qwen/qwen2.5-7b-instruct` | Evaluates sampled chunks for quality control. |
| `MODEL_PROPOSITION_CHUNKING` | `qwen/qwen2.5-14b-instruct` | Decomposes text into propositions when chunking requires it. |
| `MODEL_EMBEDDING` | `qwen/qwen3-embedding-8b` | Embedding model used to vectorize chunks and search queries. |
| `MODEL_ENTITY_EXTRACTION` | `qwen/qwen-2.5-7b-instruct` | Extracts entities from chunk text. |
| `MODEL_RELATIONSHIP_EXTRACTION` | `qwen/qwen-2.5-7b-instruct` | Extracts entity relationships from chunk text. |
| `MODEL_IMAGE_DESCRIPTION` | `google/gemini-2.0-flash-lite-001` | Describes inline images found during parsing. |
| `MODEL_RETRIEVAL_QUERY_VARIANTS` | `google/gemini-2.5-flash-lite` | Generates retrieval query variants. |
| `MODEL_RETRIEVAL_GRAPH` | `google/gemini-2.5-flash-lite` | Selects entities and graph traversal queries during retrieval. |
| `MODEL_RETRIEVAL_RERANKER` | `cohere/rerank-v3.5` | Reranks first-stage and final retrieval candidates. |

### Embedding, chunking, and graph extraction

| Variable | Default | Purpose |
| --- | --- | --- |
| `EMBEDDING_DIMENSIONS` | `4096` | Expected vector length stored in Postgres. |
| `CHUNK_VALIDATION_SAMPLE_RATE` | `0.10` | Fraction of chunks sampled for validation on standard documents. |
| `CHUNK_VALIDATION_SAMPLE_RATE_HIGH_STAKES` | `0.25` | Higher validation sampling rate for high-stakes domains. |
| `RELATIONSHIP_CONFIDENCE_THRESHOLD` | `0.75` | Minimum relationship confidence kept during graph extraction. |
| `ENTITY_DEDUP_COSINE_THRESHOLD` | `0.92` | Similarity threshold used when deduplicating entities. |

### Insight extraction

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENCODE_API_KEY` | empty | API key for OpenCode service used in insight extraction. |
| `INSIGHT_DEDUP_COSINE_THRESHOLD` | `0.95` | Minimum cosine similarity to reuse an existing insight instead of creating a new one. |
| `INSIGHT_LINK_TOP_K` | `10` | Number of nearest insight neighbors used for mutual top-K `RELATED_TO` edge creation. |

### Search defaults

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARCH_DEFAULT_LIMIT` | `10` | Default `rag search --limit` value. |
| `SEARCH_MIN_SCORE` | `0.7` | Default minimum score for `rag search`. |

### Retrieval defaults and tunables

| Variable | Default | Purpose |
| --- | --- | --- |
| `RETRIEVAL_RRF_K` | `60` | Reciprocal rank fusion constant used to merge candidate lists. |
| `RETRIEVAL_RRF_SCORE_FLOOR` | `0.0` | Minimum fused score retained after RRF. |
| `RETRIEVAL_SEED_COUNT` | `10` | Number of top reranked seed chunks expanded in the graph stage. |
| `RETRIEVAL_RESULT_COUNT` | `5` | Number of final root results returned. |
| `RETRIEVAL_MAX_DECOMPOSED_QUERIES` | `5` | Max number of decomposed variants generated from the query. |
| `RETRIEVAL_FIRST_STAGE_TOP_N` | `20` | Top candidates kept per first-stage search path before fusion. |
| `RETRIEVAL_FUSED_CANDIDATE_COUNT` | `50` | Max fused first-stage candidates retained before reranking. |
| `RETRIEVAL_ENTITY_SELECTION_COUNT` | `5` | Max entities selected from each seed for first-hop expansion. |
| `RETRIEVAL_SECOND_HOP_SELECTION_COUNT` | `5` | Max entities selected for second-hop expansion. |
| `RETRIEVAL_FIRST_HOP_CHUNK_COUNT` | `5` | Max chunk candidates pulled per first-hop entity query. |
| `RETRIEVAL_SECOND_HOP_CHUNK_COUNT` | `5` | Max chunk candidates pulled per second-hop entity query. |
| `RETRIEVAL_FIRST_HOP_SIMILARITY_THRESHOLD` | `0.5` | Minimum similarity for first-hop entity-linked chunks. |
| `RETRIEVAL_SECOND_HOP_SIMILARITY_THRESHOLD` | `0.5` | Minimum similarity for second-hop entity-linked chunks. |
| `RETRIEVAL_ENTITY_CONFIDENCE_THRESHOLD` | `0.75` | Minimum relationship confidence used during graph expansion. |
| `RETRIEVAL_MAX_GRAPH_LLM_CALLS` | `100` | Shared safety cap on graph-stage LLM calls across a retrieval request. |
| `RETRIEVAL_MAX_GRAPH_EXPANSION_MS` | `4000` | Global graph expansion budget knob retained in config. |
| `RETRIEVAL_MAX_GRAPH_EXPANSION_MS_PER_SEED` | `4000` | Wall-clock budget applied to each seed during graph expansion. |
| `RETRIEVAL_TEXT_SEARCH_CONFIG` | `english` | Postgres full-text search configuration for sparse retrieval. |
| `RETRIEVAL_WEIGHT_ORIGINAL` | `1.0` | Weight applied to the original query path in first-stage fusion. |
| `RETRIEVAL_WEIGHT_DECOMPOSED` | `1.0` | Weight applied to decomposed query paths. |
| `RETRIEVAL_WEIGHT_EXPANDED` | `0.85` | Weight applied to expanded query variants. |
| `RETRIEVAL_WEIGHT_STEP_BACK` | `0.75` | Weight applied to step-back query variants. |
| `RETRIEVAL_WEIGHT_HYDE` | `0.65` | Weight applied to HyDE-style variants. |
| `RETRIEVAL_FINAL_ROOT_WEIGHT` | `0.60` | Weight of the reranked root chunk in final aggregation. |
| `RETRIEVAL_FINAL_FIRST_HOP_WEIGHT` | `0.25` | Weight of first-hop evidence in final aggregation. |
| `RETRIEVAL_FINAL_SECOND_HOP_WEIGHT` | `0.15` | Weight of second-hop evidence in final aggregation. |
| `RETRIEVAL_MULTI_PATH_BONUS` | `0.05` | Bonus for evidence found through multiple graph paths. |
| `RETRIEVAL_SAME_SOURCE_NEIGHBOR_WINDOW` | `2` | Window used when falling back to same-source neighboring chunks. |
| `RETRIEVAL_SAME_SOURCE_NEIGHBOR_COUNT` | `3` | Max neighboring chunks returned by same-source fallback. |
| `RETRIEVAL_EXPANSION_MIN_TOKENS` | `200` | Lower bound when trimming related chunk text used in expansion. |
| `RETRIEVAL_EXPANSION_MAX_TOKENS` | `600` | Upper bound when trimming related chunk text used in expansion. |
| `RETRIEVAL_TRACE_MAX_CANDIDATES` | `5` | Max candidates shown per trace step. |
| `RETRIEVAL_TRACE_MAX_ENTITIES` | `5` | Max entities shown per trace step. |

### Community and worker settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMMUNITY_SEMANTIC_THRESHOLD` | `0.85` | Similarity threshold for semantic edges between entities. |
| `COMMUNITY_SOURCE_COOC_WEIGHT` | `0.1` | Extra weight added when entities co-occur in the same source. |
| `COMMUNITY_CUTOFF` | `0.5` | Minimum chunk score kept when selecting representative chunks. |
| `COMMUNITY_MIN_COMMUNITY_SIZE` | `3` | Minimum entity count required for a community. |
| `COMMUNITY_TOP_K_CHUNKS` | `5` | Max representative chunks returned per community. |
| `COMMUNITY_SUMMARIZATION_PROMPT` | empty | Optional prompt override for community summarization. |
| `COMMUNITY_CROSS_SOURCE_TOP_K` | `10` | Max cross-source semantic neighbors fetched per entity via pgvector ANN. |
| `COMMUNITY_MAX_CROSS_SOURCE_QUERIES` | `5000` | Hard cap on per-entity ANN queries; entities are prioritized by chunk-mention count. |
| `WORKER_POLL_INTERVAL` | `5` | Default idle poll interval for `rag worker`. |
| `WORKER_STUCK_JOB_MINUTES` | `30` | Default age after which a processing job is considered stuck. |

## Services and Data

Start or restart the full local stack:

```bash
docker compose up -d
docker compose ps
```

The Docker stack includes:

- `postgres`: PostgreSQL plus pgvector/ParadeDB storage for sources, chunks, jobs, metadata, and embeddings
- `memgraph`: graph store for `Source`, `Chunk`, and `Entity` nodes plus relationship edges
- `backend`: FastAPI API server on `http://localhost:${BACKEND_PORT:-8000}`
- `frontend`: nginx-served React app on `http://localhost:${FRONTEND_PORT:-80}`

Persistent local data lives under:

- `data/postgres`
- `data/memgraph`
- `data/documents`
- `data/backups`

Useful service checks:

```bash
curl http://localhost:8000/api/health
docker compose exec -T postgres psql -U rag -d rag
printf "MATCH (n) RETURN count(n);\n" | docker compose exec -T memgraph mgconsole
```

## Ingestion

The CLI submits ingestion jobs. The worker processes them asynchronously through these stages:

1. `parsing`
2. `profiling`
3. `chunking`
4. `validation`
5. `embedding`
6. `graph_extraction`
7. `graph_linking`
8. `insight_extraction`

### What `rag ingest` does

- stores the original file under `STORAGE_BASE_PATH/<source_id>/1/`
- creates a `sources` row and a queued `jobs` row
- copies local markdown image assets into the same source storage tree
- when the worker runs, parses the document into markdown, extracts metadata, chunks it, embeds it, builds graph artifacts, and extracts chunk-level insights

### Supported file types

- `PDF`
- `DOCX`
- `PPTX`
- `MD`
- `TXT`

### CLI syntax

```bash
venv/bin/rag ingest [OPTIONS] PATH...
```

Arguments:

- `PATH...`: one or more files, or one folder

Options:

- `--name TEXT`: display name stored on the source when ingesting a single file
- `--metadata key=value`: attach one or more metadata pairs at submission time

Behavior notes:

- if you pass one folder, ingest scans only that folder's immediate files and queues every supported extension it finds
- `--name` is ignored for folder ingestion and for multi-file ingestion
- submitted metadata overrides extracted metadata when keys collide
- duplicate files are rejected by MD5 against active sources

Examples:

```bash
venv/bin/rag ingest test_documents/Play\ 2.md
venv/bin/rag ingest test_documents --metadata kind=report --metadata domain=technical
venv/bin/rag ingest report.pdf --name "Quarterly Report" --metadata team=platform
```

### Inline image handling

Image handling depends on the file type:

- Markdown:
  - local relative image references such as `![diagram](images/flow.png)` are copied into source storage at submission time
  - during parsing, those local image references are replaced with an LLM-generated text description
  - remote URLs and `data:` URLs are left unchanged
  - missing local files are left unchanged
- PDF, DOCX, and PPTX:
  - the parser exports markdown
  - embedded pictures are extracted and each `<!-- image -->` placeholder is replaced with an LLM-generated description
- TXT:
  - treated as plain text, with no image handling

Image descriptions use `MODEL_IMAGE_DESCRIPTION`.

### Worker

Start the worker:

```bash
venv/bin/rag worker --poll-interval 1 --stuck-minutes 30
```

Options:

- `--poll-interval INTEGER`: seconds between polls when the queue is empty
- `--stuck-minutes INTEGER`: age after which a processing job is treated as stuck

## Job Operations

### List jobs

```bash
venv/bin/rag jobs list
venv/bin/rag jobs list --status failed
venv/bin/rag jobs list --status completed
```

Options:

- `--status TEXT`: filter by status; `failed` and `processing` match their stage-qualified forms such as `failed:chunking`
- `--stats`: show counts grouped by status instead of listing jobs
- `--retry`: retry every failed job instead of listing jobs

Examples:

```bash
venv/bin/rag jobs list --stats
venv/bin/rag jobs list --retry
```

### Inspect a job

```bash
venv/bin/rag jobs status <job_id>
```

This prints the current stage, timestamps, stage log, and error detail when present.

### Retry a failed job

```bash
venv/bin/rag jobs retry <job_id>
venv/bin/rag jobs retry <job_id> --from-stage chunking
```

Options:

- `--from-stage TEXT`: restart from one of `parsing`, `profiling`, `chunking`, `validation`, `embedding`, `graph_extraction`, `graph_linking`, or `insight_extraction`

Retry cleanup is stage-aware. Earlier stage retries remove downstream artifacts before the job is re-queued.

### Cancel a queued or running job

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
venv/bin/rag source <source_id>
```

- `sources get` prints metadata plus a markdown preview
- `source` prints the full stored markdown only

List insights for a source:

```bash
venv/bin/rag sources insights <source_id>
```

Return source IDs for the last N sources, or for sources since a date:

```bash
venv/bin/rag sources last 5
venv/bin/rag sources last 2026-01-01
```

Delete a source:

```bash
venv/bin/rag sources delete <source_id>
venv/bin/rag sources delete <source_id> --hard
```

- soft delete marks the source deleted
- hard delete removes insight links, orphan insights, `entities`, `chunks`, `jobs`, `sources`, graph nodes, and the stored file tree

## Remediation

Backfill insight extraction for sources that already have chunks but no `chunk_insights` rows:

```bash
python scripts/remediate_insights.py --batch-size 10
```

## Search

Search is the lightweight retrieval surface. It runs hybrid chunk search and returns ranked chunk matches only. It does not perform graph expansion, related-chunk aggregation, or answer synthesis.

On the default `english` text-search config, sparse search uses a Postgres GIN full-text index. If you change `RETRIEVAL_TEXT_SEARCH_CONFIG`, recreate the sparse-search index to match the new config or Postgres will fall back to slower scans.

### CLI

Syntax:

```bash
venv/bin/rag search QUERY [OPTIONS]
```

Arguments:

- `QUERY`: free-text search query

Options:

- `--limit, -n INTEGER`: maximum number of results to return
- `--min-score FLOAT`: minimum score threshold applied after ranking

Examples:

```bash
venv/bin/rag search "graph storage" 
venv/bin/rag search "customer rollout" --limit 5
venv/bin/rag search "quarterly roadmap" --limit 20 --min-score 0.8
```

Response shape:

```json
[
  {
    "score": 0.93,
    "chunk": "Chunk text...",
    "chunk_id": "chunk-uuid",
    "source_id": "source-uuid",
    "source_path": "data/documents/source-uuid/1/original_report.pdf",
    "source_metadata": {
      "kind": "report"
    }
  }
]
```

### REST API

Endpoint:

```text
POST /api/search
```

Request body:

```json
{
  "query": "quarterly roadmap",
  "limit": 10,
  "min_score": 0.7
}
```

Example:

```bash
curl -X POST http://localhost:8000/api/search \
  -H 'content-type: application/json' \
  -d '{"query":"quarterly roadmap","limit":5,"min_score":0.75}'
```

## Community

`rag community retrieve` resolves source scope with a lightweight retrieval-stage pass before running community graphing. It does not run the full graph-expanded retrieval result assembly just to collect source IDs, which keeps community detection substantially faster on retrieval-scoped runs.

Response body:

```json
{
  "results": [
    {
      "score": 0.93,
      "chunk": "Chunk text...",
      "chunk_id": "chunk-uuid",
      "source_id": "source-uuid",
      "source_path": "data/documents/source-uuid/1/original_report.pdf",
      "source_metadata": {
        "kind": "report"
      }
    }
  ]
}
```

## Retrieval

Retrieval is the graph-aware query pipeline. It:

1. generates query variants
2. runs first-stage hybrid search
3. fuses and reranks candidates
4. expands selected seeds through graph evidence
5. falls back to same-source neighbors when graph expansion has no non-seed chunk evidence
6. reranks and returns root results plus related supporting chunks

### CLI

Syntax:

```bash
venv/bin/rag retrieve QUERY [OPTIONS]
```

Arguments:

- `QUERY`: natural-language query

Options:

- `--source-id TEXT`: restrict retrieval to one or more source IDs
- `--filter key=value`: restrict retrieval by source metadata; repeatable
- `--seed-count INTEGER`: override the number of seeds expanded
- `--result-count INTEGER`: override the number of final root results
- `--rrf-k INTEGER`: override reciprocal rank fusion `k`
- `--entity-confidence-threshold FLOAT`: override graph relationship confidence threshold
- `--first-hop-similarity-threshold FLOAT`: override first-hop chunk similarity threshold
- `--second-hop-similarity-threshold FLOAT`: override second-hop chunk similarity threshold
- `--trace`: print live trace lines to stdout before the final JSON block

Examples:

```bash
venv/bin/rag retrieve "What topics are covered in the ingested reports?"
venv/bin/rag retrieve "What changed?" --source-id <source_uuid> --source-id <source_uuid>
venv/bin/rag retrieve "What topics are covered?" --filter kind=report --filter domain=technical
venv/bin/rag retrieve "What topics are covered?" \
  --seed-count 3 \
  --result-count 2 \
  --rrf-k 40 \
  --entity-confidence-threshold 0.8 \
  --first-hop-similarity-threshold 0.6 \
  --second-hop-similarity-threshold 0.6
venv/bin/rag retrieve "What topics are covered in the ingested reports?" --trace
```

Trace note:

- `--trace` prints live activity first and the final response JSON last
- omit `--trace` if you need machine-only stdout

Response shape:

```json
{
  "retrieval_results": [
    {
      "score": 0.0,
      "chunk": "Root chunk text...",
      "chunk_id": "chunk-uuid",
      "source_id": "source-uuid",
      "source_path": "data/documents/source-uuid/1/original_report.pdf",
      "source_metadata": {
        "kind": "report"
      },
      "related": [
        {
          "score": 0.0,
          "chunk": "Supporting chunk text...",
          "chunk_id": "related-chunk-uuid",
          "source_id": "source-uuid",
          "source_path": "data/documents/source-uuid/1/original_report.pdf",
          "source_metadata": {
            "kind": "report"
          },
          "metadata": {
            "path": "graph:first_hop"
          }
        }
      ]
    }
  ]
}
```

### REST API

Endpoint:

```text
POST /api/retrieve
```

Request body:

```json
{
  "query": "What changed?",
  "source_ids": ["source-uuid-1", "source-uuid-2"],
  "filters": {
    "kind": "report"
  },
  "seed_count": 3,
  "result_count": 2,
  "rrf_k": 40,
  "entity_confidence_threshold": 0.8,
  "first_hop_similarity_threshold": 0.6,
  "second_hop_similarity_threshold": 0.6,
  "trace": false
}
```

Example:

```bash
curl -X POST http://localhost:8000/api/retrieve \
  -H 'content-type: application/json' \
  -d '{
    "query": "What changed?",
    "filters": {"kind": "report"},
    "seed_count": 3,
    "result_count": 2
  }'
```

The REST API returns the same JSON structure as the CLI, without trace lines.

## Community

Community analysis groups connected entities into communities and returns representative chunks for each group. Scope can come from:

- explicit source IDs
- sources matched by search
- sources matched by retrieval

Optional summarization adds an LLM-written summary per community.

Cross-source community detection uses pgvector ANN queries against the `entities_embedding_hnsw_idx` index and is no longer disabled at large entity scopes. Tuning `--semantic-threshold` lower (e.g. 0.75–0.80) helps surface looser cross-source clusters. If the HNSW index is absent on older deployments, pgvector falls back to a sequential scan with correct but slower results; run `CREATE INDEX entities_embedding_hnsw_idx ON entities USING hnsw (embedding vector_cosine_ops)` to restore index performance.

### CLI

The community surface has three subcommands.

#### `rag community ids`

Use explicit source IDs as the scope.

```bash
venv/bin/rag community ids <source_id> <source_id> [OPTIONS]
```

Options:

- `--semantic-threshold FLOAT`: override entity-to-entity semantic edge threshold
- `--cutoff FLOAT`: minimum chunk score kept in each community
- `--min-community-size INTEGER`: minimum entities per community
- `--top-k INTEGER`: max representative chunks per community
- `--summarize TEXT`: model name used to summarize each community
- `--cross-source-top-k INTEGER`: max cross-source ANN neighbors fetched per entity
- `--max-cross-source-queries INTEGER`: hard cap on per-entity ANN queries

Example:

```bash
venv/bin/rag community ids <source_id> <source_id> --top-k 3 --summarize google/gemini-2.5-flash-lite
```

#### `rag community search`

Use search criteria to select sources first.

```bash
venv/bin/rag community search CRITERION... [OPTIONS]
```

Arguments:

- `CRITERION...`: one or more search strings

Options:

- `--filter key=value`: metadata filter; repeatable
- `--limit INTEGER`: max search results per criterion
- `--min-score FLOAT`: minimum search score
- `--semantic-threshold FLOAT`
- `--cutoff FLOAT`
- `--min-community-size INTEGER`
- `--top-k INTEGER`
- `--summarize TEXT`
- `--cross-source-top-k INTEGER`
- `--max-cross-source-queries INTEGER`

Example:

```bash
venv/bin/rag community search "quarterly roadmap" "launch plan" \
  --filter kind=report \
  --limit 5 \
  --min-score 0.75 \
  --top-k 3
```

#### `rag community retrieve`

Use retrieval criteria to select sources first.

```bash
venv/bin/rag community retrieve CRITERION... [OPTIONS]
```

Arguments:

- `CRITERION...`: one or more retrieval query strings

Options:

- `--filter key=value`: metadata filter; repeatable
- `--seed-count INTEGER`
- `--result-count INTEGER`
- `--rrf-k INTEGER`
- `--entity-confidence-threshold FLOAT`
- `--first-hop-similarity-threshold FLOAT`
- `--second-hop-similarity-threshold FLOAT`
- `--trace`
- `--semantic-threshold FLOAT`
- `--cutoff FLOAT`
- `--min-community-size INTEGER`
- `--top-k INTEGER`
- `--summarize TEXT`
- `--cross-source-top-k INTEGER`
- `--max-cross-source-queries INTEGER`

Example:

```bash
venv/bin/rag community retrieve "What changed?" \
  --filter kind=report \
  --seed-count 3 \
  --result-count 2 \
  --top-k 3 \
  --summarize google/gemini-2.5-flash-lite
```

Response shape:

```json
{
  "metadata": {
    "scope_mode": "search",
    "source_count": 2,
    "sources_excluded": [],
    "parameters": {
      "semantic_threshold": 0.85,
      "source_cooc_weight": 0.1,
      "cutoff": 0.5,
      "min_community_size": 3,
      "top_k_chunks": 5,
      "cross_source_top_k": 10,
      "max_cross_source_queries": 5000
    }
  },
  "communities": [
    {
      "community_id": "0",
      "is_cross_source": true,
      "entity_count": 4,
      "entities": [
        {
          "entity_id": "entity-uuid",
          "canonical_name": "OpenRouter",
          "entity_type": "ORG"
        }
      ],
      "contributing_sources": [
        {
          "source_id": "source-uuid",
          "source_name": "Quarterly Report"
        }
      ],
      "chunks": [
        {
          "chunk_id": "chunk-uuid",
          "source_id": "source-uuid",
          "source_name": "Quarterly Report",
          "entity_overlap_count": 2,
          "score": 0.81,
          "content": "Representative chunk text..."
        }
      ],
      "summary": "Optional community summary."
    }
  ]
}
```

### REST API

Endpoint:

```text
POST /api/community
```

Request body:

```json
{
  "scope_mode": "search",
  "source_ids": [],
  "criteria": ["quarterly roadmap"],
  "filters": {
    "kind": "report"
  },
  "search_options": {
    "limit": 5,
    "min_score": 0.75
  },
  "retrieve_options": {
    "seed_count": null,
    "result_count": null,
    "rrf_k": null,
    "entity_confidence_threshold": null,
    "first_hop_similarity_threshold": null,
    "second_hop_similarity_threshold": null,
    "trace": false
  },
  "community_options": {
    "semantic_threshold": 0.85,
    "cutoff": 0.5,
    "min_community_size": 3,
    "top_k_chunks": 5,
    "cross_source_top_k": null,
    "max_cross_source_queries": null
  },
  "summarize_model": null
}
```

Notes:

- `scope_mode` must be one of `ids`, `search`, or `retrieve`
- use `source_ids` only with `scope_mode: "ids"`
- use `criteria` with `search` or `retrieve`
- the response body matches the CLI JSON

## Prompts

Prompt templates live in [src/rag/prompts/__init__.py](/Users/estebanf/development/knowledge-graphrag/src/rag/prompts/__init__.py).

That module is the canonical place to maintain prompt text for:

- document profiling
- chunk validation
- entity extraction
- relationship extraction
- proposition decomposition
- answer generation
- community summarization
- retrieval query variants
- graph-stage entity selection and entity query generation

Maintenance rules:

- add or update prompt strings in `src/rag/prompts/__init__.py`
- keep prompt call sites formatting values through `.format(...)` where needed
- if prompt behavior changes, update or add tests in [tests/test_prompts.py](/Users/estebanf/development/knowledge-graphrag/tests/test_prompts.py)
- if you need a deployment-specific community summary prompt without changing code, set `COMMUNITY_SUMMARIZATION_PROMPT`

## Migrations and Existing Databases

If your local database predates the current code, apply the repo migrations in `scripts/migrate/`.

The migrations most likely to matter for current code are:

- `scripts/migrate/001_add_markdown_content.sql`
- `scripts/migrate/002_update_vector_dimensions.sql`
- `scripts/migrate/004_job_improvements.sql`
- `scripts/migrate/006_search_performance_indexes.sql`
- `scripts/migrate/007_insights.sql`

Example:

```bash
docker compose exec -T postgres psql -U rag -d rag -f scripts/migrate/004_job_improvements.sql
docker compose exec -T postgres psql -U rag -d rag -f scripts/migrate/006_search_performance_indexes.sql
docker compose exec -T postgres psql -U rag -d rag -f scripts/migrate/007_insights.sql
```

## Verification

Targeted retrieval and CLI tests:

```bash
pytest -q tests/test_retrieval.py tests/test_cli_retrieve.py tests/test_config.py
```

Search, community, and API coverage:

```bash
pytest -q tests/test_cli_search.py tests/test_cli_community.py tests/test_api.py tests/test_api_community.py
```

Fast ingestion-oriented suite:

```bash
pytest -q tests/test_cli_jobs.py tests/test_job_lifecycle.py tests/test_worker.py tests/test_ingestion_submit.py tests/test_parser.py tests/test_storage.py tests/test_cli_ingest.py tests/test_chunking.py tests/test_observability.py tests/test_cli_health.py tests/test_profiling.py tests/test_chunk_validation.py tests/test_embedding.py
```

Prompt regression coverage:

```bash
pytest -q tests/test_prompts.py
```

Insight extraction coverage:

```bash
pytest -q tests/test_insight_extraction.py tests/test_config.py tests/test_prompts.py tests/test_cli_sources.py
```

Full suite:

```bash
pytest -q
```

Live integration:

```bash
pytest -q tests/test_ingestion.py
venv/bin/rag search "What topics are covered in the ingested reports?" --limit 1
venv/bin/rag retrieve "What topics are covered in the ingested reports?" --result-count 1 --seed-count 1 --trace
```

## Troubleshooting

- `rag health` fails:
  - confirm `docker compose ps`
  - confirm `POSTGRES_URL` and `MEMGRAPH_URL`
- Search or retrieval fails before ranking:
  - confirm `OPENROUTER_API_KEY`
  - confirm embedding and retrieval model env vars are set
- Retrieval returns roots with empty `related`:
  - some chunks still have no useful linked graph evidence
  - retrieval may still return the reranked root chunk alone
- Markdown images are not described:
  - only local relative image paths are copied and described
  - remote URLs, `data:` URLs, and missing files are left untouched
- Hard delete breaks:
  - dependent rows must be removed in this order: `entities`, `chunks`, `jobs`, `sources`
