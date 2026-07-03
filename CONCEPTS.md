# Concepts

Shared domain vocabulary for this project -- entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

## Retrieval

### Retrieval Pipeline
The graph-aware process that turns a user query into ranked chunk results and insight results by combining query variants, dense and sparse search, reranking, graph expansion, and final result shaping.

### Query Variant
An alternate form of the user's query used to broaden or sharpen first-stage retrieval before candidate fusion.

### HyDE Variant
A query variant written as a hypothetical answer-like passage so dense retrieval can match documents by likely answer semantics rather than only the original query wording.

### Expanded Variant
A query variant that adds likely context to a short or underspecified query; in this project it is gated because expansion can add fanout without improving longer intent-rich queries.

### Step-Back Variant
A broad abstraction of the user's query that was previously part of retrieval fanout but is no longer scheduled by default because it added latency with weak value.

### Dense Prefilter
An approximate candidate-selection stage that narrows high-dimensional vector search to a smaller pool before the system applies full-precision similarity scoring.

### Binary-Quantized HNSW
The indexed dense-prefilter strategy used for high-dimensional embeddings: embeddings are converted to binary form for HNSW candidate lookup, then original vectors are used for final reranking.

### Full-Precision Rerank
The stage that orders a prefiltered candidate pool with the original embedding values, preserving the final cosine-similarity signal after approximate candidate lookup.

### Chunk
A retrievable text unit from an ingested source. Chunks are the root evidence returned by retrieval and can be expanded through entity mentions and neighboring context.

### Insight
A distilled claim extracted from chunks and linked back to its source evidence. Insights have their own dense and sparse search path and can be expanded through related insights.

### Insight Expansion
The retrieval stage that starts from selected insight seeds, follows related-insight links, and optionally searches LLM-generated subqueries for second-hop supporting insights.

### Graph Expansion
The retrieval stage that expands selected chunk seeds through entity mentions, same-source fallback, and second-hop relationships to gather supporting chunks.

### Entity Mention
The graph relationship that connects a chunk to an entity it discusses. Retrieval treats this as the authoritative edge for chunk graph expansion.

## Storage

### Schema Reconciliation
The startup-time process that idempotently re-applies the graph store's declared index and constraint statements against the live instance, so a schema declaration can't silently drift out of sync with what the running database actually has applied.
