-- Binary-quantized ANN prefilter indexes for 4096-dimensional embeddings.
--
-- pgvector 0.8.x cannot build standard HNSW/IVFFlat indexes over vector(4096),
-- but it can index binary_quantize(embedding) and use that as a fast candidate
-- prefilter. Retrieval reranks the prefiltered rows with the original
-- full-precision cosine distance.

CREATE INDEX CONCURRENTLY IF NOT EXISTS chunks_embedding_binary_hnsw_idx
  ON chunks USING hnsw ((binary_quantize(embedding)::bit(4096)) bit_hamming_ops)
  WHERE deleted_at IS NULL AND embedding IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS insights_embedding_binary_hnsw_idx
  ON insights USING hnsw ((binary_quantize(embedding)::bit(4096)) bit_hamming_ops)
  WHERE embedding IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS insights_content_fts_idx
  ON insights USING gin (to_tsvector('english', coalesce(content, '')))
  WHERE embedding IS NOT NULL;
