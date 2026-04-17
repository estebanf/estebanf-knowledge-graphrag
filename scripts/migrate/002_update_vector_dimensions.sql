-- qwen/qwen3-embedding-8b outputs 4096 dimensions; schema was 1536
-- NOTE: pgvector HNSW/IVFFlat indexes are limited to 2000 dimensions.
-- 4096-dim vectors use exact (sequential scan) cosine search instead.
BEGIN;

DROP INDEX IF EXISTS chunks_embedding_hnsw_idx;
DROP INDEX IF EXISTS entities_embedding_hnsw_idx;

ALTER TABLE chunks DROP COLUMN embedding;
ALTER TABLE chunks ADD COLUMN embedding vector(4096);

ALTER TABLE entities DROP COLUMN embedding;
ALTER TABLE entities ADD COLUMN embedding vector(4096);

COMMIT;
