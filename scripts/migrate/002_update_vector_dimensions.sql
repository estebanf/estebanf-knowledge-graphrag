-- qwen/qwen3-embedding-8b outputs 4096 dimensions; schema was 1536
-- WARNING: drops all existing embedding values in chunks and entities tables.
-- Safe to run on fresh or dev databases. Re-running will fail if columns
-- already exist at vector(4096) — this migration is intentionally one-shot.
--
-- Note: pgvector 0.8.1 caps HNSW/IVFFlat indexes at 2000 dimensions.
-- HNSW indexes are omitted; queries use exact sequential scan at 4096 dims.
BEGIN;

DROP INDEX IF EXISTS chunks_embedding_hnsw_idx;
DROP INDEX IF EXISTS entities_embedding_hnsw_idx;

ALTER TABLE chunks DROP COLUMN IF EXISTS embedding;
ALTER TABLE chunks ADD COLUMN embedding vector(4096);

ALTER TABLE entities DROP COLUMN IF EXISTS embedding;
ALTER TABLE entities ADD COLUMN embedding vector(4096);

COMMIT;
