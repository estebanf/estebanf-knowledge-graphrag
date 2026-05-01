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

-- NOTE: pgvector 0.8.x HNSW supports max 2000 dims; vector(4096) exceeds this limit.
-- The index below is kept as the intended target for when pgvector supports >2000 dims
-- or when embeddings are projected to a lower dimension.
-- Queries against insights.embedding use sequential scan at 4096 dims until pgvector supports >2000-dim HNSW.
-- Re-enable the index below when pgvector is upgraded.
-- CREATE INDEX IF NOT EXISTS insights_embedding_hnsw_idx
--   ON insights USING hnsw (embedding vector_cosine_ops)
--   WITH (m = 16, ef_construction = 64)
--   WHERE embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS chunk_insights_chunk_id_idx ON chunk_insights(chunk_id);
CREATE INDEX IF NOT EXISTS chunk_insights_insight_id_idx ON chunk_insights(insight_id);

COMMIT;
