BEGIN;

CREATE INDEX IF NOT EXISTS chunks_source_chunk_index_idx
  ON chunks(source_id, chunk_index) WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
  ON chunks USING gin (to_tsvector('english', coalesce(content, '')))
  WHERE deleted_at IS NULL AND embedding IS NOT NULL;

COMMIT;
