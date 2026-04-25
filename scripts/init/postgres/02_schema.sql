CREATE TABLE IF NOT EXISTS api_keys (
  id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  name            text NOT NULL UNIQUE,
  key_hash        text NOT NULL UNIQUE,
  prefix          text NOT NULL,
  created_at      timestamptz DEFAULT now(),
  last_used_at    timestamptz,
  revoked_at      timestamptz
);

CREATE TABLE IF NOT EXISTS sources (
  id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  name            text,
  file_name       text,
  file_type       text,
  storage_path    text,
  md5             text,
  version         int DEFAULT 1,
  metadata        jsonb,
  markdown_content text,
  created_at      timestamptz DEFAULT now(),
  deleted_at      timestamptz
);

CREATE TABLE IF NOT EXISTS jobs (
  id               uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  source_id        uuid REFERENCES sources(id),
  api_key_name     text,
  status           text,
  current_stage    text,
  stage_log        jsonb,
  retry_of         uuid REFERENCES jobs(id),
  retry_from_stage text,
  created_at       timestamptz DEFAULT now(),
  updated_at       timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
  id                uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  source_id         uuid REFERENCES sources(id),
  job_id            uuid REFERENCES jobs(id),
  content           text,
  token_count       int,
  chunk_index       int,
  parent_chunk_id   uuid REFERENCES chunks(id),
  chunking_strategy text,
  chunking_config   jsonb,
  metadata          jsonb,
  embedding         vector(1536),
  created_at        timestamptz DEFAULT now(),
  deleted_at        timestamptz
);

CREATE TABLE IF NOT EXISTS entities (
  id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  canonical_name  text,
  entity_type     text,
  aliases         text[],
  embedding       vector(1536),
  created_at      timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
  id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  api_key_name    text,
  action          text,
  resource_type   text,
  resource_id     uuid,
  metadata        jsonb,
  created_at      timestamptz DEFAULT now()
);

-- HNSW vector indexes — created here for fresh setups.
-- For large bulk imports, drop and recreate after loading for better performance.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
  ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS entities_embedding_hnsw_idx
  ON entities USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- Supporting indexes for common query patterns
CREATE INDEX IF NOT EXISTS chunks_source_id_idx ON chunks(source_id) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS chunks_source_chunk_index_idx
  ON chunks(source_id, chunk_index) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS chunks_job_id_idx ON chunks(job_id);
CREATE INDEX IF NOT EXISTS chunks_content_fts_idx
  ON chunks USING gin (to_tsvector('english', coalesce(content, '')))
  WHERE deleted_at IS NULL AND embedding IS NOT NULL;
CREATE INDEX IF NOT EXISTS sources_md5_idx ON sources(md5) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS jobs_source_id_idx ON jobs(source_id);
CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs(status);
