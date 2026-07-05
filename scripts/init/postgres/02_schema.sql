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
  embedding         vector(4096),
  created_at        timestamptz DEFAULT now(),
  deleted_at        timestamptz
);

CREATE TABLE IF NOT EXISTS entities (
  id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  canonical_name  text,
  entity_type     text,
  aliases         text[],
  embedding       vector(4096),
  created_at      timestamptz DEFAULT now()
) WITH (
  autovacuum_vacuum_scale_factor = 0.02,
  autovacuum_vacuum_threshold = 500,
  autovacuum_analyze_scale_factor = 0.02,
  autovacuum_analyze_threshold = 500
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

-- Binary-quantized HNSW prefilter indexes — created here for fresh setups.
-- pgvector 0.8.x cannot build standard HNSW indexes over vector(4096);
-- binary_quantize() + bit_hamming_ops is indexable and used as a fast
-- prefilter before full-precision reranking.
CREATE INDEX IF NOT EXISTS chunks_embedding_binary_hnsw_idx
  ON chunks USING hnsw ((binary_quantize(embedding)::bit(4096)) bit_hamming_ops)
  WHERE deleted_at IS NULL AND embedding IS NOT NULL;

CREATE INDEX IF NOT EXISTS entities_embedding_binary_hnsw_idx
  ON entities USING hnsw ((binary_quantize(embedding)::bit(4096)) bit_hamming_ops)
  WHERE embedding IS NOT NULL;

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

-- ---------------------------------------------------------------------------
-- Research workspace tables (migration 013)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS community_runs (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  status          text NOT NULL DEFAULT 'running'
                  CHECK (status IN ('running', 'completed', 'failed')),
  params          jsonb NOT NULL DEFAULT '{}',
  source_ids      jsonb NOT NULL DEFAULT '[]',
  stage_log       jsonb NOT NULL DEFAULT '[]',
  result          jsonb,
  error           text,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS community_runs_status_idx ON community_runs(status);
CREATE INDEX IF NOT EXISTS community_runs_created_at_idx ON community_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS theme_reports (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id              uuid NOT NULL REFERENCES community_runs(id) ON DELETE CASCADE,
  status              text NOT NULL DEFAULT 'completed'
                      CHECK (status IN ('completed', 'partial', 'failed')),
  failed_community_ids jsonb NOT NULL DEFAULT '[]',
  report              jsonb NOT NULL DEFAULT '{}',
  model               text NOT NULL DEFAULT '',
  created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS theme_reports_run_id_idx ON theme_reports(run_id);
CREATE INDEX IF NOT EXISTS theme_reports_created_at_idx ON theme_reports(created_at DESC);

CREATE TABLE IF NOT EXISTS saved_answers (
  id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  question          text NOT NULL,
  answer            text NOT NULL,
  model             text NOT NULL DEFAULT '',
  params            jsonb NOT NULL DEFAULT '{}',
  evidence_snapshot jsonb NOT NULL DEFAULT '[]',
  created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS saved_answers_created_at_idx ON saved_answers(created_at DESC);

CREATE TABLE IF NOT EXISTS working_sets (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name            text NOT NULL UNIQUE,
  source_ids      jsonb NOT NULL DEFAULT '[]',
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS working_sets_name_idx ON working_sets(name);

CREATE TABLE IF NOT EXISTS entity_semantic_edges (
  entity_a    uuid NOT NULL,
  entity_b    uuid NOT NULL,
  similarity  real NOT NULL,
  computed_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entity_a, entity_b)
);

CREATE INDEX IF NOT EXISTS entity_semantic_edges_entity_b_idx ON entity_semantic_edges(entity_b);
