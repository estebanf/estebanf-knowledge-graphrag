-- Migration 013: research workspace. Idempotent: safe to re-run.
--
-- Adds:
--   1. Binary-quantized HNSW index on entities.embedding (vector(4096)).
--      pgvector 0.8.x cannot build a standard HNSW index on vector(4096),
--      but binary_quantize() + bit_hamming_ops is indexable and serves as a
--      fast prefilter. Community edge computation uses this index with the
--      prefilter-then-full-precision-rerank shape already proven on chunks
--      and insights (migration 009).
--   2. New tables for the research workspace (KTD2, KTD4):
--      community_runs, theme_reports, saved_answers, working_sets,
--      entity_semantic_edges.

-- ---------------------------------------------------------------------------
-- 1. Binary-quantized HNSW prefilter index on entities
-- ---------------------------------------------------------------------------

CREATE INDEX CONCURRENTLY IF NOT EXISTS entities_embedding_binary_hnsw_idx
  ON entities USING hnsw ((binary_quantize(embedding)::bit(4096)) bit_hamming_ops)
  WHERE embedding IS NOT NULL;

-- ---------------------------------------------------------------------------
-- 2. Community runs
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

-- ---------------------------------------------------------------------------
-- 3. Theme reports
-- ---------------------------------------------------------------------------

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

-- ---------------------------------------------------------------------------
-- 4. Saved answers
-- ---------------------------------------------------------------------------

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

-- ---------------------------------------------------------------------------
-- 5. Working sets
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS working_sets (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name            text NOT NULL UNIQUE,
  source_ids      jsonb NOT NULL DEFAULT '[]',
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS working_sets_name_idx ON working_sets(name);

-- ---------------------------------------------------------------------------
-- 6. Entity semantic edge cache
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entity_semantic_edges (
  entity_a    uuid NOT NULL,
  entity_b    uuid NOT NULL,
  similarity  real NOT NULL,
  computed_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (entity_a, entity_b)
);

CREATE INDEX IF NOT EXISTS entity_semantic_edges_entity_b_idx ON entity_semantic_edges(entity_b);
