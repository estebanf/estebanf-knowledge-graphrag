-- pgvector: dense vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- ParadeDB BM25 sparse retrieval
-- Extension name changed from pg_bm25 to pg_search in ParadeDB 0.8+
-- Try pg_search first; fall back to pg_bm25 if on an older image
DO $$
BEGIN
  BEGIN
    CREATE EXTENSION IF NOT EXISTS pg_search;
  EXCEPTION WHEN undefined_file THEN
    CREATE EXTENSION IF NOT EXISTS pg_bm25;
  END;
END;
$$;

-- UUID generation helpers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
