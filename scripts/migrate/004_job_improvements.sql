-- Add structured error storage for failed jobs
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS error_detail JSONB;

-- Replace non-unique MD5 index with a unique partial index
-- to enforce deduplication at the DB level (handles concurrent submissions)
DROP INDEX IF EXISTS sources_md5_idx;
CREATE UNIQUE INDEX IF NOT EXISTS sources_md5_unique_idx
    ON sources(md5) WHERE deleted_at IS NULL;

-- Index for stuck-job recovery query (only covers processing rows)
CREATE INDEX IF NOT EXISTS jobs_status_updated_idx
    ON jobs(status, updated_at) WHERE status LIKE 'processing:%';
