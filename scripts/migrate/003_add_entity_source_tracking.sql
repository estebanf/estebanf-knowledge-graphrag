-- Adds source_id to entities for per-source cleanup during job retry.
ALTER TABLE entities ADD COLUMN IF NOT EXISTS source_id UUID REFERENCES sources(id);
CREATE INDEX IF NOT EXISTS entities_source_id_idx ON entities(source_id);

-- Normalize any legacy stage_log arrays to empty objects
UPDATE jobs SET stage_log = '{}'::jsonb WHERE jsonb_typeof(stage_log) = 'array';
