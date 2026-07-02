-- Tighten autovacuum thresholds on entities so update/delete-heavy
-- entity-merge runs (scripts/merge_semantic_duplicates.py,
-- scripts/merge_duplicate_entities.py) get reclaimed automatically instead
-- of waiting on the default 20% table-size scale factor. Without this,
-- entities can accumulate multiple GB of dead TOAST before autovacuum
-- ever triggers on a large table.
--
-- Safe to run on any existing database; ALTER TABLE ... SET only changes
-- storage parameters, no data or index rebuild involved.

ALTER TABLE entities SET (
  autovacuum_vacuum_scale_factor = 0.02,
  autovacuum_vacuum_threshold = 500,
  autovacuum_analyze_scale_factor = 0.02,
  autovacuum_analyze_threshold = 500
);
