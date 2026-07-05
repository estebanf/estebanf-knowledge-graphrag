-- Migration 014: allow theme_reports to track in-progress generation.
--
-- Theme report generation was previously a fully synchronous HTTP request;
-- the row was inserted with a 'completed' placeholder status and immediately
-- overwritten once analysis finished. Generation now runs in a background
-- thread so the API can return immediately and the frontend can poll for
-- progress, so the placeholder status needs to be a distinct, real state.

ALTER TABLE theme_reports DROP CONSTRAINT IF EXISTS theme_reports_status_check;
ALTER TABLE theme_reports ADD CONSTRAINT theme_reports_status_check
  CHECK (status IN ('generating', 'completed', 'partial', 'failed'));
ALTER TABLE theme_reports ALTER COLUMN status SET DEFAULT 'generating';
