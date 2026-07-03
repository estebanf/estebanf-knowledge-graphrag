-- Pinned per-stage duration baseline for the stage-timing drift guardrail
-- (R11). Deliberately a single frozen snapshot per stage, not a moving/
-- rolling window: `rag jobs stats --set-baseline` overwrites the row for each
-- stage with the current window's per-stage median duration_ms, and the
-- worker (`src/rag/worker.py`) compares each newly completed job's stage
-- durations against this frozen value. A recomputed recent-window median
-- would drift upward together with a gradual corpus-coupled regression and
-- never fire — this is exactly the failure mode that let intake processing
-- creep from ~55s to an ~86-minute median between April and June without any
-- alert (see this plan's Problem Frame). Persisted here (not in-process) so
-- the baseline survives worker restarts.

CREATE TABLE IF NOT EXISTS stage_duration_baseline (
    stage text PRIMARY KEY,
    baseline_ms bigint NOT NULL,
    set_at timestamptz NOT NULL DEFAULT now()
);
