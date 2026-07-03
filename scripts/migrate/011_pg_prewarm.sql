-- Install pg_prewarm so the binary-quantized HNSW vector indexes can be
-- loaded into shared_buffers on demand (the FastAPI lifespan calls
-- pg_prewarm() at startup) and, with pg_prewarm.autoprewarm=on preloaded in
-- docker-compose.yml, so Postgres re-warms them automatically after any
-- restart. Without a warm buffer cache the first dense-retrieval query after a
-- restart pages ~186MB of index from disk and blows the latency target.
--
-- Safe to run on any existing database; CREATE EXTENSION IF NOT EXISTS is
-- idempotent. The autoprewarm background worker additionally requires
-- shared_preload_libraries to include pg_prewarm (set in docker-compose.yml),
-- which needs a server restart the container already performs when recreated.

CREATE EXTENSION IF NOT EXISTS pg_prewarm;
