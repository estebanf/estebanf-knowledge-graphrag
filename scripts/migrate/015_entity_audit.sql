-- Jev-backed corpus audit tables (weekly maintenance "entity audit" phase).
--
-- entity_mention_audit: one row per (chunk, entity) MENTIONS edge that the Jev
-- audit has scored. `is_entity` is the Noul probability that the surface form is
-- a specific real-world named entity, `entity_type`/`type_confidence` the
-- averaged (order-reversed) Choice winner and its probability, `salience` a 0..1
-- normalized Score (how central the entity is to the chunk). The apply phase
-- writes `salience` into the Memgraph MENTIONS.confidence property and drops
-- mentions whose `is_entity` is below ENTITY_AUDIT_INVALID_THRESHOLD.
-- Deliberately NO foreign keys: entities and chunks are hard-deleted by merge
-- and source-delete paths, and queries join to them to ignore stale rows.
--
-- entity_merge_review / insight_pair_review: review queues for borderline merge
-- candidates (populated by a later maintenance phase; created here so there is a
-- single migration).

CREATE TABLE IF NOT EXISTS entity_mention_audit (
    chunk_id        uuid NOT NULL,
    entity_id       uuid NOT NULL,
    is_entity       double precision,
    entity_type     text,
    type_confidence double precision,
    salience        double precision,
    model           text,
    scored_at       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, entity_id)
);

CREATE INDEX IF NOT EXISTS entity_mention_audit_entity_id_idx ON entity_mention_audit(entity_id);
CREATE INDEX IF NOT EXISTS entity_mention_audit_scored_at_idx ON entity_mention_audit(scored_at);

CREATE TABLE IF NOT EXISTS entity_merge_review (
    id          bigserial PRIMARY KEY,
    entity_a    uuid NOT NULL,
    entity_b    uuid NOT NULL,
    score       double precision,
    reason      text,
    status      text NOT NULL DEFAULT 'pending',
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (entity_a, entity_b)
);

CREATE TABLE IF NOT EXISTS insight_pair_review (
    id          bigserial PRIMARY KEY,
    insight_a   uuid NOT NULL,
    insight_b   uuid NOT NULL,
    cosine      double precision,
    score       double precision,
    status      text NOT NULL DEFAULT 'pending',
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (insight_a, insight_b)
);
