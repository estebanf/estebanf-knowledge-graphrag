-- Migration 008: auth (users + sessions) and worker process tracking.
-- Idempotent: safe to re-run.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS users (
  id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  username        text NOT NULL UNIQUE,
  password_hash   text NOT NULL,
  is_active       boolean NOT NULL DEFAULT true,
  created_at      timestamptz NOT NULL DEFAULT now(),
  last_login_at   timestamptz
);

CREATE TABLE IF NOT EXISTS user_sessions (
  id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at      timestamptz NOT NULL DEFAULT now(),
  expires_at      timestamptz NOT NULL,
  revoked_at      timestamptz,
  user_agent      text,
  ip              text
);

CREATE INDEX IF NOT EXISTS user_sessions_user_id_idx ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS user_sessions_expires_at_idx ON user_sessions(expires_at);

CREATE TABLE IF NOT EXISTS worker_processes (
  id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
  pid             integer,
  status          text NOT NULL CHECK (status IN ('starting','running','stopped','crashed')),
  started_at      timestamptz NOT NULL DEFAULT now(),
  stopped_at      timestamptz,
  exit_code       integer,
  log_path        text NOT NULL,
  host            text NOT NULL DEFAULT '',
  launched_by     uuid REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS worker_processes_status_idx ON worker_processes(status);

CREATE TABLE IF NOT EXISTS api_key_audit (
  id              bigserial PRIMARY KEY,
  key_id          text NOT NULL,
  route           text,
  ts              timestamptz NOT NULL DEFAULT now(),
  ip              text,
  user_agent      text
);

CREATE INDEX IF NOT EXISTS api_key_audit_key_id_idx ON api_key_audit(key_id);
CREATE INDEX IF NOT EXISTS api_key_audit_ts_idx ON api_key_audit(ts);
