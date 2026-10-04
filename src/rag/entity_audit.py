"""Jev-backed audit of chunk -> entity MENTIONS edges.

Two independent halves, both driven from `scripts/weekly_maintenance.py`:

**Scoring** (`score_mentions`) asks Jev about every `(chunk, entity)` mention.
ONE System One call per chunk, with the chunk text as the state and, per
mentioned entity, three questions answered in the same call:

- `e{i}_is_entity` (Noul): is the surface form a specific, real-world named
  entity as used in the passage (false for generic terms, roles, dates, phrases);
- `e{i}_type` (Choice over the 7 schema entity types), asked twice with the
  options in reversed order and averaged because Jev is biased toward the first
  option (`jev_profiling._choice_questions` / `average_choice`, reused);
- `e{i}_salience` (Score, 4 levels) normalised to 0..1 (`score / 3`).

Results go ONLY to the `entity_mention_audit` table (never to `entities`,
`chunks` or Memgraph), so scoring is safe while intake jobs are running. It is
resumable (already-scored pairs are skipped unless `rescore`) and commits after
every batch of chunks.

**Apply** (`apply_audit`) mutates the corpus from the audit rows and therefore
must only run under the weekly-maintenance lock with no active jobs:

a. delete `MENTIONS` edges whose `is_entity` is below
   `ENTITY_AUDIT_INVALID_THRESHOLD`, then delete entities left with no mention
   at all (Memgraph `DETACH DELETE` + Postgres `entities`);
b. retype entities whose *current* type is outside the schema when the majority
   audited type has a mean type probability >= `ENTITY_AUDIT_RETYPE_CONFIDENCE`;
c. write `salience` into `MENTIONS.confidence` of every audited, kept mention
   (which is what `entity_confidence_threshold` filters on at retrieval time).

Audit rows describe LIVE mentions: rows for dropped mentions are removed after
a successful drop. `entity_mention_audit` deliberately has no foreign keys
(entities/chunks are hard-deleted elsewhere); queries join to `entities` and
`chunks` so stale rows are ignored.

Every function takes the Postgres `conn` and the Memgraph `session` as
parameters, so tests pass mocks and only the script opens real connections.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from rag import jev
from rag.config import settings
from rag.graph_extraction import _ENTITY_TYPES
from rag.jev_profiling import JevProfilingError, _choice_questions, average_choice

log = logging.getLogger(__name__)

ENTITY_TYPES: tuple[str, ...] = tuple(_ENTITY_TYPES)

# Characters of the chunk sent as the Jev state (a chunk is normally far smaller).
STATE_CHARS = 4000
# Entities per Jev call; extra entities of the same chunk go in further calls.
MAX_ENTITIES_PER_CALL = 40
# Chunks fetched, scored and committed together.
BATCH_CHUNKS = 100
# Larger read-only batch used when only counting unscored mentions (dry-run).
COUNT_BATCH_CHUNKS = 500
# Rows per Memgraph write / Postgres page.
GRAPH_BATCH = 5000
# Entity ids per Memgraph mention-count query.
COUNT_ID_BATCH = 2000
# Stop scoring after this many consecutive batches in which every call failed
# (out of credits, missing key, outage): continuing would only burn time.
MAX_CONSECUTIVE_FAILED_BATCHES = 3

# Short on purpose: repeated twice per entity, up to 40 entities per call.
TYPE_DESCRIPTIONS: dict[str, str] = {
    "ORGANIZATION": "A company, agency, institution, team or other organized group.",
    "PERSON": "A specific named individual.",
    "POLICY": "A named policy, framework, standard, program or guideline.",
    "PRODUCT": "A named product, service, tool, platform or technology.",
    "REGULATION": "A law, regulation, directive or formal legal requirement.",
    "CONCEPT": "A distinct named concept, method, theory or technical idea.",
    "LOCATION": "A specific named place: country, city, region or site.",
}
assert tuple(TYPE_DESCRIPTIONS) == ENTITY_TYPES, "update TYPE_DESCRIPTIONS with the schema types"

SALIENCE_LEVELS: list[str] = [
    "Only mentioned in passing.",
    "Has a minor supporting role in the passage.",
    "Is important to the passage.",
    "Is a central subject of the passage.",
]
SALIENCE_MAX = len(SALIENCE_LEVELS) - 1

_NIL_UUID = "00000000-0000-0000-0000-000000000000"
_NAME_CHARS = 120


def _print(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Mention:
    """A live `(:Chunk)-[:MENTIONS]->(:Entity)` edge as read from Memgraph."""

    chunk_id: str
    entity_id: str
    name: str
    entity_type: str = ""


@dataclass
class MentionScore:
    chunk_id: str
    entity_id: str
    is_entity: float
    entity_type: str | None = None
    type_confidence: float | None = None
    salience: float | None = None


@dataclass
class ScoreStats:
    chunks_total: int = 0
    chunks_seen: int = 0
    chunks_with_pending: int = 0
    mentions_already_scored: int = 0
    mentions_scored: int = 0
    mentions_malformed: int = 0
    calls: int = 0
    call_errors: int = 0
    cost_usd: float = 0.0
    aborted: bool = False
    first_error: str = ""


@dataclass
class PendingBatch:
    chunk_ids: list[str]
    pending: dict[str, list[Mention]]
    mentions_total: int
    mentions_already_scored: int


@dataclass
class Retype:
    entity_id: str
    name: str
    old_type: str
    new_type: str
    votes: int
    mean_confidence: float


@dataclass
class ApplyPlan:
    since_days: int | None
    audited_mentions: int = 0
    drop_mentions: int = 0
    # entity_id -> number of its audited mentions that would be dropped
    drop_counts: dict[str, int] = field(default_factory=dict)
    drop_examples: list[tuple[str, str, int]] = field(default_factory=list)
    # Estimated (dry-run only): entities left with no MENTIONS after the drop.
    orphan_entity_ids: list[str] | None = None
    retypes: list[Retype] = field(default_factory=list)
    # salience level (0..3, -1 = missing) -> kept audited mentions
    salience_levels: dict[int, int] = field(default_factory=dict)
    unscored_mentions: int | None = None


@dataclass
class ApplyResult:
    edges_dropped: int = 0
    deleted_entity_ids: set[str] = field(default_factory=set)
    retyped: int = 0
    salience_written: int = 0
    audit_rows_removed: int = 0
    stale_pruned: int = 0


# ---------------------------------------------------------------------------
# Question construction and answer parsing
# ---------------------------------------------------------------------------


def _label(name: str) -> str:
    return '"' + name.replace('"', "'")[:_NAME_CHARS] + '"'


def build_questions(mentions: list[Mention]) -> dict[str, dict]:
    """Jev questions for one chunk's entities, keyed by position `i`:
    `e{i}_is_entity`, `e{i}_type__a` / `e{i}_type__b` (options reversed) and
    `e{i}_salience`. Positions, not names, are the keys so odd entity names
    can never collide or break the schema."""
    questions: dict[str, dict] = {}
    for i, mention in enumerate(mentions):
        label = _label(mention.name)
        questions[f"e{i}_is_entity"] = jev.noul(
            f"Is {label} a specific, real-world named entity as it is used in this passage: "
            "a particular organization, person, product, regulation, policy, place, or a "
            "distinct named concept or framework?",
            true=(
                "It names one particular real-world thing, such as a specific company, person, "
                "product, law, policy, place, or a distinct named concept or framework."
            ),
            false=(
                'It is a generic term (for example "AI", "data", "strategy", "customers"), a '
                "role or job title, a date or number, or a descriptive phrase rather than a name."
            ),
        )
        questions.update(
            _choice_questions(
                {
                    f"e{i}_type": (
                        f"What type of thing is {label} as it is used in this passage?",
                        TYPE_DESCRIPTIONS,
                    )
                }
            )
        )
        questions[f"e{i}_salience"] = jev.score(
            f"How prominent is {label} in this passage?", SALIENCE_LEVELS
        )
    return questions


def split_into_calls(
    mentions_by_chunk: dict[str, list[Mention]],
    max_entities: int = MAX_ENTITIES_PER_CALL,
) -> list[tuple[str, list[Mention]]]:
    """One `(chunk_id, entity group)` per Jev call: at most `max_entities`
    entities per call, extra entities of the same chunk go in further calls."""
    calls: list[tuple[str, list[Mention]]] = []
    for chunk_id, mentions in mentions_by_chunk.items():
        for start in range(0, len(mentions), max_entities):
            calls.append((chunk_id, mentions[start : start + max_entities]))
    return calls


def _normalise_salience(raw: float) -> float:
    return round(min(1.0, max(0.0, raw / SALIENCE_MAX)), 4)


def parse_answers(
    group: list[Mention], answers: dict[str, dict]
) -> tuple[list[MentionScore], int]:
    """Turn one call's answers into `MentionScore`s.

    Returns `(scores, malformed_count)`. An entity without a usable
    `is_entity` answer is counted malformed and skipped (it is retried on the
    next run); an unusable type or salience answer only leaves that field None.
    """
    scores: list[MentionScore] = []
    malformed = 0
    for i, mention in enumerate(group):
        try:
            is_entity = min(1.0, max(0.0, jev.noul_p(answers[f"e{i}_is_entity"])))
        except (KeyError, TypeError, ValueError):
            malformed += 1
            continue
        try:
            entity_type, type_confidence = average_choice(answers, f"e{i}_type", ENTITY_TYPES)
            type_confidence = round(type_confidence, 4)
        except (JevProfilingError, KeyError, TypeError, ValueError):
            entity_type, type_confidence = None, None
        try:
            salience = _normalise_salience(jev.score_of(answers[f"e{i}_salience"]))
        except (KeyError, TypeError, ValueError):
            salience = None
        scores.append(
            MentionScore(
                chunk_id=mention.chunk_id,
                entity_id=mention.entity_id,
                is_entity=round(is_entity, 4),
                entity_type=entity_type,
                type_confidence=type_confidence,
                salience=salience,
            )
        )
    return scores, malformed


# ---------------------------------------------------------------------------
# Reading scope: chunks (Postgres) and their mentions (Memgraph)
# ---------------------------------------------------------------------------


def fetch_scoped_chunk_ids(conn, since_days: int | None) -> list[str]:
    """Live chunk ids in scope, newest first. `since_days=None` is `--full`."""
    if since_days is None:
        rows = conn.execute(
            "SELECT id::text FROM chunks WHERE deleted_at IS NULL "
            "ORDER BY created_at DESC, id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id::text FROM chunks WHERE deleted_at IS NULL "
            "AND created_at >= now() - make_interval(days => %s) "
            "ORDER BY created_at DESC, id",
            (since_days,),
        ).fetchall()
    return [r[0] for r in rows]


def fetch_chunk_contents(conn, chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    rows = conn.execute(
        "SELECT id::text, content FROM chunks "
        "WHERE id = ANY(%s::uuid[]) AND deleted_at IS NULL",
        (chunk_ids,),
    ).fetchall()
    return {r[0]: r[1] or "" for r in rows}


def fetch_scored_pairs(conn, chunk_ids: list[str]) -> set[tuple[str, str]]:
    if not chunk_ids:
        return set()
    rows = conn.execute(
        "SELECT chunk_id::text, entity_id::text FROM entity_mention_audit "
        "WHERE chunk_id = ANY(%s::uuid[])",
        (chunk_ids,),
    ).fetchall()
    return {(r[0], r[1]) for r in rows}


def load_chunk_mentions(session, chunk_ids: list[str]) -> dict[str, list[Mention]]:
    """MENTIONS entities of each chunk, one UNWIND query for the whole batch,
    de-duplicated per entity id and ordered by name for stable question order."""
    if not chunk_ids:
        return {}
    records = session.run(
        """
        UNWIND $chunk_ids AS chunk_id
        MATCH (c:Chunk {chunk_id: chunk_id})-[:MENTIONS]->(e:Entity)
        RETURN c.chunk_id AS chunk_id,
               e.entity_id AS entity_id,
               e.canonical_name AS name,
               e.entity_type AS entity_type
        """,
        chunk_ids=chunk_ids,
    )
    by_chunk: dict[str, dict[str, Mention]] = {}
    for record in records:
        name = (record["name"] or "").strip()
        if not name:
            continue
        mention = Mention(
            chunk_id=record["chunk_id"],
            entity_id=record["entity_id"],
            name=name,
            entity_type=record["entity_type"] or "",
        )
        by_chunk.setdefault(mention.chunk_id, {}).setdefault(mention.entity_id, mention)
    return {
        chunk_id: sorted(entities.values(), key=lambda m: (m.name.lower(), m.entity_id))
        for chunk_id, entities in by_chunk.items()
    }


def iter_pending_batches(
    conn,
    session,
    chunk_ids: list[str],
    *,
    rescore: bool = False,
    batch_size: int = BATCH_CHUNKS,
) -> Iterator[PendingBatch]:
    """Walk `chunk_ids` in batches, yielding each chunk's mentions that still
    need scoring (all of them with `rescore`). Memory stays bounded by the
    batch, so `--full` never holds the whole corpus."""
    for start in range(0, len(chunk_ids), batch_size):
        batch_ids = chunk_ids[start : start + batch_size]
        mentions_by_chunk = load_chunk_mentions(session, batch_ids)
        scored = set() if rescore else fetch_scored_pairs(conn, list(mentions_by_chunk))
        pending: dict[str, list[Mention]] = {}
        total = 0
        already = 0
        for chunk_id, mentions in mentions_by_chunk.items():
            total += len(mentions)
            todo = [m for m in mentions if (m.chunk_id, m.entity_id) not in scored]
            already += len(mentions) - len(todo)
            if todo:
                pending[chunk_id] = todo
        yield PendingBatch(batch_ids, pending, total, already)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

UPSERT_AUDIT_SQL = """
    INSERT INTO entity_mention_audit
        (chunk_id, entity_id, is_entity, entity_type, type_confidence, salience, model)
    VALUES (%s::uuid, %s::uuid, %s, %s, %s, %s, %s)
    ON CONFLICT (chunk_id, entity_id) DO UPDATE SET
        is_entity = EXCLUDED.is_entity,
        entity_type = EXCLUDED.entity_type,
        type_confidence = EXCLUDED.type_confidence,
        salience = EXCLUDED.salience,
        model = EXCLUDED.model,
        scored_at = now()
"""


def upsert_scores(conn, scores: list[MentionScore], model: str) -> None:
    """Batch-upsert into `entity_mention_audit`. The caller commits."""
    if not scores:
        return
    rows = [
        (
            s.chunk_id,
            s.entity_id,
            s.is_entity,
            s.entity_type,
            s.type_confidence,
            s.salience,
            model,
        )
        for s in scores
    ]
    with conn.cursor() as cur:
        cur.executemany(UPSERT_AUDIT_SQL, rows)


def _format_stats(stats: ScoreStats) -> str:
    return (
        f"chunks {stats.chunks_seen}/{stats.chunks_total} | "
        f"mentions scored {stats.mentions_scored} "
        f"(already scored {stats.mentions_already_scored}, malformed {stats.mentions_malformed}) | "
        f"jev calls {stats.calls}, errors {stats.call_errors} | "
        f"cost ${stats.cost_usd:.4f}"
    )


def score_mentions(
    conn,
    session,
    since_days: int | None,
    *,
    rescore: bool = False,
    batch_size: int = BATCH_CHUNKS,
    max_workers: int | None = None,
    progress: Callable[[str], None] = _print,
) -> ScoreStats:
    """Score unscored mentions of the chunks in scope with Jev.

    Writes only to `entity_mention_audit`, committing after every batch so an
    interrupted run resumes where it stopped. A failed Jev call is logged and
    counted, never raised. Returns the run statistics.
    """
    chunk_ids = fetch_scoped_chunk_ids(conn, since_days)
    conn.commit()
    stats = ScoreStats(chunks_total=len(chunk_ids))
    scope = "all chunks (--full)" if since_days is None else f"chunks from the last {since_days} day(s)"
    progress(f"[entity audit] scoring: {len(chunk_ids)} {scope}; rescore={rescore}")

    consecutive_failed = 0
    for batch in iter_pending_batches(
        conn, session, chunk_ids, rescore=rescore, batch_size=batch_size
    ):
        stats.chunks_seen += len(batch.chunk_ids)
        stats.mentions_already_scored += batch.mentions_already_scored
        if not batch.pending:
            continue
        stats.chunks_with_pending += len(batch.pending)

        contents = fetch_chunk_contents(conn, list(batch.pending))
        units: list[tuple[str, list[Mention]]] = []
        items: list[tuple[str, dict[str, dict]]] = []
        for chunk_id, group in split_into_calls(batch.pending):
            content = contents.get(chunk_id)
            if not content or not content.strip():
                continue
            units.append((chunk_id, group))
            items.append((content[:STATE_CHARS], build_questions(group)))
        if not items:
            continue

        results = jev.system_one_many(items, max_workers=max_workers, with_usage=True)
        scores: list[MentionScore] = []
        batch_errors = 0
        for (chunk_id, group), result in zip(units, results):
            stats.calls += 1
            if isinstance(result, Exception):
                batch_errors += 1
                stats.call_errors += 1
                log.warning("Jev entity audit call failed for chunk %s: %s", chunk_id, result)
                if not stats.first_error:
                    stats.first_error = str(result)
                continue
            parsed, malformed = parse_answers(group, result.answers)
            scores.extend(parsed)
            stats.mentions_malformed += malformed
            stats.cost_usd += float((result.usage or {}).get("cost") or 0.0)

        upsert_scores(conn, scores, settings.JEV_MODEL)
        conn.commit()
        stats.mentions_scored += len(scores)
        progress("[entity audit] " + _format_stats(stats))

        if batch_errors == len(items):
            consecutive_failed += 1
            if consecutive_failed >= MAX_CONSECUTIVE_FAILED_BATCHES:
                stats.aborted = True
                progress(
                    f"[entity audit] stopping: {consecutive_failed} consecutive batches had "
                    f"every Jev call fail (first error: {stats.first_error}). "
                    "Already-scored pairs are kept; rerun to resume."
                )
                break
        else:
            consecutive_failed = 0

    progress("[entity audit] scoring done: " + _format_stats(stats))
    return stats


def count_unscored(
    conn,
    session,
    since_days: int | None,
    *,
    rescore: bool = False,
    batch_size: int = COUNT_BATCH_CHUNKS,
) -> tuple[int, int]:
    """Read-only: `(unscored_mentions, total_mentions)` for the chunks in scope."""
    chunk_ids = fetch_scoped_chunk_ids(conn, since_days)
    unscored = total = 0
    for batch in iter_pending_batches(
        conn, session, chunk_ids, rescore=rescore, batch_size=batch_size
    ):
        total += batch.mentions_total
        unscored += batch.mentions_total - batch.mentions_already_scored
    return unscored, total


# ---------------------------------------------------------------------------
# Apply: planning
# ---------------------------------------------------------------------------


def _scope(since_days: int | None) -> tuple[str, list]:
    """FROM/WHERE prefix for audit rows in scope, joined to `chunks` (scope) and
    `entities` (ignores stale rows). Callers append `AND ...` conditions."""
    base = (
        "FROM entity_mention_audit a "
        "JOIN chunks c ON c.id = a.chunk_id "
        "JOIN entities e ON e.id = a.entity_id "
    )
    if since_days is None:
        return base + "WHERE TRUE", []
    return base + "WHERE c.created_at >= now() - make_interval(days => %s)", [since_days]


def mention_counts(session, entity_ids: list[str]) -> dict[str, int]:
    """Current MENTIONS in-degree per entity id. An id with no Memgraph node is
    absent from the result (callers treat that as zero)."""
    counts: dict[str, int] = {}
    for start in range(0, len(entity_ids), COUNT_ID_BATCH):
        batch = entity_ids[start : start + COUNT_ID_BATCH]
        records = session.run(
            """
            UNWIND $ids AS id
            MATCH (e:Entity {entity_id: id})
            OPTIONAL MATCH (:Chunk)-[m:MENTIONS]->(e)
            RETURN id AS entity_id, count(m) AS mentions
            """,
            ids=batch,
        )
        for record in records:
            counts[record["entity_id"]] = int(record["mentions"])
    return counts


def _salience_level(salience: float | None) -> int:
    if salience is None:
        return -1
    return min(SALIENCE_MAX, max(0, round(salience * SALIENCE_MAX)))


def _plan_retypes(conn, since_days: int | None) -> list[Retype]:
    """Off-schema entities whose majority audited type is confident enough.

    Only mentions that survive the drop (`is_entity` >= invalid threshold) vote.
    The winner is the type with the most votes (ties: higher summed
    confidence); it is applied when the mean `type_confidence` of the mentions
    voting for it reaches `ENTITY_AUDIT_RETYPE_CONFIDENCE`.
    """
    scope_sql, scope_params = _scope(since_days)
    rows = conn.execute(
        f"""
        SELECT a.entity_id::text, e.canonical_name, e.entity_type, a.entity_type,
               count(*), sum(a.type_confidence)
        {scope_sql}
          AND a.is_entity >= %s
          AND a.entity_type = ANY(%s)
          AND a.type_confidence IS NOT NULL
          AND (e.entity_type IS NULL OR e.entity_type <> ALL(%s))
        GROUP BY a.entity_id, e.canonical_name, e.entity_type, a.entity_type
        """,
        (
            *scope_params,
            settings.ENTITY_AUDIT_INVALID_THRESHOLD,
            list(ENTITY_TYPES),
            list(ENTITY_TYPES),
        ),
    ).fetchall()

    by_entity: dict[str, dict] = {}
    for entity_id, name, old_type, voted_type, votes, conf_sum in rows:
        entry = by_entity.setdefault(
            entity_id, {"name": name or "", "old_type": old_type or "", "votes": {}}
        )
        entry["votes"][voted_type] = (int(votes), float(conf_sum or 0.0))

    retypes: list[Retype] = []
    for entity_id, entry in by_entity.items():
        winner, (votes, conf_sum) = max(
            entry["votes"].items(), key=lambda kv: (kv[1][0], kv[1][1])
        )
        mean = conf_sum / votes if votes else 0.0
        if mean >= settings.ENTITY_AUDIT_RETYPE_CONFIDENCE and winner != entry["old_type"]:
            retypes.append(
                Retype(entity_id, entry["name"], entry["old_type"], winner, votes, round(mean, 4))
            )
    retypes.sort(key=lambda r: (-r.votes, r.name.lower()))
    return retypes


def build_plan(
    conn,
    session,
    since_days: int | None,
    *,
    estimate_orphans: bool = True,
) -> ApplyPlan:
    """Everything `apply_audit` would do, computed from audit rows. Reads only.

    `estimate_orphans` asks Memgraph how many MENTIONS each affected entity has
    today and predicts which end up with none after the drop (dry-run report).
    """
    threshold = settings.ENTITY_AUDIT_INVALID_THRESHOLD
    scope_sql, scope_params = _scope(since_days)
    plan = ApplyPlan(since_days=since_days)

    plan.audited_mentions = int(
        conn.execute(f"SELECT count(*) {scope_sql}", tuple(scope_params)).fetchone()[0]
    )

    drop_rows = conn.execute(
        f"SELECT a.entity_id::text, count(*) {scope_sql} AND a.is_entity < %s "
        "GROUP BY a.entity_id",
        (*scope_params, threshold),
    ).fetchall()
    plan.drop_counts = {r[0]: int(r[1]) for r in drop_rows}
    plan.drop_mentions = sum(plan.drop_counts.values())

    if plan.drop_counts:
        example_rows = conn.execute(
            f"SELECT e.canonical_name, e.entity_type, count(*) {scope_sql} AND a.is_entity < %s "
            "GROUP BY e.id, e.canonical_name, e.entity_type "
            "ORDER BY count(*) DESC LIMIT 15",
            (*scope_params, threshold),
        ).fetchall()
        plan.drop_examples = [(r[0] or "", r[1] or "", int(r[2])) for r in example_rows]

    if estimate_orphans:
        ids = list(plan.drop_counts)
        counts = mention_counts(session, ids) if ids else {}
        plan.orphan_entity_ids = [
            eid for eid in ids if counts.get(eid, 0) - plan.drop_counts[eid] <= 0
        ]

    plan.retypes = _plan_retypes(conn, since_days)

    level_rows = conn.execute(
        f"""
        SELECT CASE WHEN a.salience IS NULL THEN -1
                    ELSE LEAST({SALIENCE_MAX}, GREATEST(0, round(a.salience * {SALIENCE_MAX})::int))
               END AS level,
               count(*)
        {scope_sql} AND a.is_entity >= %s
        GROUP BY 1 ORDER BY 1
        """,
        (*scope_params, threshold),
    ).fetchall()
    plan.salience_levels = {int(r[0]): int(r[1]) for r in level_rows}
    return plan


def format_plan(plan: ApplyPlan) -> list[str]:
    """Human-readable dry-run report lines for an `ApplyPlan`."""
    scope = "all chunks (--full)" if plan.since_days is None else f"chunks from the last {plan.since_days} day(s)"
    lines = [
        f"Entity audit apply (scope: {scope})",
        f"  audited mentions in scope: {plan.audited_mentions}",
        f"  mentions to drop (is_entity < {settings.ENTITY_AUDIT_INVALID_THRESHOLD}): "
        f"{plan.drop_mentions} across {len(plan.drop_counts)} entity/entities",
    ]
    for name, entity_type, count in plan.drop_examples:
        lines.append(f"    - {name!r} [{entity_type}] x{count}")
    if plan.orphan_entity_ids is not None:
        lines.append(
            f"  entities left with no mentions (would be deleted): {len(plan.orphan_entity_ids)}"
        )
    lines.append(
        f"  off-schema entities to retype (>= {settings.ENTITY_AUDIT_RETYPE_CONFIDENCE} mean type "
        f"confidence): {len(plan.retypes)}"
    )
    for retype in plan.retypes[:15]:
        lines.append(
            f"    - {retype.name!r} {retype.old_type or '(none)'} -> {retype.new_type} "
            f"({retype.votes} mention(s), mean {retype.mean_confidence})"
        )
    labels = {-1: "no salience", 0: "passing mention (0)", 1: "minor (1)", 2: "important (2)", 3: "central (3)"}
    distribution = ", ".join(
        f"{labels.get(level, level)}: {count}" for level, count in sorted(plan.salience_levels.items())
    )
    lines.append(
        "  salience written to MENTIONS.confidence for kept mentions: "
        + (distribution or "none")
    )
    if plan.unscored_mentions is not None:
        lines.append(f"  in-scope mentions not yet scored: {plan.unscored_mentions}")
    return lines


# ---------------------------------------------------------------------------
# Apply: mutations
# ---------------------------------------------------------------------------

DELETE_MENTIONS_CQL = """
    UNWIND $pairs AS p
    MATCH (c:Chunk {chunk_id: p.chunk_id})-[r:MENTIONS]->(e:Entity {entity_id: p.entity_id})
    DELETE r
"""
DELETE_ENTITIES_CQL = """
    UNWIND $ids AS id
    MATCH (e:Entity {entity_id: id})
    DETACH DELETE e
"""
RETYPE_ENTITIES_CQL = """
    UNWIND $rows AS r
    MATCH (e:Entity {entity_id: r.entity_id})
    SET e.entity_type = r.entity_type
"""
WRITE_SALIENCE_CQL = """
    UNWIND $rows AS r
    MATCH (c:Chunk {chunk_id: r.chunk_id})-[m:MENTIONS]->(e:Entity {entity_id: r.entity_id})
    SET m.confidence = r.salience
"""


def _iter_audit_pages(
    conn, since_days: int | None, extra_sql: str, extra_params: tuple, columns: str, page_size: int
) -> Iterator[list[tuple]]:
    """Keyset-paginate audit rows in scope on `(chunk_id, entity_id)` so
    `--full` never holds every row at once."""
    scope_sql, scope_params = _scope(since_days)
    cursor = (_NIL_UUID, _NIL_UUID)
    while True:
        rows = conn.execute(
            f"SELECT a.chunk_id::text, a.entity_id::text{columns} {scope_sql} {extra_sql} "
            "AND (a.chunk_id, a.entity_id) > (%s::uuid, %s::uuid) "
            "ORDER BY a.chunk_id, a.entity_id LIMIT %s",
            (*scope_params, *extra_params, *cursor, page_size),
        ).fetchall()
        if not rows:
            return
        yield rows
        if len(rows) < page_size:
            return
        cursor = (rows[-1][0], rows[-1][1])


def apply_audit(
    conn,
    session,
    since_days: int | None,
    *,
    page_size: int = GRAPH_BATCH,
    progress: Callable[[str], None] = _print,
) -> ApplyResult:
    """Apply the audit to the corpus (see module docstring). The caller must
    hold the maintenance lock with no active jobs.

    Order matters for crash safety: edges are dropped first, orphan entities
    are found from Memgraph's live in-degree (a missing node counts as zero),
    and the audit rows of dropped mentions are removed only AFTER the orphans
    are gone, so a re-run after an interruption sees the same drop set and
    finishes the job. Memgraph changes precede their Postgres counterpart (the
    `merge_*` convention); each Postgres step is committed explicitly.

    The returned `deleted_entity_ids` are for the caller to invalidate the
    `entity_semantic_edges` cache with.
    """
    threshold = settings.ENTITY_AUDIT_INVALID_THRESHOLD
    plan = build_plan(conn, session, since_days, estimate_orphans=False)
    result = ApplyResult()
    progress(
        f"[entity audit] applying: {plan.drop_mentions} mention(s) to drop, "
        f"{len(plan.retypes)} entity/entities to retype"
    )

    # a. drop invalid MENTIONS edges, then entities with no mention left.
    if plan.drop_mentions:
        for page in _iter_audit_pages(
            conn, since_days, "AND a.is_entity < %s", (threshold,), "", page_size
        ):
            session.run(
                DELETE_MENTIONS_CQL,
                pairs=[{"chunk_id": r[0], "entity_id": r[1]} for r in page],
            )
            result.edges_dropped += len(page)

        drop_ids = list(plan.drop_counts)
        counts = mention_counts(session, drop_ids)
        orphan_ids = [eid for eid in drop_ids if counts.get(eid, 0) == 0]
        for start in range(0, len(orphan_ids), page_size):
            session.run(DELETE_ENTITIES_CQL, ids=orphan_ids[start : start + page_size])
        if orphan_ids:
            conn.execute("DELETE FROM entities WHERE id = ANY(%s::uuid[])", (orphan_ids,))
            conn.execute(
                "DELETE FROM entity_semantic_edges "
                "WHERE entity_a = ANY(%s::uuid[]) OR entity_b = ANY(%s::uuid[])",
                (orphan_ids, orphan_ids),
            )
        result.deleted_entity_ids = set(orphan_ids)

        if since_days is None:
            removed = conn.execute(
                "DELETE FROM entity_mention_audit WHERE is_entity < %s", (threshold,)
            ).rowcount
        else:
            removed = conn.execute(
                "DELETE FROM entity_mention_audit a USING chunks c "
                "WHERE c.id = a.chunk_id AND a.is_entity < %s "
                "AND c.created_at >= now() - make_interval(days => %s)",
                (threshold, since_days),
            ).rowcount
        result.audit_rows_removed = removed or 0
        conn.commit()
        progress(
            f"[entity audit] dropped {result.edges_dropped} mention edge(s); "
            f"deleted {len(orphan_ids)} entity/entities left with no mentions"
        )

    # b. retype off-schema entities (never entities already deleted above).
    retypes = [r for r in plan.retypes if r.entity_id not in result.deleted_entity_ids]
    if retypes:
        rows = [{"entity_id": r.entity_id, "entity_type": r.new_type} for r in retypes]
        for start in range(0, len(rows), page_size):
            session.run(RETYPE_ENTITIES_CQL, rows=rows[start : start + page_size])
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE entities SET entity_type = %s WHERE id = %s::uuid",
                [(r.new_type, r.entity_id) for r in retypes],
            )
        conn.commit()
        result.retyped = len(retypes)
        progress(f"[entity audit] retyped {result.retyped} off-schema entity/entities")

    # c. write salience into MENTIONS.confidence for kept, audited mentions.
    for page in _iter_audit_pages(
        conn,
        since_days,
        "AND a.is_entity >= %s AND a.salience IS NOT NULL",
        (threshold,),
        ", a.salience",
        page_size,
    ):
        session.run(
            WRITE_SALIENCE_CQL,
            rows=[
                {"chunk_id": r[0], "entity_id": r[1], "salience": float(r[2])} for r in page
            ],
        )
        result.salience_written += len(page)
    conn.commit()
    progress(f"[entity audit] wrote salience to {result.salience_written} MENTIONS edge(s)")

    result.stale_pruned = prune_stale_audit_rows(conn)
    conn.commit()
    if result.stale_pruned:
        progress(f"[entity audit] pruned {result.stale_pruned} stale audit row(s)")
    return result


def prune_stale_audit_rows(conn) -> int:
    """Delete audit rows whose chunk or entity no longer exists (source deletes
    and entity merges hard-delete them; the table has no foreign keys)."""
    cursor = conn.execute(
        "DELETE FROM entity_mention_audit a "
        "WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.id = a.chunk_id) "
        "OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.id = a.entity_id)"
    )
    return cursor.rowcount or 0


def dry_run_report(
    conn,
    session,
    since_days: int | None,
    *,
    rescore: bool = False,
    progress: Callable[[str], None] = _print,
) -> ApplyPlan:
    """Print what `apply_audit` would do, from existing audit rows, plus how
    many in-scope mentions are still unscored. Reads only; writes nothing."""
    plan = build_plan(conn, session, since_days, estimate_orphans=True)
    unscored, _total = count_unscored(conn, session, since_days, rescore=rescore)
    plan.unscored_mentions = unscored
    for line in format_plan(plan):
        progress(line)
    return plan
