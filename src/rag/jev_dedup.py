"""Jev confirmation of merge candidates (weekly maintenance, Jev phases only).

Merges are irreversible, so candidate pairs found by the existing cheap
heuristics (fuzzy name + embedding cosine for entities, embedding cosine for
insights) are confirmed by Jev before anything is merged. ONE System One call
per pair, with a 3-level Score following TypeSafe's entity-alignment recipe:

- level 0: two different things / different claims -> dropped;
- level 1: closely related, may or may not be the same -> review table;
- level 2: one and the same / the same claim stated differently -> merge.

The probability-weighted score is routed with cut points 0.5 and 1.5
(`REVIEW_CUT`, `MERGE_CUT`), i.e. rounding to the nearest level. Only the
high-confidence band auto-merges; the middle band is written to
`entity_merge_review` / `insight_pair_review` for a human to read.

Failure policy: a pair whose Jev call failed (or whose answer is unusable) is
**unconfirmed** and never merged. A missing `OPENROUTER_API_KEY` short-circuits
the whole batch to "error" without any call, and a run of consecutive fully
failed batches (no credits, outage) stops the remaining calls, mirroring
`rag.entity_audit`.

This module only judges and writes review rows; it never merges. Callers
(`scripts/merge_semantic_duplicates.py`, `scripts/weekly_maintenance.py`) own
the merge and the dry-run/--execute gating. Judging calls Jev even in a
dry-run (that is how a dry-run can report proposals) and therefore spends
credits, but it writes nothing to the corpus.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag import jev
from rag.config import settings

log = logging.getLogger(__name__)

# Score routing cut points (levels 0/1/2, rounded to the nearest level).
MERGE_CUT = 1.5
REVIEW_CUT = 0.5

# Pairs per `system_one_many` batch; also the granularity of the abort check.
BATCH_SIZE = 100
# Stop calling Jev after this many consecutive batches in which every call failed.
MAX_CONSECUTIVE_FAILED_BATCHES = 2

ALIAS_CAP = 10  # aliases per entity in the Jev state
INSIGHT_CHARS = 1500  # characters per insight in the Jev state

ENTITY_ALIGNMENT_LEVELS: list[str] = [
    "Two different things.",
    "Closely related, and may or may not be the same thing.",
    "One and the same thing.",
]
INSIGHT_ALIGNMENT_LEVELS: list[str] = [
    "Different claims.",
    "Related claims that differ in substance or scope.",
    "The same claim stated differently.",
]

ENTITY_ALIGNMENT_KEY = "alignment"
ENTITY_NAME_VARIANT_KEY = "same_name_variant"
ENTITY_SAME_TYPE_KEY = "same_type"

ROUTE_MERGE = "merge"
ROUTE_REVIEW = "review"
ROUTE_DROP = "drop"
ROUTE_ERROR = "error"


def _print(message: str) -> None:
    print(message, flush=True)


@dataclass
class Judgement:
    """Jev's verdict on one candidate pair (ids in normalized order)."""

    id_a: str
    id_b: str
    similarity: float
    route: str
    score: float | None = None
    reason: str = ""
    error: str = ""


@dataclass
class JudgeStats:
    calls: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    first_error: str = ""
    aborted: bool = False

    def describe(self) -> str:
        return (
            f"jev calls {self.calls}, errors {self.errors}, cost ${self.cost_usd:.4f}"
            + (f" (first error: {self.first_error})" if self.first_error else "")
        )


def route_score(score: float) -> str:
    """merge for >= 1.5, review for [0.5, 1.5), otherwise drop."""
    if score >= MERGE_CUT:
        return ROUTE_MERGE
    if score >= REVIEW_CUT:
        return ROUTE_REVIEW
    return ROUTE_DROP


def normalize_pair(id_a: str, id_b: str) -> tuple[str, str]:
    """Stable storage order for a pair (the review tables are UNIQUE on it)."""
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


def dedupe_pairs(pairs: Sequence[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    """Collapse (a, b) / (b, a) duplicates into one normalized pair, keeping the
    highest similarity. Probe-vs-corpus candidate search emits both directions
    whenever both members were probes."""
    best: dict[tuple[str, str], float] = {}
    for id_a, id_b, similarity in pairs:
        key = normalize_pair(id_a, id_b)
        if key[0] == key[1]:
            continue
        if similarity > best.get(key, float("-inf")):
            best[key] = similarity
    return [(a, b, sim) for (a, b), sim in best.items()]


# ---------------------------------------------------------------------------
# Question and state construction
# ---------------------------------------------------------------------------


def entity_summary(meta: dict) -> dict:
    return {
        "name": meta.get("canonical_name") or "",
        "type": meta.get("entity_type") or "",
        "aliases": list(meta.get("aliases") or [])[:ALIAS_CAP],
    }


def entity_pair_state(meta_a: dict, meta_b: dict) -> dict:
    return {"entity_a": entity_summary(meta_a), "entity_b": entity_summary(meta_b)}


def entity_pair_questions() -> dict[str, dict]:
    """The alignment Score plus two Nouls on which fields agree (the Nouls only
    enrich the review `reason`; routing uses the Score alone)."""
    return {
        ENTITY_ALIGNMENT_KEY: jev.score(
            "Do entity_a and entity_b refer to the same real-world entity? Compare their "
            "names, aliases and types.",
            ENTITY_ALIGNMENT_LEVELS,
        ),
        ENTITY_NAME_VARIANT_KEY: jev.noul(
            "Is the name of entity_a a spelling, spacing, casing, abbreviation or plural "
            "variant of the name of entity_b?",
            true="The names are variants of one name (for example 'Open AI' and 'OpenAI').",
            false="The names are different names, even if related.",
        ),
        ENTITY_SAME_TYPE_KEY: jev.noul(
            "Are entity_a and entity_b the same kind of thing (organization, person, "
            "product, concept, ...)?",
            true="Both are the same kind of thing.",
            false="They are different kinds of thing.",
        ),
    }


def insight_pair_state(text_a: str, text_b: str) -> dict:
    return {"insight_a": text_a[:INSIGHT_CHARS], "insight_b": text_b[:INSIGHT_CHARS]}


def insight_pair_questions() -> dict[str, dict]:
    return {
        ENTITY_ALIGNMENT_KEY: jev.score(
            "How do insight_a and insight_b relate? Compare the claims they make.",
            INSIGHT_ALIGNMENT_LEVELS,
        )
    }


# ---------------------------------------------------------------------------
# Running calls
# ---------------------------------------------------------------------------


def run_jev_items(
    items: list[tuple[dict, dict[str, dict]]],
    *,
    progress: Callable[[str], None] = _print,
    label: str = "jev",
) -> tuple[list, JudgeStats]:
    """Run `(state, questions)` items through `jev.system_one_many` in batches.

    Returns `(results, stats)` aligned with `items`; each result is a
    `JevResponse` or an Exception. Without an API key nothing is called and
    every result is an error; after `MAX_CONSECUTIVE_FAILED_BATCHES` consecutive
    batches with every call failing, the remaining items are errors too.
    """
    stats = JudgeStats()
    if not items:
        return [], stats
    if not settings.OPENROUTER_API_KEY:
        error = RuntimeError("OPENROUTER_API_KEY is not set (Jev runs on OpenRouter)")
        stats.errors = len(items)
        stats.first_error = str(error)
        stats.aborted = True
        progress(f"[{label}] {stats.first_error}; nothing will be merged or changed.")
        return [error] * len(items), stats

    results: list = []
    consecutive_failed = 0
    for start in range(0, len(items), BATCH_SIZE):
        if consecutive_failed >= MAX_CONSECUTIVE_FAILED_BATCHES:
            skipped = RuntimeError("skipped: earlier Jev batches all failed")
            remaining = len(items) - start
            results.extend([skipped] * remaining)
            stats.errors += remaining
            stats.aborted = True
            progress(
                f"[{label}] stopping: {consecutive_failed} consecutive batches had every Jev "
                f"call fail (first error: {stats.first_error}); {remaining} pair(s) left unconfirmed."
            )
            break
        chunk = items[start : start + BATCH_SIZE]
        batch = jev.system_one_many(chunk, with_usage=True)
        failed = 0
        for result in batch:
            stats.calls += 1
            if isinstance(result, Exception):
                failed += 1
                stats.errors += 1
                if not stats.first_error:
                    stats.first_error = str(result)
            else:
                stats.cost_usd += float((getattr(result, "usage", None) or {}).get("cost") or 0.0)
        results.extend(batch)
        consecutive_failed = consecutive_failed + 1 if failed == len(chunk) else 0
        progress(f"[{label}] judged {min(start + BATCH_SIZE, len(items))}/{len(items)} pair(s); {stats.describe()}")
    return results, stats


def _alignment(response) -> float:
    """Clamped 0..2 alignment score of a response; raises when unusable."""
    value = jev.score_of(response.answers[ENTITY_ALIGNMENT_KEY])
    return min(2.0, max(0.0, value))


def _noul(response, key: str) -> float | None:
    try:
        return round(jev.noul_p(response.answers[key]), 2)
    except (KeyError, TypeError, ValueError):
        return None


def _judgement(
    id_a: str, id_b: str, similarity: float, result, reason_for: Callable[[object, float], str]
) -> Judgement:
    if isinstance(result, Exception):
        return Judgement(id_a, id_b, similarity, ROUTE_ERROR, error=str(result))
    try:
        score = _alignment(result)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return Judgement(id_a, id_b, similarity, ROUTE_ERROR, error=f"unusable answer: {exc!r}")
    route = route_score(score)
    reason = reason_for(result, score) if route == ROUTE_REVIEW else ""
    return Judgement(id_a, id_b, similarity, route, score=round(score, 4), reason=reason)


def judge_entity_pairs(
    pairs: Sequence[tuple[str, str, float]],
    entity_meta: dict[str, dict],
    *,
    progress: Callable[[str], None] = _print,
) -> tuple[list[Judgement], JudgeStats]:
    """Jev verdict for each `(id_a, id_b, cosine)` entity pair. Pairs whose
    metadata is missing are errors (unconfirmed)."""
    normalized = dedupe_pairs(pairs)
    cosine_of = {normalize_pair(a, b): sim for a, b, sim in normalized}
    judged = [(a, b) for a, b, _ in normalized if a in entity_meta and b in entity_meta]
    questions = entity_pair_questions()
    items = [(entity_pair_state(entity_meta[a], entity_meta[b]), questions) for a, b in judged]
    results, stats = run_jev_items(items, progress=progress, label="entity merge")

    def reason_for(meta_a: dict, meta_b: dict, similarity: float):
        def build(result, _score: float) -> str:
            return (
                f"{meta_a.get('canonical_name')!r} vs {meta_b.get('canonical_name')!r}; "
                f"same_name_variant={_noul(result, ENTITY_NAME_VARIANT_KEY)}; "
                f"same_type={_noul(result, ENTITY_SAME_TYPE_KEY)}; cosine={similarity:.3f}"
            )

        return build

    judgements = [
        _judgement(a, b, cosine_of[(a, b)], result, reason_for(entity_meta[a], entity_meta[b], cosine_of[(a, b)]))
        for (a, b), result in zip(judged, results)
    ]
    for a, b, similarity in normalized:
        if a not in entity_meta or b not in entity_meta:
            judgements.append(
                Judgement(a, b, similarity, ROUTE_ERROR, error="missing entity metadata")
            )
    return judgements, stats


def judge_insight_pairs(
    pairs: Sequence[tuple[str, str, float]],
    texts: dict[str, str],
    *,
    progress: Callable[[str], None] = _print,
) -> tuple[list[Judgement], JudgeStats]:
    """Jev verdict for each `(id_a, id_b, cosine)` insight pair."""
    normalized = dedupe_pairs(pairs)
    judged = [(a, b, sim) for a, b, sim in normalized if a in texts and b in texts]
    questions = insight_pair_questions()
    items = [(insight_pair_state(texts[a], texts[b]), questions) for a, b, _ in judged]
    results, stats = run_jev_items(items, progress=progress, label="insight near-dup")
    judgements = [
        _judgement(a, b, sim, result, lambda _r, _s: "")
        for (a, b, sim), result in zip(judged, results)
    ]
    for a, b, sim in normalized:
        if a not in texts or b not in texts:
            judgements.append(Judgement(a, b, sim, ROUTE_ERROR, error="missing insight text"))
    return judgements, stats


def summarize(judgements: Sequence[Judgement]) -> dict[str, int]:
    counts = {ROUTE_MERGE: 0, ROUTE_REVIEW: 0, ROUTE_DROP: 0, ROUTE_ERROR: 0}
    for judgement in judgements:
        counts[judgement.route] += 1
    return counts


# ---------------------------------------------------------------------------
# Review tables (entity_merge_review / insight_pair_review, migration 015)
# ---------------------------------------------------------------------------

ENTITY_REVIEW_SQL = (
    "INSERT INTO entity_merge_review (entity_a, entity_b, score, reason) "
    "VALUES (%s::uuid, %s::uuid, %s, %s) "
    "ON CONFLICT (entity_a, entity_b) DO NOTHING"
)
INSIGHT_REVIEW_SQL = (
    "INSERT INTO insight_pair_review (insight_a, insight_b, cosine, score) "
    "VALUES (%s::uuid, %s::uuid, %s, %s) "
    "ON CONFLICT (insight_a, insight_b) DO NOTHING"
)


def fetch_reviewed_entity_pairs(conn) -> set[tuple[str, str]]:
    """Pairs that already have a review row, of any status. They are neither
    re-judged nor merged: a pair a human rejected must not be auto-merged on a
    later run, and a pending one is already waiting for a decision."""
    rows = conn.execute("SELECT entity_a::text, entity_b::text FROM entity_merge_review").fetchall()
    return {normalize_pair(r[0], r[1]) for r in rows}


def fetch_reviewed_insight_pairs(conn) -> set[tuple[str, str]]:
    rows = conn.execute("SELECT insight_a::text, insight_b::text FROM insight_pair_review").fetchall()
    return {normalize_pair(r[0], r[1]) for r in rows}


def insert_entity_reviews(conn, judgements: Sequence[Judgement]) -> int:
    """Insert review-band entity pairs (normalized order, ON CONFLICT DO NOTHING)
    and commit. Returns the number of rows submitted."""
    rows = [
        (*normalize_pair(j.id_a, j.id_b), j.score, j.reason)
        for j in judgements
        if j.route == ROUTE_REVIEW
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(ENTITY_REVIEW_SQL, rows)
    conn.commit()  # get_connection() is not autocommit
    return len(rows)


def insert_insight_reviews(conn, judgements: Sequence[Judgement]) -> int:
    """Insert review-band insight pairs (normalized order, ON CONFLICT DO
    NOTHING) and commit. Returns the number of rows submitted."""
    rows = [
        (*normalize_pair(j.id_a, j.id_b), j.similarity, j.score)
        for j in judgements
        if j.route == ROUTE_REVIEW
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(INSIGHT_REVIEW_SQL, rows)
    conn.commit()
    return len(rows)
