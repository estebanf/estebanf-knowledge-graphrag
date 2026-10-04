"""Jev-backed canonical-name correction for hub entities (weekly maintenance).

Merging keeps the best-connected row, but historically the biggest hubs were
the *misspelled* ones ("Open AI", "Antropic", "Googles") while the correctly
spelled rows ("OpenAI", "Anthropic", "Google") lived on as separate entities,
because new extractions match by exact `canonical_name`. This module fixes the
name of the hubs:

1. **Targets**: entities with at least `ENTITY_NAME_REVIEW_MIN_ALIASES` aliases
   (the hubs), highest alias count first, limited to `--name-limit` per run.
2. **Judge** (`judge_names`): ONE Jev call per entity. State =
   `{"entity_type": ..., "names": [canonical_name, *aliases]}` (deduplicated,
   capped at `MAX_NAMES`); question = a Choice "which of these is the correct,
   standard real-world spelling?" over the candidate strings. Option keys are
   `n0..nK` (safe identifiers) with the string as the description. Jev is
   biased toward the first option, so the Choice is asked twice with the
   option order reversed (`jev_profiling._choice_questions` /
   `average_choice`). A rename is proposed only when both orders pick the SAME
   option, the averaged probability of that option is >=
   `ENTITY_NAME_MIN_CONFIDENCE`, and the winner differs from the current name.
3. **Plan** (`plan_actions`, reads only): if another entity row already has
   exactly the winning name (and a compatible type) the two are MERGED -- the
   survivor is the row with more MENTIONS edges, so edges move the least -- and
   the survivor ends up carrying the winning name; otherwise the entity is just
   renamed in place. A same-name row of an incompatible type blocks the rename
   (reported, never applied) so a cross-type name clash is not created.
4. **Apply** (`apply_actions`, `--execute` only): Memgraph first, then Postgres,
   committed per action. The merge itself is delegated to an injected
   `merge_fn` (weekly_maintenance passes the existing
   `merge_semantic_duplicates.merge_memgraph/merge_postgres`), so MENTIONS are
   repointed by exactly the code every other merge uses. The old canonical name
   and every other name go into the survivor's aliases. Actions that touch an
   entity already consumed earlier in the same run are skipped, and member rows
   are re-read just before each action.

Jev failures and low-confidence or order-disagreeing answers change nothing.
Known limits: the entity's embedding is not recomputed after a rename, and
insert-time dedup still matches `canonical_name` exactly, so a *new* extraction
of the old misspelling creates a fresh row until the next run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rag import jev
from rag.config import settings
from rag.entity_audit import attach_mention_counts
from rag.jev_dedup import run_jev_items
from rag.jev_profiling import JevProfilingError, _choice_questions, average_choice

log = logging.getLogger(__name__)

MAX_NAMES = 120  # candidate spellings sent per entity (Jev Choice max is 255)
NAME_FIELD = "name"  # questions are keyed name__a / name__b

NAME_INSTRUCTIONS = (
    "Which of these is the correct, standard real-world spelling of this entity's name? "
    "Judge spelling, spacing and capitalisation as the entity is normally written; ignore "
    "the order of the options."
)

STATUS_RENAME = "rename"
STATUS_KEEP = "keep"
STATUS_UNCERTAIN = "uncertain"
STATUS_ERROR = "error"

ACTION_RENAME = "rename"
ACTION_MERGE = "merge"

RENAME_NODE_CQL = "MATCH (e:Entity {entity_id: $entity_id}) SET e.canonical_name = $name"

_ENTITY_COLUMNS = (
    "id::text, canonical_name, entity_type, aliases, "
    "embedding IS NOT NULL AS has_embedding, created_at"
)


def _print(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NameJudgement:
    entity: dict
    names: list[str]
    status: str
    winner: str | None = None
    confidence: float | None = None
    error: str = ""


@dataclass
class RenameAction:
    kind: str  # ACTION_RENAME (in place) or ACTION_MERGE (into an existing row)
    old_name: str
    winner: str
    confidence: float
    members: list[dict]
    survivor_id: str
    dup_ids: list[str]


@dataclass
class BlockedRename:
    entity: dict
    winner: str
    reason: str


@dataclass
class NamePlan:
    actions: list[RenameAction] = field(default_factory=list)
    blocked: list[BlockedRename] = field(default_factory=list)


@dataclass
class NameStats:
    targets: int = 0
    jev_calls: int = 0
    jev_errors: int = 0
    cost_usd: float = 0.0
    kept: int = 0
    uncertain: int = 0


@dataclass
class ApplyResult:
    renamed: int = 0
    merged: int = 0  # rows merged away
    edges_repointed: int = 0
    skipped: int = 0
    # Every entity id involved in an applied action (survivors and merged-away),
    # for entity_semantic_edges cache invalidation.
    touched_ids: set[str] = field(default_factory=set)
    deleted_ids: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Targets and questions
# ---------------------------------------------------------------------------


def _entity_row(row) -> dict:
    return {
        "id": row[0],
        "canonical_name": row[1] or "",
        "entity_type": row[2] or "",
        "aliases": list(row[3] or []),
        "has_embedding": row[4],
        "created_at": row[5],
    }


def fetch_name_targets(conn, min_aliases: int, limit: int) -> list[dict]:
    """Hub entities (>= `min_aliases` aliases), most aliases first."""
    rows = conn.execute(
        f"SELECT {_ENTITY_COLUMNS} FROM entities "
        "WHERE cardinality(aliases) >= %s AND canonical_name IS NOT NULL AND canonical_name <> '' "
        "ORDER BY cardinality(aliases) DESC, canonical_name, id "
        "LIMIT %s",
        (min_aliases, limit),
    ).fetchall()
    return [_entity_row(r) for r in rows]


def candidate_names(canonical_name: str, aliases: Sequence[str], cap: int = MAX_NAMES) -> list[str]:
    """`[canonical_name, *aliases]` deduplicated (case-sensitive: casing is part
    of a spelling), blanks dropped, capped. The canonical name is always first."""
    seen: set[str] = set()
    names: list[str] = []
    for name in (canonical_name, *aliases):
        name = (name or "").strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names[:cap]


def option_keys(names: Sequence[str]) -> list[str]:
    return [f"n{i}" for i in range(len(names))]


def build_name_state(entity_type: str, names: Sequence[str]) -> dict:
    return {"entity_type": entity_type, "names": list(names)}


def build_name_questions(names: Sequence[str]) -> dict[str, dict]:
    """`name__a` (options n0..nK in order) and `name__b` (same options reversed).
    The key `nI` always denotes `names[I]`, whichever order it is shown in."""
    options = {key: name for key, name in zip(option_keys(names), names)}
    return _choice_questions({NAME_FIELD: (NAME_INSTRUCTIONS, options)})


def _order_vote(answer: dict, keys: Sequence[str]) -> tuple[str, dict[str, float]]:
    """argmax option and the probabilities of ONE order's answer."""
    probs = {k: p for k, p in jev.choice_probs(answer).items() if k in keys}
    if probs:
        winner = max(keys, key=lambda k: probs.get(k, -1.0))
        return winner, probs
    chosen = jev.choice_of(answer)
    if chosen not in keys:
        raise ValueError(f"unknown option {chosen!r}")
    return chosen, {chosen: 1.0}


def evaluate_answers(
    answers: dict[str, dict], names: Sequence[str], min_confidence: float
) -> tuple[str, str | None, float | None]:
    """`(status, winner, confidence)` for one entity's answers.

    `rename` needs: both orders usable, both argmaxes the same option, the
    averaged probability of that option >= `min_confidence`, and a winner
    different from `names[0]` (the current canonical name). `keep` is the same
    agreement on the current name. Anything else is `uncertain`.
    """
    keys = option_keys(names)
    try:
        winner_a, _ = _order_vote(answers[f"{NAME_FIELD}__a"], keys)
        winner_b, _ = _order_vote(answers[f"{NAME_FIELD}__b"], keys)
        averaged, probability = average_choice(answers, NAME_FIELD, keys)
    except (KeyError, TypeError, ValueError, JevProfilingError):
        return STATUS_UNCERTAIN, None, None
    if winner_a != winner_b or averaged != winner_a:
        return STATUS_UNCERTAIN, None, round(probability, 4)
    probability = round(probability, 4)
    if probability < min_confidence:
        return STATUS_UNCERTAIN, None, probability
    winner = names[int(winner_a[1:])]
    if winner == names[0]:
        return STATUS_KEEP, winner, probability
    return STATUS_RENAME, winner, probability


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------


def judge_names(
    targets: Sequence[dict],
    *,
    min_confidence: float | None = None,
    progress: Callable[[str], None] = _print,
) -> tuple[list[NameJudgement], NameStats]:
    """One Jev call per target with at least two distinct candidate names."""
    threshold = settings.ENTITY_NAME_MIN_CONFIDENCE if min_confidence is None else min_confidence
    stats = NameStats(targets=len(targets))
    judgements: list[NameJudgement] = []
    pending: list[tuple[dict, list[str]]] = []
    items: list[tuple[dict, dict[str, dict]]] = []
    for entity in targets:
        names = candidate_names(entity["canonical_name"], entity["aliases"])
        if len(names) < 2:
            judgements.append(NameJudgement(entity, names, STATUS_KEEP, names[0] if names else None))
            continue
        pending.append((entity, names))
        items.append((build_name_state(entity["entity_type"], names), build_name_questions(names)))

    results, jev_stats = run_jev_items(items, progress=progress, label="entity names")
    stats.jev_calls = jev_stats.calls
    stats.jev_errors = jev_stats.errors
    stats.cost_usd = jev_stats.cost_usd
    for (entity, names), result in zip(pending, results):
        if isinstance(result, Exception):
            judgements.append(NameJudgement(entity, names, STATUS_ERROR, error=str(result)))
            continue
        status, winner, confidence = evaluate_answers(result.answers, names, threshold)
        judgements.append(NameJudgement(entity, names, status, winner, confidence))
    stats.kept = sum(1 for j in judgements if j.status == STATUS_KEEP)
    stats.uncertain = sum(1 for j in judgements if j.status == STATUS_UNCERTAIN)
    return judgements, stats


# ---------------------------------------------------------------------------
# Planning (reads only)
# ---------------------------------------------------------------------------


def _types_compatible(a: str, b: str) -> bool:
    return not a or not b or a == b


def fetch_entities_by_name(conn, names: Sequence[str]) -> dict[str, list[dict]]:
    if not names:
        return {}
    rows = conn.execute(
        f"SELECT {_ENTITY_COLUMNS} FROM entities WHERE canonical_name = ANY(%s)",
        (list(names),),
    ).fetchall()
    by_name: dict[str, list[dict]] = {}
    for row in rows:
        entity = _entity_row(row)
        by_name.setdefault(entity["canonical_name"], []).append(entity)
    return by_name


def fetch_entities_by_id(conn, ids: Sequence[str]) -> dict[str, dict]:
    if not ids:
        return {}
    rows = conn.execute(
        f"SELECT {_ENTITY_COLUMNS} FROM entities WHERE id = ANY(%s::uuid[])", (list(ids),)
    ).fetchall()
    return {entity["id"]: entity for entity in (_entity_row(r) for r in rows)}


def plan_actions(
    conn,
    session,
    judgements: Sequence[NameJudgement],
    pick_survivor: Callable[[list[dict]], dict],
) -> NamePlan:
    """Turn `rename` judgements into rename-in-place / merge actions.

    `pick_survivor` is the shared mentions-first picker from
    `scripts/merge_semantic_duplicates.py`. Reads Postgres and Memgraph only.
    """
    plan = NamePlan()
    renames = [j for j in judgements if j.status == STATUS_RENAME and j.winner]
    existing = fetch_entities_by_name(conn, sorted({j.winner for j in renames}))
    for judgement in renames:
        target = dict(judgement.entity)
        others = [e for e in existing.get(judgement.winner, []) if e["id"] != target["id"]]
        compatible = [e for e in others if _types_compatible(target["entity_type"], e["entity_type"])]
        if others and not compatible:
            plan.blocked.append(
                BlockedRename(
                    target,
                    judgement.winner,
                    "an entity with that exact name already exists with an incompatible type "
                    f"({', '.join(sorted({e['entity_type'] or '?' for e in others}))} vs "
                    f"{target['entity_type'] or '?'})",
                )
            )
            continue
        members = [target, *compatible]
        attach_mention_counts(session, members)
        survivor = pick_survivor(members)
        plan.actions.append(
            RenameAction(
                kind=ACTION_MERGE if compatible else ACTION_RENAME,
                old_name=target["canonical_name"],
                winner=judgement.winner,
                confidence=judgement.confidence or 0.0,
                members=members,
                survivor_id=survivor["id"],
                dup_ids=[m["id"] for m in members if m["id"] != survivor["id"]],
            )
        )
    return plan


def format_action(action: RenameAction) -> str:
    mentions = {m["id"]: m.get("mentions", 0) for m in action.members}
    if action.kind == ACTION_RENAME:
        return (
            f"rename {action.old_name!r} -> {action.winner!r} "
            f"({mentions[action.survivor_id]} mention(s), confidence {action.confidence:.2f})"
        )
    names = {m["id"]: m["canonical_name"] for m in action.members}
    return (
        f"merge {action.old_name!r} with existing {action.winner!r}: keep "
        f"{names[action.survivor_id]!r} ({mentions[action.survivor_id]} mention(s)) as "
        f"{action.winner!r}, absorb {[names[i] for i in action.dup_ids]} "
        f"({sum(mentions[i] for i in action.dup_ids)} mention(s)); confidence {action.confidence:.2f}"
    )


# ---------------------------------------------------------------------------
# Applying (--execute only)
# ---------------------------------------------------------------------------

MergeFn = Callable[[object, object, str, list[str], list[dict]], int]


def apply_actions(
    conn,
    session,
    actions: Sequence[RenameAction],
    merge_fn: MergeFn,
    *,
    progress: Callable[[str], None] = _print,
) -> ApplyResult:
    """Apply planned actions. The caller must hold the maintenance lock with no
    active jobs. `merge_fn(session, conn, survivor_id, dup_ids, members)` moves
    the MENTIONS edges and deletes the merged-away rows (Memgraph first, then
    Postgres, without committing) and returns the number of edges repointed.

    Per action: skip when a member was already consumed earlier in this run or
    no longer matches what was planned; otherwise merge (if needed), then give
    the survivor the winning `canonical_name` with every other name of the
    cluster as an alias, and commit.
    """
    result = ApplyResult()
    for action in actions:
        ids = [m["id"] for m in action.members]
        if result.touched_ids.intersection(ids):
            result.skipped += 1
            progress(f"  skipped (entity already changed in this run): {action.old_name!r}")
            continue
        fresh = fetch_entities_by_id(conn, ids)
        by_id = {m["id"]: m for m in action.members}
        if len(fresh) != len(ids) or fresh[ids[0]]["canonical_name"] != action.old_name:
            result.skipped += 1
            progress(f"  skipped (entity changed since planning): {action.old_name!r}")
            continue
        members = []
        for entity_id in ids:
            member = dict(fresh[entity_id])
            member["mentions"] = by_id[entity_id].get("mentions", 0)
            members.append(member)
        survivor = next(m for m in members if m["id"] == action.survivor_id)

        all_names: set[str] = set()
        for member in members:
            all_names.add(member["canonical_name"])
            all_names.update(member["aliases"])
        aliases = sorted(n for n in all_names if n and n != action.winner)

        try:
            edges = 0
            if action.dup_ids:
                edges = merge_fn(session, conn, action.survivor_id, action.dup_ids, members)
            if survivor["canonical_name"] != action.winner:
                session.run(RENAME_NODE_CQL, entity_id=action.survivor_id, name=action.winner)
                conn.execute(
                    "UPDATE entities SET canonical_name = %s, aliases = %s WHERE id = %s::uuid",
                    (action.winner, aliases, action.survivor_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        result.touched_ids.update(ids)
        result.deleted_ids.update(action.dup_ids)
        result.edges_repointed += edges
        result.merged += len(action.dup_ids)
        if survivor["canonical_name"] != action.winner:
            result.renamed += 1
        progress("  applied: " + format_action(action))
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_name_correction(
    conn,
    session,
    *,
    dry_run: bool,
    limit: int,
    merge_fn: MergeFn,
    pick_survivor: Callable[[list[dict]], dict],
    min_aliases: int | None = None,
    min_confidence: float | None = None,
    progress: Callable[[str], None] = _print,
) -> ApplyResult:
    """Target -> judge -> plan -> (apply under --execute). Dry-run prints the
    proposals and writes nothing (Jev is still called, which spends credits).
    Returns the apply result (empty in a dry-run)."""
    prefix = "[DRY RUN] " if dry_run else ""
    aliases_needed = settings.ENTITY_NAME_REVIEW_MIN_ALIASES if min_aliases is None else min_aliases
    targets = fetch_name_targets(conn, aliases_needed, limit)
    progress(
        f"{prefix}Entity names: {len(targets)} hub entity/entities with >= {aliases_needed} alias(es) "
        f"(limit {limit})."
    )
    if not targets:
        return ApplyResult()

    judgements, stats = judge_names(targets, min_confidence=min_confidence, progress=progress)
    plan = plan_actions(conn, session, judgements, pick_survivor)
    progress(
        f"{prefix}Entity names: {len(plan.actions)} proposed ({sum(1 for a in plan.actions if a.kind == ACTION_RENAME)} "
        f"rename, {sum(1 for a in plan.actions if a.kind == ACTION_MERGE)} merge into existing), "
        f"{len(plan.blocked)} blocked, {stats.kept} already correct, {stats.uncertain} uncertain, "
        f"{stats.jev_errors} Jev error(s) (left unchanged); {stats.jev_calls} Jev call(s), "
        f"cost ${stats.cost_usd:.4f}."
    )
    for blocked in plan.blocked:
        progress(
            f"  {prefix}blocked: {blocked.entity['canonical_name']!r} -> {blocked.winner!r}: {blocked.reason}"
        )
    if dry_run:
        for action in plan.actions:
            progress(f"  {prefix}would " + format_action(action))
        return ApplyResult()

    result = apply_actions(conn, session, plan.actions, merge_fn, progress=progress)
    progress(
        f"Entity names: renamed {result.renamed}, merged away {result.merged} row(s), "
        f"re-pointed {result.edges_repointed} edge(s), skipped {result.skipped}."
    )
    return result
