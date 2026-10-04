#!/usr/bin/env python3
"""Merge semantically duplicate entities (same real-world entity, different name/type).

Uses the same combined fuzzy + embedding approach as assess_semantic_duplicates.py
to find candidate pairs, groups them into clusters via union-find, then merges each
cluster the same way as merge_duplicate_entities.py.

Survivor selection (`pick_survivor`): the member with the most MENTIONS edges
wins (counted from Memgraph in one batched query); ties fall back to the cluster's
dominant entity_type, then has_embedding=True, then oldest created_at. Absorbed
entity names are added to the survivor's aliases list; the survivor's own
canonical_name is left alone (spelling correction is src/rag/entity_names.py).

`--jev` (and weekly_maintenance's Jev gate) additionally confirms every candidate
pair with Jev before merging; see `confirm_pairs_with_jev`.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from rag import jev_dedup
from rag.db import get_connection, vacuum_analyze_entities
from rag.entity_audit import attach_mention_counts
from rag.graph_db import get_graph_driver


# ── Fuzzy + embedding candidate detection ────────────────────────────────────

def _prefix_key(name: str, length: int) -> str:
    return name.strip().lower()[:length]


def _length_band(n: int, tolerance: float = 0.40) -> tuple[int, int]:
    return int(n * (1 - tolerance)), int(n * (1 + tolerance)) + 1


_DIGITS_RE = re.compile(r"\d+")

# Words that look fuzzy-similar but are semantically incompatible
_MAGNITUDE_WORDS = {"thousand", "million", "billion", "trillion", "k", "m", "b"}
_EXCLUSIVE_KEYWORDS = [
    {"generative", "agentic"},
    {"million", "billion"},
    {"billion", "trillion"},
    {"thousand", "million"},
]


def _digit_sets_differ(name_a: str, name_b: str) -> bool:
    """Return True if the two names differ in any digit sequence."""
    return _DIGITS_RE.findall(name_a) != _DIGITS_RE.findall(name_b)


def _has_exclusive_keyword_conflict(name_a: str, name_b: str) -> bool:
    """Return True if the names contain mutually exclusive keywords."""
    tokens_a = set(re.findall(r"[a-z]+", name_a.lower()))
    tokens_b = set(re.findall(r"[a-z]+", name_b.lower()))
    for exclusive_set in _EXCLUSIVE_KEYWORDS:
        if tokens_a & exclusive_set and tokens_b & exclusive_set and not (tokens_a & exclusive_set) == (tokens_b & exclusive_set):
            return True
    return False


def find_candidate_pairs(
    entities: list[tuple[str, str, str]],
    fuzzy_threshold: float,
    person_fuzzy_threshold: float,
    prefix_len: int,
) -> list[tuple[str, str, float]]:
    """Return (id_a, id_b, fuzzy_ratio) pairs above the relevant fuzzy threshold."""
    blocks: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for eid, name, etype in entities:
        if not name:
            continue
        blocks[_prefix_key(name, prefix_len)].append((eid, name, etype))

    pairs = []
    for block in blocks.values():
        if len(block) < 2:
            continue
        for i in range(len(block)):
            id_a, name_a, type_a = block[i]
            low, high = _length_band(len(name_a))
            for j in range(i + 1, len(block)):
                id_b, name_b, type_b = block[j]
                if not (low <= len(name_b) <= high):
                    continue
                if name_a == name_b:
                    continue
                # Skip pairs whose only difference is a digit sequence
                if _digit_sets_differ(name_a, name_b):
                    continue
                # Skip pairs with mutually exclusive keywords (magnitude, domain)
                if _has_exclusive_keyword_conflict(name_a, name_b):
                    continue
                min_ratio = person_fuzzy_threshold if "PERSON" in (type_a, type_b) else fuzzy_threshold
                ratio = SequenceMatcher(None, name_a.lower(), name_b.lower(), autojunk=False).ratio()
                if ratio >= min_ratio:
                    pairs.append((id_a, id_b, ratio))
    return pairs


def fetch_embeddings(conn, ids: list[str]) -> dict[str, np.ndarray]:
    rows = conn.execute(
        "SELECT id::text, embedding::text FROM entities WHERE id = ANY(%s::uuid[]) AND embedding IS NOT NULL",
        (ids,),
    ).fetchall()
    result = {}
    for row in rows:
        vec = np.array(list(map(float, row[1].strip("[]").split(","))), dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        result[row[0]] = vec
    return result


# ── Union-find clustering ─────────────────────────────────────────────────────

class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for node in self.parent:
            root = self.find(node)
            result.setdefault(root, []).append(node)
        return {k: v for k, v in result.items() if len(v) > 1}


# ── Merge helpers (mirrored from merge_duplicate_entities.py) ─────────────────

def pick_survivor(members: list[dict]) -> dict:
    """Pick the entity row that survives a merge cluster.

    One sort key, in order: most MENTIONS edges (`member["mentions"]`, filled by
    `attach_mention_counts`; missing counts as 0), then membership of the
    cluster's dominant entity_type, then has_embedding, then the oldest row.
    The row with the most mentions keeps its id (and its edges) so the merge
    repoints the fewest edges and the best-connected name stays the hub. The
    dominant type is now only a tie-breaker: as a hard filter it let a
    mis-typed one-mention row beat a hub just because its type was the majority.
    """
    type_counts = Counter(m["entity_type"] for m in members)
    dominant_type = type_counts.most_common(1)[0][0]
    return sorted(
        members,
        key=lambda m: (
            -m.get("mentions", 0),
            m["entity_type"] != dominant_type,
            not m["has_embedding"],
            m["created_at"] is None,
            m["created_at"],
        ),
    )[0]


def merge_postgres(conn, survivor_id: str, dup_ids: list[str], all_members: list[dict]) -> None:
    if not dup_ids:
        return
    # Collect all aliases + absorbed canonical_names as new aliases
    all_aliases: set[str] = set()
    survivor_name = next(m["canonical_name"] for m in all_members if m["id"] == survivor_id)
    for m in all_members:
        all_aliases.update(m["aliases"])
        if m["id"] != survivor_id and m["canonical_name"]:
            all_aliases.add(m["canonical_name"])
    all_aliases.discard(survivor_name)

    conn.execute(
        "UPDATE entities SET aliases = %s WHERE id = %s",
        (sorted(all_aliases), survivor_id),
    )
    conn.execute(
        "DELETE FROM entities WHERE id = ANY(%s::uuid[])",
        (dup_ids,),
    )


def merge_memgraph(session, survivor_id: str, dup_ids: list[str]) -> int:
    edges_repointed = 0
    for dup_id in dup_ids:
        result = session.run(
            "MATCH (c:Chunk)-[old:MENTIONS]->(e:Entity {entity_id: $dup_id}) "
            "MATCH (s:Entity {entity_id: $survivor_id}) "
            # Match on the endpoints only: MERGE-ing on the confidence value would
            # create a parallel edge once salience (entity audit) makes the
            # confidences of the two mentions differ.
            "MERGE (c)-[m:MENTIONS]->(s) ON CREATE SET m.confidence = old.confidence "
            "WITH e, count(old) AS repointed "
            "DETACH DELETE e "
            "RETURN repointed",
            dup_id=dup_id,
            survivor_id=survivor_id,
        )
        record = result.single()
        if record:
            edges_repointed += record["repointed"]

        session.run(
            "MATCH (e:Entity {entity_id: $dup_id})-[r:RELATED_TO]->(other:Entity) "
            "WHERE other.entity_id <> $survivor_id "
            "MATCH (s:Entity {entity_id: $survivor_id}) "
            "MERGE (s)-[:RELATED_TO {type: r.type, confidence: r.confidence, chunk_id: r.chunk_id}]->(other)",
            dup_id=dup_id,
            survivor_id=survivor_id,
        )
        session.run(
            "MATCH (other:Entity)-[r:RELATED_TO]->(e:Entity {entity_id: $dup_id}) "
            "WHERE other.entity_id <> $survivor_id "
            "MATCH (s:Entity {entity_id: $survivor_id}) "
            "MERGE (other)-[:RELATED_TO {type: r.type, confidence: r.confidence, chunk_id: r.chunk_id}]->(s)",
            dup_id=dup_id,
            survivor_id=survivor_id,
        )
        session.run(
            "MATCH (e:Entity {entity_id: $dup_id}) DETACH DELETE e",
            dup_id=dup_id,
        )

    return edges_repointed


def merge_cluster(conn, session, cluster_members: list[dict], dry_run: bool) -> int:
    # No-op when run_entity_merge already batched the counts for all clusters.
    attach_mention_counts(session, cluster_members)
    survivor = pick_survivor(cluster_members)
    dup_ids = [m["id"] for m in cluster_members if m["id"] != survivor["id"]]
    dup_names = [m["canonical_name"] for m in cluster_members if m["id"] != survivor["id"]]

    if dry_run:
        print(
            f"  [{survivor['entity_type']}] {survivor['canonical_name']!r} "
            f"({survivor.get('mentions', 0)} mention(s)) — absorbs: {dup_names}"
        )
        return 0

    try:
        edges = merge_memgraph(session, survivor["id"], dup_ids)
        merge_postgres(conn, survivor["id"], dup_ids, cluster_members)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    print(
        f"  [{survivor['entity_type']}] {survivor['canonical_name']!r} — "
        f"merged {len(dup_ids)} duplicate(s), re-pointed {edges} edge(s)"
    )
    return len(dup_ids)


# ── Jev confirmation ──────────────────────────────────────────────────────────

def confirm_pairs_with_jev(
    pairs: list[tuple[str, str, float]],
    entity_meta: dict[str, dict],
    dry_run: bool,
) -> list[tuple[str, str, float]]:
    """Confirm candidate pairs with Jev and return only the ones to merge.

    Routing by the 3-level alignment score (`rag.jev_dedup`): >= 1.5 merge,
    0.5..1.5 review (`entity_merge_review`, written only when not `dry_run`,
    reported otherwise), < 0.5 dropped. A Jev error leaves the pair
    unconfirmed (no merge). Pairs that already have a review row, of any status,
    are neither re-judged nor merged. Judging calls Jev even in a dry-run (it
    spends credits) but writes nothing to the corpus.
    """
    prefix = "[DRY RUN] " if dry_run else ""
    if not pairs:
        return []

    with get_connection() as conn:
        reviewed = jev_dedup.fetch_reviewed_entity_pairs(conn)
    fresh = [p for p in pairs if jev_dedup.normalize_pair(p[0], p[1]) not in reviewed]
    print(
        f"{prefix}Jev: confirming {len(fresh)} candidate pair(s) "
        f"({len(pairs) - len(fresh)} already in entity_merge_review, skipped)...",
        flush=True,
    )
    judgements, stats = jev_dedup.judge_entity_pairs(fresh, entity_meta)
    counts = jev_dedup.summarize(judgements)
    print(
        f"{prefix}Jev entity pairs: {counts['merge']} confirmed merge, {counts['review']} for review, "
        f"{counts['drop']} dropped, {counts['error']} unconfirmed (Jev error); {stats.describe()}",
        flush=True,
    )

    for judgement in judgements:
        if judgement.route == jev_dedup.ROUTE_REVIEW:
            print(f"  {prefix}review ({judgement.score:.2f}): {judgement.reason}")
    if counts["review"]:
        if dry_run:
            print(f"{prefix}Would add {counts['review']} pair(s) to entity_merge_review.")
        else:
            with get_connection() as conn:
                added = jev_dedup.insert_entity_reviews(conn, judgements)
            print(f"Added {added} pair(s) to entity_merge_review.")

    return [
        (j.id_a, j.id_b, j.similarity)
        for j in judgements
        if j.route == jev_dedup.ROUTE_MERGE
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_entity_merge(
    dry_run: bool,
    fuzzy_threshold: float = 0.82,
    person_fuzzy_threshold: float = 0.90,
    cosine_threshold: float = 0.93,
    prefix_len: int = 3,
    use_jev: bool = False,
) -> tuple[int, set[str]]:
    """Find and merge semantic-duplicate entity clusters.

    With `use_jev`, every pair that passes the fuzzy + cosine filters is first
    confirmed by Jev (see `confirm_pairs_with_jev`) and only confirmed pairs are
    clustered; without it, behavior is unchanged.

    Returns (total_merged, set of all entity ids that were part of any merge
    cluster — survivors and merged-away alike — so the caller can invalidate
    the entity_semantic_edges cache).

    Extracted from `main()` so `scripts/weekly_maintenance.py` can invoke the
    exact same code path as this script's own CLI entry point (U7) — both
    call this function; neither re-implements the candidate-detection/merge
    logic. Vacuums `entities` (via `vacuum_analyze_entities`) only when at
    least one merge actually happened; a dry run or a zero-cluster run never
    vacuums.
    """
    prefix = "[DRY RUN] " if dry_run else ""

    with get_connection() as conn:
        print(f"{prefix}Loading entities...", flush=True)
        rows = conn.execute(
            """
            SELECT id::text, canonical_name, entity_type, aliases,
                   embedding IS NOT NULL AS has_embedding, created_at
            FROM entities
            ORDER BY canonical_name
            """
        ).fetchall()

        entity_meta: dict[str, dict] = {
            r[0]: {
                "id": r[0],
                "canonical_name": r[1] or "",
                "entity_type": r[2] or "",
                "aliases": r[3] or [],
                "has_embedding": r[4],
                "created_at": r[5],
            }
            for r in rows
        }
        entities = [(r[0], r[1] or "", r[2] or "") for r in rows]

        print(f"{prefix}Loaded {len(entities)} entities. Running fuzzy matching...", flush=True)
        fuzzy_pairs = find_candidate_pairs(
            entities, fuzzy_threshold, person_fuzzy_threshold, prefix_len
        )
        print(f"{prefix}Fuzzy candidates: {len(fuzzy_pairs)}. Fetching embeddings...", flush=True)

        unique_ids = list({id_ for a, b, _ in fuzzy_pairs for id_ in (a, b)})
        embeddings = fetch_embeddings(conn, unique_ids)

    print(f"{prefix}Computing cosine similarities...", flush=True)
    candidate_pairs: list[tuple[str, str, float]] = []
    for id_a, id_b, _ in fuzzy_pairs:
        emb_a = embeddings.get(id_a)
        emb_b = embeddings.get(id_b)
        if emb_a is None or emb_b is None:
            continue
        cosine = float(np.dot(emb_a, emb_b))
        if cosine >= cosine_threshold:
            candidate_pairs.append((id_a, id_b, cosine))

    if use_jev:
        # Only Jev-confirmed pairs are clustered; unconfirmed ones (review band,
        # drops, Jev errors) are never merged.
        merge_pairs = confirm_pairs_with_jev(candidate_pairs, entity_meta, dry_run)
    else:
        merge_pairs = candidate_pairs

    uf = UnionFind()
    for id_a, id_b, _ in merge_pairs:
        uf.union(id_a, id_b)

    clusters = uf.groups()
    total_members = sum(len(v) for v in clusters.values())
    print(
        f"{prefix}Found {len(clusters)} semantic duplicate clusters "
        f"({total_members} total entities)."
    )

    total_merged = 0
    merged_entity_ids: set[str] = set()
    with get_connection() as conn, get_graph_driver() as driver:
        with driver.session() as session:
            cluster_members = [
                [entity_meta[eid] for eid in cluster_ids if eid in entity_meta]
                for cluster_ids in clusters.values()
            ]
            # One batched MENTIONS count for every clustered entity, so survivor
            # selection (pick_survivor) costs a single UNWIND query, dry-run included.
            attach_mention_counts(session, [m for members in cluster_members for m in members])
            for cluster_ids, members in zip(clusters.values(), cluster_members):
                if len(members) < 2:
                    continue
                total_merged += merge_cluster(conn, session, members, dry_run=dry_run)
                merged_entity_ids |= set(cluster_ids)

    if total_merged > 0:
        print(f"Reclaiming space from {total_merged} merged row(s)...", flush=True)
        vacuum_analyze_entities()

    return total_merged, merged_entity_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge semantic duplicate entities.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.82)
    parser.add_argument("--person-fuzzy-threshold", type=float, default=0.90)
    parser.add_argument("--cosine-threshold", type=float, default=0.93)
    parser.add_argument("--prefix-len", type=int, default=3)
    parser.add_argument(
        "--jev",
        action="store_true",
        help=(
            "Confirm every candidate pair with Jev before merging: confirmed pairs merge, "
            "borderline pairs go to entity_merge_review, the rest are dropped. Spends Jev "
            "credits even with --dry-run."
        ),
    )
    args = parser.parse_args()

    total_merged, _ = run_entity_merge(
        dry_run=args.dry_run,
        fuzzy_threshold=args.fuzzy_threshold,
        person_fuzzy_threshold=args.person_fuzzy_threshold,
        cosine_threshold=args.cosine_threshold,
        prefix_len=args.prefix_len,
        use_jev=args.jev,
    )

    print(f"Total merged: {total_merged}. Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
