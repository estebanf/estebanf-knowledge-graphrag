#!/usr/bin/env python3
"""Combined semantic duplicate assessment: fuzzy string + embedding cosine similarity.

Phase 1 — fuzzy blocking (fast, full coverage):
    Groups entities by name prefix + length band, computes SequenceMatcher ratio.
    Collects candidate pairs above --fuzzy-threshold.

Phase 2 — embedding confirmation (batch fetch, numpy):
    Fetches embeddings for all candidate entity IDs in one query.
    Computes cosine similarity for each pair.

A pair is reported if it passes BOTH filters. PERSON entities require a higher
fuzzy threshold (--person-fuzzy-threshold) to suppress the context-similarity
false positives seen in the embedding-only assessment.

Read-only — no writes to DB or graph.
"""
import argparse
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from rag.db import get_connection


def _prefix_key(name: str, length: int) -> str:
    return name.strip().lower()[:length]


def _length_band(n: int, tolerance: float = 0.40) -> tuple[int, int]:
    return int(n * (1 - tolerance)), int(n * (1 + tolerance)) + 1


_DIGITS_RE = re.compile(r"\d+")

_EXCLUSIVE_KEYWORDS = [
    {"generative", "agentic"},
    {"million", "billion"},
    {"billion", "trillion"},
    {"thousand", "million"},
]


def _digit_sets_differ(name_a: str, name_b: str) -> bool:
    return _DIGITS_RE.findall(name_a) != _DIGITS_RE.findall(name_b)


def _has_exclusive_keyword_conflict(name_a: str, name_b: str) -> bool:
    tokens_a = set(re.findall(r"[a-z]+", name_a.lower()))
    tokens_b = set(re.findall(r"[a-z]+", name_b.lower()))
    for exclusive_set in _EXCLUSIVE_KEYWORDS:
        if tokens_a & exclusive_set and tokens_b & exclusive_set and not (tokens_a & exclusive_set) == (tokens_b & exclusive_set):
            return True
    return False


def fuzzy_candidates(
    entities: list[tuple[str, str, str]],
    threshold: float,
    person_threshold: float,
    prefix_len: int,
) -> list[tuple[str, str, str, str, str, str, float]]:
    """Return (id_a, name_a, type_a, id_b, name_b, type_b, ratio) pairs above threshold."""
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
                if _digit_sets_differ(name_a, name_b):
                    continue
                if _has_exclusive_keyword_conflict(name_a, name_b):
                    continue
                min_ratio = person_threshold if "PERSON" in (type_a, type_b) else threshold
                ratio = SequenceMatcher(None, name_a.lower(), name_b.lower(), autojunk=False).ratio()
                if ratio >= min_ratio:
                    pairs.append((id_a, name_a, type_a, id_b, name_b, type_b, ratio))
    return pairs


def fetch_embeddings(conn, ids: list[str]) -> dict[str, np.ndarray]:
    """Batch-fetch embeddings for a list of entity IDs."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Combined fuzzy + embedding semantic duplicate assessment.")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.82,
                        help="SequenceMatcher ratio threshold for non-PERSON pairs (default 0.82)")
    parser.add_argument("--person-fuzzy-threshold", type=float, default=0.90,
                        help="SequenceMatcher ratio threshold for PERSON pairs (default 0.90)")
    parser.add_argument("--cosine-threshold", type=float, default=0.93,
                        help="Cosine similarity threshold (default 0.93)")
    parser.add_argument("--prefix-len", type=int, default=3,
                        help="Blocking prefix length (default 3)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max pairs to print (default 100)")
    args = parser.parse_args()

    print(
        f"Fuzzy threshold: {args.fuzzy_threshold} (PERSON: {args.person_fuzzy_threshold})  "
        f"|  Cosine threshold: {args.cosine_threshold}"
    )

    with get_connection() as conn:
        print("Loading entities...", flush=True)
        rows = conn.execute(
            "SELECT id::text, canonical_name, entity_type FROM entities ORDER BY canonical_name"
        ).fetchall()
        entities = [(r[0], r[1] or "", r[2] or "") for r in rows]
        print(f"Loaded {len(entities)} entities. Running fuzzy matching...", flush=True)

        candidates = fuzzy_candidates(
            entities, args.fuzzy_threshold, args.person_fuzzy_threshold, args.prefix_len
        )
        print(f"Fuzzy candidates: {len(candidates)}. Fetching embeddings...", flush=True)

        unique_ids = list({id_ for c in candidates for id_ in (c[0], c[3])})
        embeddings = fetch_embeddings(conn, unique_ids)

    print(f"Embeddings fetched for {len(embeddings)} entities. Computing cosine similarities...", flush=True)

    confirmed: list[tuple[float, float, str, str, str, str, str, str]] = []
    no_embedding: list[tuple[float, str, str, str, str, str, str]] = []

    for id_a, name_a, type_a, id_b, name_b, type_b, ratio in candidates:
        emb_a = embeddings.get(id_a)
        emb_b = embeddings.get(id_b)
        if emb_a is None or emb_b is None:
            no_embedding.append((ratio, id_a, name_a, type_a, id_b, name_b, type_b))
            continue
        cosine = float(np.dot(emb_a, emb_b))
        if cosine >= args.cosine_threshold:
            confirmed.append((ratio, cosine, id_a, name_a, type_a, id_b, name_b, type_b))

    # Sort by cosine desc, then fuzzy desc
    confirmed.sort(key=lambda x: (x[1], x[0]), reverse=True)

    print(f"\n{'='*70}")
    print(f"Confirmed duplicate pairs: {len(confirmed)}")
    print(f"Pairs lacking embeddings (fuzzy only): {len(no_embedding)}")
    print(f"{'='*70}\n")

    # Group by entity type for summary
    by_type: dict[str, int] = defaultdict(int)
    for _, _, _, _, type_a, _, _, type_b in confirmed:
        key = type_a if type_a == type_b else f"{type_a}/{type_b}"
        by_type[key] += 1

    print("By type:")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    print(f"\nTop {min(args.limit, len(confirmed))} pairs (sorted by cosine sim):\n")
    for ratio, cosine, id_a, name_a, type_a, id_b, name_b, type_b in confirmed[: args.limit]:
        print(f"  fuzzy={ratio:.3f}  cosine={cosine:.3f}  [{type_a}] {name_a!r}  ↔  [{type_b}] {name_b!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
