#!/usr/bin/env python3
"""Script B: find semantically similar entities via fuzzy string matching.

Uses difflib.SequenceMatcher (stdlib) with blocking to avoid O(n²) comparisons.
Read-only assessment — no writes to DB or graph.
"""
import argparse
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.db import get_connection


def _prefix_key(name: str, length: int = 3) -> str:
    return name.strip().lower()[:length]


def _length_band(n: int, tolerance: float = 0.40) -> tuple[int, int]:
    low = int(n * (1 - tolerance))
    high = int(n * (1 + tolerance)) + 1
    return low, high


def main() -> int:
    parser = argparse.ArgumentParser(description="Assess fuzzy string semantic duplicates.")
    parser.add_argument("--threshold", type=float, default=0.82, help="SequenceMatcher ratio threshold (default 0.82)")
    parser.add_argument("--limit", type=int, default=50, help="Max pairs to print (default 50)")
    parser.add_argument("--prefix-len", type=int, default=3, help="Blocking prefix length (default 3)")
    args = parser.parse_args()

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id::text, canonical_name, entity_type FROM entities ORDER BY canonical_name"
        ).fetchall()

    entities = [(row[0], row[1] or "", row[2] or "") for row in rows]
    print(f"Threshold: {args.threshold}  |  Entities loaded: {len(entities)}")
    print(f"Blocking: prefix-{args.prefix_len} + length-band-40%\n")

    # Build blocks by prefix
    blocks: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for eid, name, etype in entities:
        if not name:
            continue
        key = _prefix_key(name, args.prefix_len)
        blocks[key].append((eid, name, etype))

    pairs: list[tuple[float, str, str, str, str, str, str]] = []
    comparisons = 0

    for block in blocks.values():
        if len(block) < 2:
            continue
        for i in range(len(block)):
            id_a, name_a, type_a = block[i]
            len_a = len(name_a)
            low, high = _length_band(len_a)
            for j in range(i + 1, len(block)):
                id_b, name_b, type_b = block[j]
                # Length-band filter
                if not (low <= len(name_b) <= high):
                    continue
                # Skip exact canonical_name matches (already handled)
                if name_a == name_b:
                    continue
                comparisons += 1
                ratio = SequenceMatcher(None, name_a.lower(), name_b.lower(), autojunk=False).ratio()
                if ratio >= args.threshold:
                    pairs.append((ratio, id_a, name_a, type_a, id_b, name_b, type_b))

    pairs.sort(reverse=True)

    print(f"Comparisons performed: {comparisons:,}")
    print(f"Candidate pairs found: {len(pairs)}")
    print(f"\nTop {min(args.limit, len(pairs))} pairs (sorted by ratio desc):\n")

    for ratio, id_a, name_a, type_a, id_b, name_b, type_b in pairs[: args.limit]:
        print(f"  ratio={ratio:.3f}  {name_a!r} [{type_a}]  ↔  {name_b!r} [{type_b}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
