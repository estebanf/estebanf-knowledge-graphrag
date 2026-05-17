#!/usr/bin/env python3
"""Script A: find semantically similar entities via embedding cosine similarity.

Loads a sample of entity embeddings into Python and computes the full pairwise
similarity matrix with numpy — no per-entity DB round-trips, no index issues.

Read-only assessment — no writes to DB or graph.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from rag.db import get_connection


class UnionFind:
    def __init__(self):
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)

    def groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for node in self.parent:
            root = self.find(node)
            result.setdefault(root, []).append(node)
        return {k: v for k, v in result.items() if len(v) > 1}


def main() -> int:
    parser = argparse.ArgumentParser(description="Assess embedding-based semantic duplicates.")
    parser.add_argument("--threshold", type=float, default=0.97, help="Cosine similarity threshold (default 0.97)")
    parser.add_argument("--sample", type=int, default=500, help="Number of entities to sample (default 500)")
    parser.add_argument("--limit", type=int, default=30, help="Max clusters to print (default 30)")
    args = parser.parse_args()

    print(f"Threshold: {args.threshold}  |  Sample: {args.sample}")
    print("Loading embeddings...", flush=True)

    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id::text, canonical_name, entity_type, embedding::text
            FROM entities
            WHERE embedding IS NOT NULL
            ORDER BY random()
            LIMIT %s
            """,
            (args.sample,),
        ).fetchall()

    n = len(rows)
    ids = [r[0] for r in rows]
    names = [r[1] for r in rows]
    types = [r[2] for r in rows]

    print(f"Loaded {n} entities. Computing {n}×{n} similarity matrix...", flush=True)

    # Parse pgvector text format "[0.1,0.2,...]" into numpy matrix
    matrix = np.array([
        list(map(float, r[3].strip("[]").split(",")))
        for r in rows
    ], dtype=np.float32)

    # L2-normalise rows so dot product = cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    # Full pairwise cosine similarity — upper triangle only
    sim_matrix = matrix @ matrix.T

    print("Finding pairs above threshold...", flush=True)

    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if names[i] == names[j]:
                continue  # exact-name matches already merged
            sim = float(sim_matrix[i, j])
            if sim >= args.threshold:
                pairs.append((i, j, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\nCandidate pairs found: {len(pairs)}")

    uf = UnionFind()
    for i, j, _ in pairs:
        uf.union(i, j)

    clusters = uf.groups()
    print(f"Clusters: {len(clusters)}")

    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    print(f"\nTop {min(args.limit, len(sorted_clusters))} clusters:\n")
    for cluster in sorted_clusters[: args.limit]:
        anchor = cluster[0]
        print(f"  [{types[anchor]}] {names[anchor]!r}  ({len(cluster)} members)")
        for mid in cluster[1:]:
            key = (min(anchor, mid), max(anchor, mid))
            sim = next((s for a, b, s in pairs if (min(a,b), max(a,b)) == key), 0.0)
            print(f"    sim={sim:.3f}  {names[mid]!r} [{types[mid]}]")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
