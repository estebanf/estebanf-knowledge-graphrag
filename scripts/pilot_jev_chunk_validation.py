#!/usr/bin/env python3
"""Pilot: how would Jev chunk validation (`CHUNK_VALIDATION_BACKEND=jev`) behave
on the existing corpus?

READ-ONLY BY CONSTRUCTION. The script only runs SELECT statements against
Postgres (on a connection marked read-only), never writes to Postgres or
Memgraph, and never soft-deletes anything: it only *reports* which chunks would
be skipped. The only side effect is outbound Jev calls through OpenRouter
(small cost, reported at the end).

It samples live chunks from the `--sources N` most recent sources (at most
`--chunks-per-source` random chunks each, to bound cost), scores each with the
same three Jev questions the pipeline uses (complete / coherent / substantive;
low quality = minimum probability below the threshold) and prints

* the distribution of the per-chunk minimum probability and per-criterion means,
* for thresholds 0.2 / 0.3 / 0.4 / 0.5: chunks that would be skipped, sources
  that would lose at least one chunk, and sources that would hit the
  `CHUNK_VALIDATION_MAX_SKIP_FRACTION` cap (so nothing would be skipped there),
* a handful of example low-scoring chunk texts (truncated),
* the total Jev cost.

The per-source cap check uses the SAMPLED chunks of each source, which is an
approximation of the full-source fraction the pipeline sees. Parent chunks
(chunks referenced by another chunk's `parent_chunk_id`) are never counted as
skippable, exactly as in the pipeline.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag import jev
from rag.chunk_validation import (
    JEV_MAX_CHUNK_CHARS,
    QUALITY_CRITERIA,
    build_quality_questions,
    quality_probabilities,
    select_skips,
)
from rag.config import settings
from rag.db import get_connection

THRESHOLDS = (0.2, 0.3, 0.4, 0.5)

_SAMPLE_SQL = """
WITH recent AS (
  SELECT id FROM sources
  WHERE deleted_at IS NULL
  ORDER BY created_at DESC, id DESC
  LIMIT %s
), ranked AS (
  SELECT c.id, c.source_id, c.content,
         row_number() OVER (PARTITION BY c.source_id ORDER BY random()) AS rn
  FROM chunks c
  JOIN recent r ON r.id = c.source_id
  WHERE c.deleted_at IS NULL
)
SELECT id::text, source_id::text, content
FROM ranked
WHERE rn <= %s
ORDER BY source_id, id
"""

_PARENTS_SQL = """
SELECT DISTINCT parent_chunk_id::text
FROM chunks
WHERE source_id = ANY(%s::uuid[])
  AND parent_chunk_id IS NOT NULL
  AND deleted_at IS NULL
"""


def score_chunk(row: dict) -> dict:
    """Score one sampled chunk with a single Jev call (uses `system_one_response`
    so the usage cost is available, unlike `system_one_many`)."""
    scored = dict(row)
    try:
        response = jev.system_one_response(
            row["text"][:JEV_MAX_CHUNK_CHARS], build_quality_questions()
        )
        probs = quality_probabilities(response.answers)
        scored["probs"] = probs
        scored["min_p"] = min(probs.values())
        scored["cost"] = float((response.usage or {}).get("cost") or 0.0)
    except Exception as exc:
        scored["error"] = str(exc)
        scored["min_p"] = None
        scored["cost"] = 0.0
    return scored


def analyze(
    scored: list[dict],
    protected_ids: set[str],
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    max_skip_fraction: float,
    bins: int = 10,
) -> dict:
    """Aggregate scored chunks into the pilot report (pure; no I/O)."""
    ok = [r for r in scored if r.get("min_p") is not None]
    result: dict = {
        "chunks": len(scored),
        "scored": len(ok),
        "errors": len(scored) - len(ok),
        "sources": len({r["source_id"] for r in scored}),
        "cost_usd": sum(r.get("cost", 0.0) for r in scored),
    }

    histogram = [0] * bins
    for r in ok:
        histogram[min(bins - 1, int(r["min_p"] * bins))] += 1
    result["histogram"] = histogram
    result["criterion_means"] = {
        name: (sum(r["probs"][name] for r in ok) / len(ok)) if ok else None
        for name in QUALITY_CRITERIA
    }

    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_source[r["source_id"]].append(r)

    per_threshold = []
    for threshold in thresholds:
        skipped_chunks = 0
        affected_sources = 0
        capped_sources = 0
        for rows in by_source.values():
            skipped, _, capped = select_skips(
                [(r["chunk_id"], r["min_p"]) for r in rows],
                protected_ids,
                threshold=threshold,
                max_skip_fraction=max_skip_fraction,
            )
            skipped_chunks += len(skipped)
            affected_sources += 1 if skipped else 0
            capped_sources += 1 if capped else 0
        per_threshold.append(
            {
                "threshold": threshold,
                "skipped_chunks": skipped_chunks,
                "skipped_fraction": (skipped_chunks / len(ok)) if ok else 0.0,
                "affected_sources": affected_sources,
                "capped_sources": capped_sources,
            }
        )
    result["thresholds"] = per_threshold
    result["lowest"] = sorted(ok, key=lambda r: r["min_p"])
    return result


def _one_line(text: str, width: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def print_report(report: dict, *, examples: int, max_skip_fraction: float) -> None:
    print()
    print("=== Jev chunk validation pilot ===")
    print(
        f"Sources: {report['sources']}  chunks sampled: {report['chunks']}  "
        f"scored: {report['scored']}  jev errors: {report['errors']}"
    )
    print()
    print("Distribution of min(complete, coherent, substantive):")
    bins = len(report["histogram"])
    peak = max(report["histogram"]) or 1
    for index, count in enumerate(report["histogram"]):
        low, high = index / bins, (index + 1) / bins
        bar = "#" * int(40 * count / peak)
        print(f"  [{low:.1f}, {high:.1f}{']' if index == bins - 1 else ')'} {count:>6}  {bar}")
    print("Mean probability per criterion:")
    for name, mean in report["criterion_means"].items():
        print(f"  {name:<12} {mean:.3f}" if mean is not None else f"  {name:<12} n/a")
    print()
    print(f"Skips by threshold (max skip fraction per source {max_skip_fraction}):")
    print("  threshold  skipped chunks  (% of scored)  sources affected  sources capped")
    for row in report["thresholds"]:
        print(
            f"  {row['threshold']:>9.2f}  {row['skipped_chunks']:>14}  "
            f"{row['skipped_fraction']:>12.1%}  {row['affected_sources']:>16}  "
            f"{row['capped_sources']:>14}"
        )
    shown = report["lowest"][:examples]
    if shown:
        print()
        print(f"{len(shown)} lowest-scoring chunks:")
        for r in shown:
            probs = ", ".join(f"{k}={v:.2f}" for k, v in r["probs"].items())
            print(f"  [{r['min_p']:.2f}] {r['chunk_id']} ({probs})")
            print(f"      {_one_line(r['text'])}")
    print()
    print(f"Jev usage cost: ${report['cost_usd']:.6f}")


def _load_sample(sources: int, chunks_per_source: int) -> tuple[list[dict], set[str]]:
    with get_connection() as conn:
        try:
            conn.read_only = True  # belt and braces: the session cannot write
        except Exception:
            pass
        with conn.cursor() as cur:
            cur.execute(_SAMPLE_SQL, (sources, chunks_per_source))
            rows = [
                {"chunk_id": r[0], "source_id": r[1], "text": r[2]} for r in cur.fetchall()
            ]
            source_ids = sorted({r["source_id"] for r in rows})
            protected: set[str] = set()
            if source_ids:
                cur.execute(_PARENTS_SQL, (source_ids,))
                protected = {r[0] for r in cur.fetchall()}
    return rows, protected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only pilot: what Jev chunk validation would skip on the existing corpus.",
    )
    parser.add_argument("--sources", type=int, default=200, help="Most recent sources to sample (default 200).")
    parser.add_argument("--chunks-per-source", type=int, default=20, help="Max random chunks scored per source (default 20).")
    parser.add_argument("--workers", type=int, default=None, help="Concurrent Jev calls (default JEV_CONCURRENCY).")
    parser.add_argument("--examples", type=int, default=8, help="Low-scoring chunk examples to print (default 8).")
    parser.add_argument(
        "--max-skip-fraction", type=float, default=None,
        help="Per-source cap (default CHUNK_VALIDATION_MAX_SKIP_FRACTION).",
    )
    args = parser.parse_args(argv)

    max_skip_fraction = (
        settings.CHUNK_VALIDATION_MAX_SKIP_FRACTION
        if args.max_skip_fraction is None
        else args.max_skip_fraction
    )
    rows, protected = _load_sample(args.sources, args.chunks_per_source)
    print(f"Sampled {len(rows)} chunks from up to {args.sources} recent sources.")
    if not rows:
        print("Nothing to do.")
        return 0

    workers = max(1, args.workers or settings.JEV_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scored = list(pool.map(score_chunk, rows))

    report = analyze(scored, protected, max_skip_fraction=max_skip_fraction)
    print_report(report, examples=args.examples, max_skip_fraction=max_skip_fraction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
