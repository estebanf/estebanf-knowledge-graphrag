#!/usr/bin/env python3
"""Pilot: compare the Jev profiling/metadata path against the legacy LLM path.

READ-ONLY BY CONSTRUCTION. The script only ever runs a single SELECT against
Postgres (and marks the connection read-only), never writes to Postgres, never
touches Memgraph and never changes a source's metadata or chunks. There is no
`--dry-run` flag because there is nothing to dry-run: the only side effects
are outbound OpenRouter calls (the legacy profiler/metadata LLM calls and the
Jev calls), which cost a small amount of OpenRouter credit.

For the `--limit N` most recent sources with `markdown_content` it runs

* the LEGACY path: `profile_document` + `extract_metadata` (two LLM calls, run
  one after the other like the pipeline's profiling and parsing stages), and
* the JEV path: `extract_profile_and_metadata` (one Jev call plus the small
  author/source LLM call, concurrently, exactly as the parsing stage does),

and reports what matters for `PROFILING_BACKEND=jev`:

* chunking decisions derived from the profile: `select_strategy(profile)`,
  `use_hierarchical` (avg_section_length == "long") and `use_propositions`
  (profile domain in legal/financial/medical), plus per-field agreement,
* kind / metadata-domain agreement,
* Jev usage cost and per-path latency.

Note: the legacy profiler silently returns a default profile when its LLM call
fails; sources where the legacy profile equals that default are counted so the
agreement figures can be read with that caveat.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.chunking import select_strategy
from rag.db import get_connection
from rag.jev_profiling import extract_profile_and_metadata
from rag.metadata_extraction import extract_metadata
from rag.profiling import _DEFAULT_PROFILE, profile_document

_SOURCES_SQL = """
SELECT id::text, name, markdown_content
FROM sources
WHERE deleted_at IS NULL
  AND markdown_content IS NOT NULL
  AND markdown_content <> ''
ORDER BY created_at DESC, id DESC
LIMIT %s
"""

_PROFILE_FIELDS = (
    "structure_type",
    "heading_consistency",
    "content_density",
    "primary_content_type",
    "avg_section_length",
    "has_tables",
    "has_code_blocks",
    "domain",
)


def chunking_decisions(profile) -> dict:
    """The profile-driven chunking switches (domain predicate only; the real
    `use_propositions` additionally requires an OpenRouter key)."""
    return {
        "strategy": select_strategy(profile),
        "hierarchical": profile.avg_section_length == "long",
        "propositions": profile.domain in ("legal", "financial", "medical"),
    }


def evaluate_source(source_id: str, name: str, markdown: str) -> dict:
    """Run both paths for one source. Never raises: failures are recorded."""
    row: dict = {"source_id": source_id, "name": name}

    started = time.perf_counter()
    try:
        llm_profile = profile_document(markdown)
        llm_metadata = extract_metadata(markdown)
        row["llm"] = {"profile": llm_profile, "metadata": llm_metadata}
    except Exception as exc:  # legacy helpers swallow errors; belt and braces
        row["llm_error"] = str(exc)
    row["llm_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    try:
        jev_metadata, classification = extract_profile_and_metadata(markdown)
        row["jev"] = {
            "profile": classification.profile,
            "metadata": jev_metadata,
            "kind": classification.kind,
            "metadata_domain": classification.metadata_domain,
            "cost": float((classification.usage or {}).get("cost") or 0.0),
            "confidence": classification.confidence,
        }
    except Exception as exc:
        row["jev_error"] = str(exc)
    row["jev_seconds"] = time.perf_counter() - started
    return row


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summarize(rows: list[dict]) -> dict:
    """Aggregate per-source results into the comparison report (pure)."""
    both = [r for r in rows if "llm" in r and "jev" in r]
    summary: dict = {
        "sources": len(rows),
        "compared": len(both),
        "llm_errors": sum(1 for r in rows if "llm_error" in r),
        "jev_errors": sum(1 for r in rows if "jev_error" in r),
        "llm_default_profiles": sum(
            1 for r in both if r["llm"]["profile"] == _DEFAULT_PROFILE
        ),
        "jev_cost_usd": sum(r["jev"]["cost"] for r in both),
    }

    def agreement(match) -> dict:
        hits = sum(1 for r in both if match(r))
        return {"agree": hits, "total": len(both), "rate": (hits / len(both)) if both else None}

    decisions = {
        r["source_id"]: (
            chunking_decisions(r["llm"]["profile"]),
            chunking_decisions(r["jev"]["profile"]),
        )
        for r in both
    }
    summary["strategy"] = agreement(lambda r: decisions[r["source_id"]][0]["strategy"] == decisions[r["source_id"]][1]["strategy"])
    summary["use_hierarchical"] = agreement(lambda r: decisions[r["source_id"]][0]["hierarchical"] == decisions[r["source_id"]][1]["hierarchical"])
    summary["use_propositions"] = agreement(lambda r: decisions[r["source_id"]][0]["propositions"] == decisions[r["source_id"]][1]["propositions"])
    summary["kind"] = agreement(lambda r: (r["llm"]["metadata"].get("kind") or "") == (r["jev"]["kind"] or ""))
    summary["metadata_domain"] = agreement(lambda r: (r["llm"]["metadata"].get("domain") or "") == (r["jev"]["metadata_domain"] or ""))
    summary["fields"] = {
        field: agreement(lambda r, f=field: getattr(r["llm"]["profile"], f) == getattr(r["jev"]["profile"], f))
        for field in _PROFILE_FIELDS
    }
    summary["strategy_confusion"] = dict(
        Counter(
            f"{decisions[r['source_id']][0]['strategy']} -> {decisions[r['source_id']][1]['strategy']}"
            for r in both
        )
    )
    summary["strategy_counts"] = {
        "llm": dict(Counter(decisions[r["source_id"]][0]["strategy"] for r in both)),
        "jev": dict(Counter(decisions[r["source_id"]][1]["strategy"] for r in both)),
    }

    latency: dict = {}
    for path in ("llm", "jev"):
        values = [r[f"{path}_seconds"] for r in rows if path in r]
        if values:
            latency[path] = {
                "mean": statistics.fmean(values),
                "p50": _percentile(values, 0.5),
                "p90": _percentile(values, 0.9),
            }
    summary["latency_seconds"] = latency
    summary["disagreements"] = [
        r for r in both
        if decisions[r["source_id"]][0] != decisions[r["source_id"]][1]
    ]
    summary["_decisions"] = decisions
    return summary


def _fmt_rate(entry: dict) -> str:
    if entry["rate"] is None:
        return "n/a"
    return f"{entry['agree']}/{entry['total']} ({entry['rate']:.0%})"


def print_report(summary: dict, examples: int) -> None:
    print()
    print("=== Jev vs LLM profiling pilot ===")
    print(
        f"Sources: {summary['sources']}  compared: {summary['compared']}  "
        f"llm errors: {summary['llm_errors']}  jev errors: {summary['jev_errors']}"
    )
    print(
        f"Legacy profiles equal to the default (likely failed LLM calls): "
        f"{summary['llm_default_profiles']}"
    )
    print()
    print("Agreement on fields that drive chunking:")
    print(f"  select_strategy(profile)  {_fmt_rate(summary['strategy'])}")
    print(f"  use_hierarchical          {_fmt_rate(summary['use_hierarchical'])}")
    print(f"  use_propositions          {_fmt_rate(summary['use_propositions'])}")
    print("Agreement on metadata:")
    print(f"  kind                      {_fmt_rate(summary['kind'])}")
    print(f"  domain (metadata)         {_fmt_rate(summary['metadata_domain'])}")
    print("Per-profile-field agreement:")
    for field, entry in summary["fields"].items():
        print(f"  {field:<22} {_fmt_rate(entry)}")
    print()
    print("Strategy counts: " + str(summary["strategy_counts"]))
    print("Strategy transitions (llm -> jev):")
    for transition, count in sorted(summary["strategy_confusion"].items(), key=lambda kv: -kv[1]):
        print(f"  {transition}: {count}")
    print()
    print(f"Jev usage cost: ${summary['jev_cost_usd']:.6f}")
    for path, stats in summary["latency_seconds"].items():
        print(
            f"Latency {path:>3}: mean {stats['mean']:.2f}s  "
            f"p50 {stats['p50']:.2f}s  p90 {stats['p90']:.2f}s"
        )
    disagreements = summary["disagreements"][:examples]
    if disagreements:
        print()
        print(f"First {len(disagreements)} sources whose chunking decisions differ:")
        for row in disagreements:
            llm_d, jev_d = summary["_decisions"][row["source_id"]]
            print(f"  {row['source_id']} {row['name']!r}")
            print(f"    llm: {llm_d}")
            print(f"    jev: {jev_d}")


def _load_sources(limit: int) -> list[tuple[str, str, str]]:
    with get_connection() as conn:
        try:
            conn.read_only = True  # belt and braces: the session cannot write
        except Exception:
            pass
        with conn.cursor() as cur:
            cur.execute(_SOURCES_SQL, (limit,))
            return [(row[0], row[1], row[2]) for row in cur.fetchall()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only pilot comparing Jev profiling/metadata with the legacy LLM path.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Most recent sources to compare (default 100).")
    parser.add_argument("--workers", type=int, default=4, help="Sources processed concurrently (default 4).")
    parser.add_argument("--examples", type=int, default=10, help="Disagreement examples to print (default 10).")
    args = parser.parse_args(argv)

    sources = _load_sources(args.limit)
    print(f"Loaded {len(sources)} sources with markdown (limit {args.limit}).")
    if not sources:
        print("Nothing to do.")
        return 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(lambda s: evaluate_source(*s), sources))

    print_report(summarize(rows), args.examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
