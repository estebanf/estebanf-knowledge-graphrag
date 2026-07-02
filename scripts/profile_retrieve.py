#!/usr/bin/env python3
"""Profile wall-clock timing for ``rag retrieve`` against the live stack."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any


_timings: list[dict[str, Any]] = []
_call_counts: dict[str, int] = defaultdict(int)


def _record(name: str, elapsed_ms: float, **meta: Any) -> None:
    _call_counts[name] += 1
    _timings.append({"fn": name, "ms": round(elapsed_ms, 1), **meta})


def _wrap(module: Any, fn_name: str, *, label: str | None = None, meta: Callable[..., dict] | None = None) -> None:
    original = getattr(module, fn_name)
    tag = label or fn_name

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started_at = time.monotonic()
        result = original(*args, **kwargs)
        elapsed_ms = (time.monotonic() - started_at) * 1000
        event_meta = meta(args, kwargs, result) if meta else {}
        _record(tag, elapsed_ms, **event_meta)
        return result

    setattr(module, fn_name, wrapper)


def _patch_retrieval() -> Any:
    import rag.retrieval as retrieval

    _wrap(
        retrieval,
        "get_embeddings",
        meta=lambda args, kwargs, result: {"n_texts": len(args[0]) if args else 0},
    )
    _wrap(
        retrieval,
        "_chat_json_opencode",
        label="llm_opencode_json",
        meta=lambda args, kwargs, result: {"model": args[0] if args else ""},
    )
    _wrap(
        retrieval,
        "_chat_json",
        label="llm_openrouter_json",
        meta=lambda args, kwargs, result: {"model": args[0] if args else ""},
    )
    _wrap(
        retrieval,
        "rerank_documents",
        meta=lambda args, kwargs, result: {"n_docs": len(args[1]) if len(args) > 1 else 0},
    )
    _wrap(
        retrieval,
        "generate_query_variants",
        meta=lambda args, kwargs, result: {"n_variants": len(result)},
    )
    _wrap(
        retrieval,
        "generate_insight_query_variants",
        meta=lambda args, kwargs, result: {"n_variants": len(result)},
    )
    _wrap(
        retrieval,
        "dense_retrieve",
        meta=lambda args, kwargs, result: {"n_results": len(result), "query": (args[1] if len(args) > 1 else "")[:60]},
    )
    _wrap(
        retrieval,
        "sparse_retrieve",
        meta=lambda args, kwargs, result: {"n_results": len(result), "query": (args[1] if len(args) > 1 else "")[:60]},
    )
    _wrap(
        retrieval,
        "insight_dense_retrieve",
        meta=lambda args, kwargs, result: {"n_results": len(result)},
    )
    _wrap(
        retrieval,
        "insight_sparse_retrieve",
        meta=lambda args, kwargs, result: {"n_results": len(result), "query": (args[1] if len(args) > 1 else "")[:60]},
    )
    _wrap(
        retrieval,
        "insight_hybrid_search",
        meta=lambda args, kwargs, result: {"n_results": len(result), "query": (args[0] if args else "")[:60]},
    )
    _wrap(
        retrieval,
        "run_first_stage_retrieval",
        meta=lambda args, kwargs, result: {"n_fused": len(result)},
    )
    _wrap(
        retrieval,
        "run_insight_first_stage_retrieval",
        meta=lambda args, kwargs, result: {"n_fused": len(result)},
    )
    _wrap(
        retrieval,
        "_load_seed_entities",
        meta=lambda args, kwargs, result: {"n_entities": len(result)},
    )
    _wrap(
        retrieval,
        "_load_chunk_ids_for_entity",
        meta=lambda args, kwargs, result: {"n_ids": len(result), "entity": args[1] if len(args) > 1 else ""},
    )
    _wrap(
        retrieval,
        "_load_related_insights",
        meta=lambda args, kwargs, result: {"n_insights": len(result)},
    )
    _wrap(
        retrieval,
        "_fetch_chunk_candidates_by_ids",
        meta=lambda args, kwargs, result: {"n_ids_in": len(args[1]) if len(args) > 1 else 0, "n_results": len(result)},
    )
    _wrap(
        retrieval,
        "_fetch_same_source_neighbor_candidates",
        meta=lambda args, kwargs, result: {"n_results": len(result)},
    )
    _wrap(
        retrieval,
        "_fetch_insight_sources_and_topics",
        meta=lambda args, kwargs, result: {"n_ids": len(args[1]) if len(args) > 1 else 0},
    )
    _wrap(
        retrieval,
        "_generate_insight_sub_query",
        meta=lambda args, kwargs, result: {"has_query": bool(result)},
    )
    _wrap(
        retrieval,
        "expand_seed_candidate",
        meta=lambda args, kwargs, result: {"chunk_id": getattr(args[0], "chunk_id", "")[:8] if args else ""},
    )
    _wrap(
        retrieval,
        "expand_seed_insight",
        meta=lambda args, kwargs, result: {"insight_id": getattr(args[0], "insight_id", "")[:8] if args else ""},
    )
    _wrap(
        retrieval,
        "finalize_root_results",
        meta=lambda args, kwargs, result: {"n_results": len(result)},
    )
    _wrap(
        retrieval,
        "finalize_insight_results",
        meta=lambda args, kwargs, result: {"n_results": len(result)},
    )
    return retrieval


def _print_report(total_ms: float, result: dict[str, Any]) -> None:
    print("=" * 90)
    print(f"TOTAL wall-clock time: {total_ms / 1000:.2f}s ({total_ms:.0f}ms)")
    print("=" * 90)

    print("\nPer-call timing")
    print(f"{'Function':<38} {'ms':>8}  meta")
    print("-" * 90)
    for entry in _timings:
        meta = {k: v for k, v in entry.items() if k not in {"fn", "ms"}}
        meta_str = "  ".join(f"{k}={v}" for k, v in meta.items())
        print(f"{entry['fn']:<38} {entry['ms']:>8.1f}  {meta_str}")

    aggregate: dict[str, dict[str, float]] = {}
    for entry in _timings:
        fn = entry["fn"]
        stats = aggregate.setdefault(fn, {"calls": 0, "total_ms": 0.0, "max_ms": 0.0})
        stats["calls"] += 1
        stats["total_ms"] += float(entry["ms"])
        stats["max_ms"] = max(stats["max_ms"], float(entry["ms"]))

    print("\nAggregate by function")
    print(f"{'Function':<38} {'calls':>5} {'total_ms':>10} {'avg_ms':>8} {'max_ms':>8} {'% total':>8}")
    print("-" * 90)
    for fn, stats in sorted(aggregate.items(), key=lambda item: item[1]["total_ms"], reverse=True):
        avg = stats["total_ms"] / stats["calls"]
        pct = stats["total_ms"] / total_ms * 100 if total_ms else 0
        print(
            f"{fn:<38} {stats['calls']:>5.0f} {stats['total_ms']:>10.0f} "
            f"{avg:>8.1f} {stats['max_ms']:>8.1f} {pct:>7.1f}%"
        )

    groups = {
        "Variant generation": ["generate_query_variants", "generate_insight_query_variants"],
        "Embedding API": ["get_embeddings"],
        "Reranker API": ["rerank_documents"],
        "Chunk SQL": [
            "dense_retrieve",
            "sparse_retrieve",
            "_fetch_chunk_candidates_by_ids",
            "_fetch_same_source_neighbor_candidates",
        ],
        "Insight SQL": [
            "insight_dense_retrieve",
            "insight_sparse_retrieve",
            "insight_hybrid_search",
            "_fetch_insight_sources_and_topics",
        ],
        "Graph DB": ["_load_seed_entities", "_load_chunk_ids_for_entity", "_load_related_insights"],
        "Expansion": ["expand_seed_candidate", "expand_seed_insight"],
        "Finalization": ["finalize_root_results", "finalize_insight_results"],
    }

    print("\nTime by category")
    for group, names in sorted(
        ((group, names) for group, names in groups.items()),
        key=lambda item: sum(aggregate.get(fn, {}).get("total_ms", 0.0) for fn in item[1]),
        reverse=True,
    ):
        ms = sum(aggregate.get(fn, {}).get("total_ms", 0.0) for fn in names)
        pct = ms / total_ms * 100 if total_ms else 0
        bar = "#" * int(pct / 2)
        print(f"  {group:<22} {ms:>8.0f}ms {pct:>6.1f}%  {bar}")

    print("\nResult count")
    print(f"  Chunks:   {len(result.get('retrieval_results', []))}")
    print(f"  Insights: {len(result.get('insights', []))}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile rag.retrieve() against configured services.")
    parser.add_argument("query", nargs="?", default="insurance triage")
    parser.add_argument("--seed-count", type=int, default=None, help="Override retrieval seed count.")
    parser.add_argument("--result-count", type=int, default=None, help="Override retrieval result count.")
    parser.add_argument("--trace", action="store_true", help="Print retrieval trace messages while profiling.")
    args = parser.parse_args()

    retrieval = _patch_retrieval()

    print(f"\nProfiling query: {args.query!r}\n")
    started_at = time.monotonic()
    result = retrieval.retrieve(
        query=args.query,
        source_ids=[],
        filters={},
        seed_count=args.seed_count,
        result_count=args.result_count,
        rrf_k=None,
        entity_confidence_threshold=None,
        first_hop_similarity_threshold=None,
        second_hop_similarity_threshold=None,
        trace=args.trace,
        trace_printer=print if args.trace else None,
    )
    total_ms = (time.monotonic() - started_at) * 1000
    _print_report(total_ms, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
