#!/usr/bin/env python3
"""
Profiling script for `rag retrieve`.
Monkey-patches key functions in retrieval.py and embedding.py to record
wall-clock timing, then prints a structured report.

Usage:
    python scripts/profile_retrieve.py "your query here"
"""
import sys
import time
import json
from collections import defaultdict

# ── Timing registry ──────────────────────────────────────────────────────────

_timings: list[dict] = []
_call_counts: dict[str, int] = defaultdict(int)


def _record(name: str, elapsed_ms: float, **meta):
    _call_counts[name] += 1
    _timings.append({"fn": name, "ms": round(elapsed_ms, 1), **meta})


def _timed(name: str, fn, *args, **kwargs):
    t0 = time.monotonic()
    result = fn(*args, **kwargs)
    elapsed = (time.monotonic() - t0) * 1000
    _record(name, elapsed)
    return result


# ── Patch helpers ─────────────────────────────────────────────────────────────

def _wrap(module, fn_name: str, label: str | None = None):
    """Replace module.fn_name with a timing wrapper."""
    orig = getattr(module, fn_name)
    tag = label or fn_name

    def wrapper(*args, **kwargs):
        t0 = time.monotonic()
        result = orig(*args, **kwargs)
        elapsed = (time.monotonic() - t0) * 1000
        _record(tag, elapsed)
        return result

    setattr(module, fn_name, wrapper)
    return orig


# ── Apply patches before importing retrieval ─────────────────────────────────

import rag.embedding as _emb_mod

_orig_get_embeddings = _emb_mod.get_embeddings


def _patched_get_embeddings(texts):
    t0 = time.monotonic()
    result = _orig_get_embeddings(texts)
    elapsed = (time.monotonic() - t0) * 1000
    _record("get_embeddings", elapsed, n_texts=len(texts))
    return result


_emb_mod.get_embeddings = _patched_get_embeddings

import rag.retrieval as _ret_mod

# Patch individual functions ─ we capture them before wrapping

_orig_chat_json = _ret_mod._chat_json


def _patched_chat_json(model, prompt, *, timeout=60):
    t0 = time.monotonic()
    result = _orig_chat_json(model, prompt, timeout=timeout)
    elapsed = (time.monotonic() - t0) * 1000
    label = "llm_query_variants" if timeout == 90 else "llm_graph"
    _record(label, elapsed, model=model)
    return result


_ret_mod._chat_json = _patched_chat_json

_orig_rerank_documents = _ret_mod.rerank_documents


def _patched_rerank_documents(query, documents, *, top_n=None):
    t0 = time.monotonic()
    result = _orig_rerank_documents(query, documents, top_n=top_n)
    elapsed = (time.monotonic() - t0) * 1000
    _record("rerank_documents", elapsed, n_docs=len(documents))
    return result


_ret_mod.rerank_documents = _patched_rerank_documents

_orig_dense_retrieve = _ret_mod.dense_retrieve


def _patched_dense_retrieve(conn, query_text, *, source_ids, filters, top_n):
    t0 = time.monotonic()
    result = _orig_dense_retrieve(conn, query_text, source_ids=source_ids, filters=filters, top_n=top_n)
    elapsed = (time.monotonic() - t0) * 1000
    _record("dense_retrieve", elapsed, n_results=len(result))
    return result


_ret_mod.dense_retrieve = _patched_dense_retrieve

_orig_sparse_retrieve = _ret_mod.sparse_retrieve


def _patched_sparse_retrieve(conn, query_text, *, source_ids, filters, top_n):
    t0 = time.monotonic()
    result = _orig_sparse_retrieve(conn, query_text, source_ids=source_ids, filters=filters, top_n=top_n)
    elapsed = (time.monotonic() - t0) * 1000
    _record("sparse_retrieve", elapsed, n_results=len(result))
    return result


_ret_mod.sparse_retrieve = _patched_sparse_retrieve

_orig_load_seed_entities = _ret_mod._load_seed_entities


def _patched_load_seed_entities(driver, chunk_id):
    t0 = time.monotonic()
    result = _orig_load_seed_entities(driver, chunk_id)
    elapsed = (time.monotonic() - t0) * 1000
    _record("graph_load_seed_entities", elapsed, n_entities=len(result))
    return result


_ret_mod._load_seed_entities = _patched_load_seed_entities

_orig_load_second_hop = _ret_mod._load_second_hop_entities


def _patched_load_second_hop(driver, entity_id, confidence_threshold):
    t0 = time.monotonic()
    result = _orig_load_second_hop(driver, entity_id, confidence_threshold)
    elapsed = (time.monotonic() - t0) * 1000
    _record("graph_load_second_hop_entities", elapsed, n_entities=len(result))
    return result


_ret_mod._load_second_hop_entities = _patched_load_second_hop

_orig_load_chunk_ids = _ret_mod._load_chunk_ids_for_entity


def _patched_load_chunk_ids(driver, entity_id):
    t0 = time.monotonic()
    result = _orig_load_chunk_ids(driver, entity_id)
    elapsed = (time.monotonic() - t0) * 1000
    _record("graph_load_chunk_ids_for_entity", elapsed, n_ids=len(result))
    return result


_ret_mod._load_chunk_ids_for_entity = _patched_load_chunk_ids

_orig_fetch_candidates = _ret_mod._fetch_chunk_candidates_by_ids


def _patched_fetch_candidates(conn, chunk_ids, query_text, *, source_ids, filters, limit):
    t0 = time.monotonic()
    result = _orig_fetch_candidates(conn, chunk_ids, query_text, source_ids=source_ids, filters=filters, limit=limit)
    elapsed = (time.monotonic() - t0) * 1000
    _record("sql_fetch_chunk_candidates", elapsed, n_ids_in=len(chunk_ids), n_results=len(result))
    return result


_ret_mod._fetch_chunk_candidates_by_ids = _patched_fetch_candidates

_orig_fetch_neighbors = _ret_mod._fetch_same_source_neighbor_candidates


def _patched_fetch_neighbors(conn, seed, query_text, *, source_ids, filters, limit):
    t0 = time.monotonic()
    result = _orig_fetch_neighbors(conn, seed, query_text, source_ids=source_ids, filters=filters, limit=limit)
    elapsed = (time.monotonic() - t0) * 1000
    _record("sql_fetch_same_source_neighbors", elapsed, n_results=len(result))
    return result


_ret_mod._fetch_same_source_neighbor_candidates = _patched_fetch_neighbors

_orig_expand_seed = _ret_mod.expand_seed_candidate


def _patched_expand_seed(seed, query, source_ids, filters, entity_conf, first_sim, second_sim,
                          *, conn, driver, trace_logger=None, budget=None):
    t0 = time.monotonic()
    result = _orig_expand_seed(seed, query, source_ids, filters, entity_conf, first_sim, second_sim,
                                conn=conn, driver=driver, trace_logger=trace_logger, budget=budget)
    elapsed = (time.monotonic() - t0) * 1000
    _record("expand_seed_candidate", elapsed, chunk_id=seed.chunk_id[:8])
    return result


_ret_mod.expand_seed_candidate = _patched_expand_seed

_orig_generate_variants = _ret_mod.generate_query_variants


def _patched_generate_variants(query, trace_logger=None):
    t0 = time.monotonic()
    result = _orig_generate_variants(query, trace_logger=trace_logger)
    elapsed = (time.monotonic() - t0) * 1000
    _record("generate_query_variants", elapsed, n_variants=len(result))
    return result


_ret_mod.generate_query_variants = _patched_generate_variants

_orig_run_first_stage = _ret_mod.run_first_stage_retrieval


def _patched_run_first_stage(**kwargs):
    t0 = time.monotonic()
    result = _orig_run_first_stage(**kwargs)
    elapsed = (time.monotonic() - t0) * 1000
    _record("run_first_stage_retrieval", elapsed, n_fused=len(result))
    return result


_ret_mod.run_first_stage_retrieval = _patched_run_first_stage

_orig_finalize = _ret_mod.finalize_root_results


def _patched_finalize(query, root_results, result_count, trace_logger=None):
    t0 = time.monotonic()
    result = _orig_finalize(query, root_results, result_count, trace_logger=trace_logger)
    elapsed = (time.monotonic() - t0) * 1000
    _record("finalize_root_results", elapsed, n_results=len(result))
    return result


_ret_mod.finalize_root_results = _patched_finalize

# ── Run the query ─────────────────────────────────────────────────────────────

from rag.retrieval import retrieve

query = sys.argv[1] if len(sys.argv) > 1 else "what are the most underrated ai use case for insurance that can be leveraged in manufacturing"

print(f"\nProfiling query: {query!r}\n")

t_total_start = time.monotonic()
result = retrieve(
    query=query,
    source_ids=[],
    filters={},
    seed_count=None,
    result_count=None,
    rrf_k=None,
    entity_confidence_threshold=None,
    first_hop_similarity_threshold=None,
    second_hop_similarity_threshold=None,
    trace=False,
    trace_printer=None,
)
total_ms = (time.monotonic() - t_total_start) * 1000

# ── Report ────────────────────────────────────────────────────────────────────

GROUPS = {
    "LLM calls": ["generate_query_variants", "llm_query_variants", "llm_graph"],
    "Embedding API": ["get_embeddings"],
    "Reranker API": ["rerank_documents"],
    "Graph DB (Cypher)": [
        "graph_load_seed_entities",
        "graph_load_second_hop_entities",
        "graph_load_chunk_ids_for_entity",
    ],
    "Postgres (SQL)": [
        "dense_retrieve",
        "sparse_retrieve",
        "sql_fetch_chunk_candidates",
        "sql_fetch_same_source_neighbors",
    ],
    "Pipeline stages": [
        "generate_query_variants",
        "run_first_stage_retrieval",
        "expand_seed_candidate",
        "finalize_root_results",
    ],
}

print("=" * 72)
print(f"TOTAL wall-clock time: {total_ms/1000:.2f}s  ({total_ms:.0f}ms)")
print("=" * 72)

# Per-call breakdown
print("\n── Per-call timing (all events in order) ──────────────────────────────")
print(f"{'Function':<42} {'ms':>8}  {'meta'}")
print("-" * 72)
for entry in _timings:
    fn = entry["fn"]
    ms = entry["ms"]
    meta = {k: v for k, v in entry.items() if k not in ("fn", "ms")}
    meta_str = "  ".join(f"{k}={v}" for k, v in meta.items())
    print(f"{fn:<42} {ms:>8.1f}  {meta_str}")

# Aggregate by function
print("\n── Aggregate by function ───────────────────────────────────────────────")
agg: dict[str, dict] = {}
for entry in _timings:
    fn = entry["fn"]
    if fn not in agg:
        agg[fn] = {"calls": 0, "total_ms": 0.0, "min_ms": float("inf"), "max_ms": 0.0}
    agg[fn]["calls"] += 1
    agg[fn]["total_ms"] += entry["ms"]
    agg[fn]["min_ms"] = min(agg[fn]["min_ms"], entry["ms"])
    agg[fn]["max_ms"] = max(agg[fn]["max_ms"], entry["ms"])

print(f"{'Function':<42} {'calls':>5} {'total_ms':>10} {'avg_ms':>8} {'max_ms':>8}  {'% total':>7}")
print("-" * 90)
sorted_agg = sorted(agg.items(), key=lambda x: x[1]["total_ms"], reverse=True)
for fn, stats in sorted_agg:
    avg = stats["total_ms"] / stats["calls"]
    pct = stats["total_ms"] / total_ms * 100
    print(f"{fn:<42} {stats['calls']:>5} {stats['total_ms']:>10.0f} {avg:>8.1f} {stats['max_ms']:>8.1f}  {pct:>6.1f}%")

# Group summary
print("\n── Time by category ────────────────────────────────────────────────────")
cat_totals: dict[str, float] = {}
for group, fns in GROUPS.items():
    if group == "Pipeline stages":
        continue
    total = sum(agg.get(fn, {}).get("total_ms", 0.0) for fn in fns)
    cat_totals[group] = total

# LLM is split — count generate_query_variants only once
cat_totals["LLM calls"] = (
    agg.get("llm_graph", {}).get("total_ms", 0.0)
    + agg.get("generate_query_variants", {}).get("total_ms", 0.0)
)
cat_totals["Embedding API"] = agg.get("get_embeddings", {}).get("total_ms", 0.0)
cat_totals["Reranker API"] = agg.get("rerank_documents", {}).get("total_ms", 0.0)

graph_total = sum(agg.get(fn, {}).get("total_ms", 0.0) for fn in GROUPS["Graph DB (Cypher)"])
sql_total = sum(agg.get(fn, {}).get("total_ms", 0.0) for fn in ["dense_retrieve", "sparse_retrieve",
    "sql_fetch_chunk_candidates", "sql_fetch_same_source_neighbors"])
cat_totals["Graph DB (Cypher)"] = graph_total
cat_totals["Postgres (SQL)"] = sql_total

for cat, ms in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True):
    pct = ms / total_ms * 100
    bar = "█" * int(pct / 2)
    print(f"  {cat:<25} {ms:>8.0f}ms  {pct:>5.1f}%  {bar}")

# Pipeline stage durations
print("\n── Pipeline stage durations ─────────────────────────────────────────────")
for fn in ["generate_query_variants", "run_first_stage_retrieval", "expand_seed_candidate", "finalize_root_results"]:
    stats = agg.get(fn, {})
    if not stats:
        continue
    pct = stats["total_ms"] / total_ms * 100
    print(f"  {fn:<42} {stats['total_ms']:>8.0f}ms ({stats['calls']} call(s))  {pct:.1f}%")

print("\n── Result count ────────────────────────────────────────────────────────")
print(f"  Results returned: {len(result.get('retrieval_results', []))}")

print()
