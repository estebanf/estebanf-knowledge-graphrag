"""Jev confirmation of merge candidates (`rag.jev_dedup`) and its use by
`run_entity_merge`: score-band routing, review-table inserts, only confirmed
pairs clustered, Jev errors never merge, Jev off keeps today's behavior.
Offline: Jev, Postgres and Memgraph are doubles."""
import importlib.util
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from audit_fakes import FakeConn, FakeDriver, FakeSession, ctx
from rag import jev
from rag import jev_dedup as jd
from rag.jev import JevError, JevResponse

NOW = datetime(2026, 1, 1)


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(jev, "_new_client", boom)
    monkeypatch.setattr(jd.settings, "OPENROUTER_API_KEY", "or-test")


def _resp(score, name_variant=0.9, same_type=0.95):
    return JevResponse(
        answers={
            "alignment": {"type": "score", "score": score, "confidence": 0.9},
            "same_name_variant": {"type": "noul", "noul": name_variant},
            "same_type": {"type": "noul", "noul": same_type},
        },
        usage={"cost": 0.0001},
    )


def _meta(eid, name, etype="ORGANIZATION", aliases=()):
    return {
        "id": eid,
        "canonical_name": name,
        "entity_type": etype,
        "aliases": list(aliases),
        "has_embedding": True,
        "created_at": NOW,
    }


# ── Routing and helpers ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "score,route",
    [
        (2.0, jd.ROUTE_MERGE),
        (1.5, jd.ROUTE_MERGE),
        (1.4999, jd.ROUTE_REVIEW),
        (1.0, jd.ROUTE_REVIEW),
        (0.5, jd.ROUTE_REVIEW),
        (0.4999, jd.ROUTE_DROP),
        (0.0, jd.ROUTE_DROP),
    ],
)
def test_route_score_cut_points(score, route):
    assert jd.route_score(score) == route


def test_normalize_and_dedupe_pairs_keep_the_highest_similarity():
    assert jd.normalize_pair("b", "a") == ("a", "b") == jd.normalize_pair("a", "b")
    pairs = [("b", "a", 0.90), ("a", "b", 0.93), ("c", "c", 0.99), ("a", "d", 0.91)]
    assert sorted(jd.dedupe_pairs(pairs)) == [("a", "b", 0.93), ("a", "d", 0.91)]


def test_entity_pair_state_and_questions_follow_the_alignment_recipe():
    a = _meta("a", "Open AI", aliases=[f"alias {i}" for i in range(25)])
    b = _meta("b", "OpenAI", etype="PRODUCT")
    state = jd.entity_pair_state(a, b)
    assert set(state) == {"entity_a", "entity_b"}
    assert state["entity_a"]["aliases"] == [f"alias {i}" for i in range(10)]
    assert state["entity_b"] == {"name": "OpenAI", "type": "PRODUCT", "aliases": []}

    q = jd.entity_pair_questions()
    assert set(q) == {"alignment", "same_name_variant", "same_type"}
    assert q["alignment"]["type"] == "score" and len(q["alignment"]["criteria"]) == 3
    assert q["alignment"]["criteria"][0].startswith("Two different")
    assert q["alignment"]["criteria"][2].startswith("One and the same")
    assert q["same_name_variant"]["type"] == q["same_type"]["type"] == "noul"

    iq = jd.insight_pair_questions()
    assert iq["alignment"]["criteria"] == jd.INSIGHT_ALIGNMENT_LEVELS
    assert jd.insight_pair_state("x" * 5000, "y") == {"insight_a": "x" * 1500, "insight_b": "y"}


def test_judge_entity_pairs_routes_by_score_band_and_isolates_errors(monkeypatch):
    meta = {e: _meta(e, f"name {e}") for e in "abcdefgh"}
    scores = {"name a": 1.9, "name c": 1.0, "name e": 0.1}

    def fake_many(items, **kwargs):
        out = []
        for state, _questions in items:
            names = {state["entity_a"]["name"], state["entity_b"]["name"]}
            if "name g" in names:
                out.append(JevError("HTTP 500", status_code=500))
            elif "name h" in names:
                out.append(JevResponse(answers={"alignment": {"type": "score"}}))  # no score
            else:
                out.append(_resp(next(scores[n] for n in names if n in scores)))
        return out

    monkeypatch.setattr(jev, "system_one_many", fake_many)
    pairs = [("a", "b", 0.97), ("c", "d", 0.95), ("e", "f", 0.94), ("g", "b", 0.94), ("h", "d", 0.94)]

    judgements, stats = jd.judge_entity_pairs(pairs, meta, progress=lambda m: None)

    routes = {(j.id_a, j.id_b): j.route for j in judgements}
    assert routes == {
        ("a", "b"): jd.ROUTE_MERGE,
        ("c", "d"): jd.ROUTE_REVIEW,
        ("e", "f"): jd.ROUTE_DROP,
        ("b", "g"): jd.ROUTE_ERROR,
        ("d", "h"): jd.ROUTE_ERROR,
    }
    assert jd.summarize(judgements) == {"merge": 1, "review": 1, "drop": 1, "error": 2}
    review = next(j for j in judgements if j.route == jd.ROUTE_REVIEW)
    assert review.score == 1.0
    assert "same_name_variant=0.9" in review.reason and "cosine=0.950" in review.reason
    assert stats.calls == 5 and stats.errors == 1 and stats.cost_usd == pytest.approx(0.0003)


def test_missing_metadata_is_an_error_not_a_merge(monkeypatch):
    monkeypatch.setattr(jev, "system_one_many", lambda items, **kw: [_resp(2.0) for _ in items])

    judgements, _ = jd.judge_entity_pairs([("a", "ghost", 0.99)], {"a": _meta("a", "A")}, progress=lambda m: None)

    assert [j.route for j in judgements] == [jd.ROUTE_ERROR]


def test_no_api_key_means_no_calls_and_everything_unconfirmed(monkeypatch):
    monkeypatch.setattr(jd.settings, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(jev, "system_one_many", MagicMock(side_effect=AssertionError("no call expected")))
    meta = {"a": _meta("a", "A"), "b": _meta("b", "B")}

    judgements, stats = jd.judge_entity_pairs([("a", "b", 0.99)], meta, progress=lambda m: None)

    assert [j.route for j in judgements] == [jd.ROUTE_ERROR]
    assert stats.calls == 0 and stats.errors == 1 and stats.aborted


def test_consecutive_fully_failed_batches_stop_further_calls(monkeypatch):
    monkeypatch.setattr(jd, "BATCH_SIZE", 2)
    calls = []

    def fake_many(items, **kwargs):
        calls.append(len(items))
        return [JevError("no credits", status_code=402) for _ in items]

    monkeypatch.setattr(jev, "system_one_many", fake_many)
    meta = {c: _meta(c, c) for c in "abcdefghij"}
    pairs = [(a, b, 0.97) for a, b in zip("aceg", "bdfh")] + [("i", "j", 0.97)]

    judgements, stats = jd.judge_entity_pairs(pairs, meta, progress=lambda m: None)

    assert calls == [2, 2]  # third batch never sent
    assert all(j.route == jd.ROUTE_ERROR for j in judgements) and len(judgements) == 5
    assert stats.aborted and stats.calls == 4 and stats.errors == 5


def test_judge_insight_pairs_routes_and_marks_missing_text(monkeypatch):
    scores = {"alpha": 1.8, "gamma": 0.9, "delta": 0.2}

    def fake_many(items, **kwargs):
        return [_resp(scores[state["insight_a"]]) for state, _ in items]

    monkeypatch.setattr(jev, "system_one_many", fake_many)
    texts = {"1": "alpha", "2": "alpha2", "3": "gamma", "4": "gamma2", "5": "delta", "6": "delta2"}
    pairs = [("1", "2", 0.94), ("3", "4", 0.9), ("5", "6", 0.89), ("1", "missing", 0.9)]

    judgements, _ = jd.judge_insight_pairs(pairs, texts, progress=lambda m: None)

    routes = {(j.id_a, j.id_b): j.route for j in judgements}
    assert routes[("1", "2")] == jd.ROUTE_MERGE
    assert routes[("3", "4")] == jd.ROUTE_REVIEW
    assert routes[("5", "6")] == jd.ROUTE_DROP
    assert routes[("1", "missing")] == jd.ROUTE_ERROR


# ── Review tables ────────────────────────────────────────────────────────────


def test_entity_review_insert_normalizes_order_commits_and_only_takes_the_review_band():
    conn = FakeConn()
    judgements = [
        jd.Judgement("z-id", "m-id", 0.95, jd.ROUTE_REVIEW, score=1.0, reason="why"),
        jd.Judgement("a-id", "b-id", 0.97, jd.ROUTE_MERGE, score=1.9),
        jd.Judgement("c-id", "d-id", 0.97, jd.ROUTE_ERROR, error="x"),
    ]

    assert jd.insert_entity_reviews(conn, judgements) == 1

    (sql, rows), = conn.executemany_calls
    assert "INSERT INTO entity_merge_review" in sql and "ON CONFLICT (entity_a, entity_b) DO NOTHING" in sql
    assert rows == [("m-id", "z-id", 1.0, "why")]
    assert conn.commits == 1  # get_connection() is not autocommit


def test_insight_review_insert_normalizes_order_and_stores_cosine_and_score():
    conn = FakeConn()
    judgements = [jd.Judgement("9", "1", 0.91, jd.ROUTE_REVIEW, score=0.8)]

    assert jd.insert_insight_reviews(conn, judgements) == 1

    (sql, rows), = conn.executemany_calls
    assert "INSERT INTO insight_pair_review" in sql and "ON CONFLICT (insight_a, insight_b) DO NOTHING" in sql
    assert rows == [("1", "9", 0.91, 0.8)]
    assert conn.commits == 1


def test_no_review_rows_means_no_write_at_all():
    conn = FakeConn()
    assert jd.insert_entity_reviews(conn, [jd.Judgement("a", "b", 0.9, jd.ROUTE_DROP, score=0.1)]) == 0
    assert jd.insert_insight_reviews(conn, []) == 0
    assert conn.executemany_calls == [] and conn.commits == 0


def test_reviewed_pairs_are_read_in_normalized_order():
    conn = FakeConn(lambda sql, params: [("z", "a")] if "entity_merge_review" in sql else [("9", "2")])
    assert jd.fetch_reviewed_entity_pairs(conn) == {("a", "z")}
    assert jd.fetch_reviewed_insight_pairs(conn) == {("2", "9")}


# ── run_entity_merge with Jev ────────────────────────────────────────────────


@pytest.fixture
def semantic(monkeypatch):
    from audit_fakes import forbid_live_stores

    path = Path(__file__).parent.parent / "scripts" / "merge_semantic_duplicates.py"
    spec = importlib.util.spec_from_file_location("merge_semantic_jev", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forbid_live_stores(monkeypatch, module)
    return module


ENTITY_ROWS = [
    ("a", "Acme Corp", "ORGANIZATION", [], True, NOW),
    ("b", "Acme Corporation", "ORGANIZATION", [], True, NOW),
    ("z", "Zeta Labs", "ORGANIZATION", [], True, NOW),
    ("m", "Zeta Lab", "ORGANIZATION", [], True, NOW),
    ("e", "Eon Systems", "ORGANIZATION", [], True, NOW),
    ("f", "Eon System", "ORGANIZATION", [], True, NOW),
    ("g", "Gamma Holdings", "ORGANIZATION", [], True, NOW),
    ("h", "Gamma Holding", "ORGANIZATION", [], True, NOW),
]
# every fuzzy pair has identical embeddings, i.e. cosine 1.0 >= the 0.93 threshold
FUZZY_PAIRS = [("a", "b", 0.9), ("z", "m", 0.9), ("e", "f", 0.9), ("g", "h", 0.9)]


def _wire(module, monkeypatch, *, reviewed=(), jev_fn=None):
    def responder(sql, params):
        if "FROM entities" in sql and "ORDER BY canonical_name" in sql:
            return ENTITY_ROWS
        if "FROM entity_merge_review" in sql:
            return list(reviewed)
        return []

    conn = FakeConn(responder)
    session = FakeSession(lambda q, p: [])
    monkeypatch.setattr(module, "get_connection", lambda: ctx(conn))
    monkeypatch.setattr(module, "get_graph_driver", lambda: ctx(FakeDriver(session)))
    monkeypatch.setattr(module, "find_candidate_pairs", lambda *a, **k: list(FUZZY_PAIRS))
    vec = np.ones(3, dtype=np.float32) / np.sqrt(3)
    monkeypatch.setattr(
        module, "fetch_embeddings", lambda conn_, ids: {i: vec.copy() for i in ids}
    )
    clusters = []

    def fake_merge_cluster(conn_, session_, members, dry_run):
        clusters.append((sorted(m["id"] for m in members), dry_run))
        return len(members) - 1

    monkeypatch.setattr(module, "merge_cluster", fake_merge_cluster)
    monkeypatch.setattr(module, "vacuum_analyze_entities", MagicMock())
    if jev_fn is not None:
        monkeypatch.setattr(jev, "system_one_many", jev_fn)
    return conn, clusters


def _band_jev(items, **kwargs):
    """Acme -> merge (1.9), Zeta -> review (1.0), Eon -> drop (0.1), Gamma -> Jev error."""
    out = []
    for state, _ in items:
        name = state["entity_a"]["name"]
        if name.startswith("Gamma"):
            out.append(JevError("boom", status_code=500))
        else:
            out.append(_resp({"Acme": 1.9, "Zeta": 1.0, "Eon": 0.1}[name.split()[0]]))
    return out


def test_run_entity_merge_with_jev_clusters_only_confirmed_pairs_and_queues_the_middle_band(semantic, monkeypatch):
    conn, clusters = _wire(semantic, monkeypatch, jev_fn=_band_jev)

    total, merged_ids = semantic.run_entity_merge(dry_run=False, use_jev=True)

    assert clusters == [(["a", "b"], False)]  # review / drop / error pairs are never merged
    assert total == 1 and merged_ids == {"a", "b"}
    (sql, rows), = conn.executemany_calls
    assert "INSERT INTO entity_merge_review" in sql and "ON CONFLICT" in sql
    ((entity_a, entity_b, score, reason),) = rows
    assert (entity_a, entity_b) == ("m", "z")  # normalized order of ("z", "m")
    assert score == 1.0 and "Zeta" in reason
    assert conn.commits >= 1


def test_run_entity_merge_with_jev_dry_run_reports_but_writes_nothing(semantic, monkeypatch, capsys):
    conn, clusters = _wire(semantic, monkeypatch, jev_fn=_band_jev)

    semantic.run_entity_merge(dry_run=True, use_jev=True)

    assert clusters == [(["a", "b"], True)]  # merge_cluster in dry-run mode: no mutation
    assert conn.write_statements() == [] and conn.executemany_calls == []
    out = capsys.readouterr().out
    assert "1 confirmed merge, 1 for review, 1 dropped, 1 unconfirmed (Jev error)" in out
    assert "Would add 1 pair(s) to entity_merge_review" in out


def test_run_entity_merge_with_jev_skips_pairs_already_in_review_entirely(semantic, monkeypatch):
    seen = []

    def jev_fn(items, **kwargs):
        seen.extend(state["entity_a"]["name"] for state, _ in items)
        return [_resp(1.9) for _ in items]

    # (a, b) was reviewed (and, say, rejected) earlier: stored in either order
    conn, clusters = _wire(semantic, monkeypatch, reviewed=[("b", "a")], jev_fn=jev_fn)

    semantic.run_entity_merge(dry_run=False, use_jev=True)

    assert "Acme Corp" not in seen  # not re-judged ...
    assert all("a" not in c[0] and "b" not in c[0] for c in clusters)  # ... and never auto-merged
    assert sorted(c[0] for c in clusters) == [["e", "f"], ["g", "h"], ["m", "z"]]


def test_run_entity_merge_jev_errors_everywhere_merge_nothing(semantic, monkeypatch):
    conn, clusters = _wire(
        semantic, monkeypatch, jev_fn=lambda items, **kw: [JevError("down", 503) for _ in items]
    )

    total, ids = semantic.run_entity_merge(dry_run=False, use_jev=True)

    assert (total, ids, clusters) == (0, set(), [])
    assert conn.executemany_calls == []


def test_run_entity_merge_with_jev_and_no_api_key_merges_nothing_and_calls_nothing(semantic, monkeypatch):
    monkeypatch.setattr(jd.settings, "OPENROUTER_API_KEY", "")
    conn, clusters = _wire(
        semantic, monkeypatch, jev_fn=MagicMock(side_effect=AssertionError("no call expected"))
    )

    assert semantic.run_entity_merge(dry_run=False, use_jev=True) == (0, set())
    assert clusters == []


def test_run_entity_merge_without_jev_clusters_every_candidate_and_never_calls_jev(semantic, monkeypatch):
    conn, clusters = _wire(
        semantic, monkeypatch, jev_fn=MagicMock(side_effect=AssertionError("Jev must stay off"))
    )

    total, ids = semantic.run_entity_merge(dry_run=False)

    assert sorted(c[0] for c in clusters) == [["a", "b"], ["e", "f"], ["g", "h"], ["m", "z"]]
    assert total == 4 and ids == set("abzmefgh")
    assert conn.sql_matching("entity_merge_review") == []  # review table untouched
