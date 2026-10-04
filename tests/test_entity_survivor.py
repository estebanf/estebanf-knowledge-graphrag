"""Survivor selection for entity merges: the row with the most MENTIONS edges
wins (so the hub keeps its id), in both merge scripts. Offline: Memgraph is a
recording double."""
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audit_fakes import FakeSession
from rag import entity_audit

NOW = datetime(2026, 1, 1)


def _load_script(name: str, filename: str):
    script_path = Path(__file__).parent.parent / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def semantic(monkeypatch):
    from audit_fakes import forbid_live_stores

    module = _load_script("merge_semantic_duplicates_survivor", "merge_semantic_duplicates.py")
    forbid_live_stores(monkeypatch, module)
    return module


@pytest.fixture
def exact(monkeypatch):
    from audit_fakes import forbid_live_stores

    module = _load_script("merge_duplicate_entities_survivor", "merge_duplicate_entities.py")
    forbid_live_stores(monkeypatch, module)
    return module


def _member(eid, *, name="X", etype="ORGANIZATION", mentions=0, emb=True, age_days=0):
    return {
        "id": eid,
        "canonical_name": name,
        "entity_type": etype,
        "aliases": [],
        "has_embedding": emb,
        "created_at": NOW - timedelta(days=age_days),
        "mentions": mentions,
    }


# ── pick_survivor (semantic script) ──────────────────────────────────────────


def test_semantic_survivor_is_the_row_with_most_mentions_not_the_oldest(semantic):
    members = [
        _member("old-small", name="Open AI", mentions=3, age_days=400),
        _member("new-hub", name="OpenAI", mentions=900, age_days=1),
    ]
    assert semantic.pick_survivor(members)["id"] == "new-hub"


def test_semantic_mentions_beat_a_dominant_type_majority(semantic):
    """The old rule filtered to the dominant type first, so a mis-typed hub
    could never win. Mention count now comes first."""
    members = [
        _member("a", etype="PRODUCT", mentions=1),
        _member("b", etype="PRODUCT", mentions=2),
        _member("hub", etype="ORGANIZATION", mentions=500),
    ]
    assert semantic.pick_survivor(members)["id"] == "hub"


def test_semantic_ties_fall_back_to_dominant_type_then_embedding_then_oldest(semantic):
    # equal mentions -> dominant type (PRODUCT x2) wins over the lone ORGANIZATION
    members = [
        _member("org", etype="ORGANIZATION", mentions=5, age_days=900),
        _member("p1", etype="PRODUCT", mentions=5, age_days=10),
        _member("p2", etype="PRODUCT", mentions=5, age_days=20),
    ]
    assert semantic.pick_survivor(members)["id"] == "p2"  # oldest of the dominant type

    # same type: an embedded row beats an older row without an embedding
    members = [
        _member("no-emb", mentions=5, emb=False, age_days=900),
        _member("emb", mentions=5, emb=True, age_days=1),
    ]
    assert semantic.pick_survivor(members)["id"] == "emb"

    # fully tied except age: oldest wins
    members = [_member("young", mentions=5, age_days=1), _member("elder", mentions=5, age_days=9)]
    assert semantic.pick_survivor(members)["id"] == "elder"


def test_semantic_missing_mentions_count_as_zero_and_none_created_at_is_safe(semantic):
    a = _member("a")
    del a["mentions"]
    b = _member("b", mentions=1)
    b["created_at"] = None
    assert semantic.pick_survivor([a, b])["id"] == "b"
    c = _member("c")
    c["created_at"] = None
    d = _member("d")
    d["created_at"] = None
    assert semantic.pick_survivor([c, d])["id"] in {"c", "d"}  # fully tied, no TypeError


# ── pick_survivor (exact-name script) ────────────────────────────────────────


def test_exact_survivor_prefers_mentions_then_embedding_then_oldest(exact):
    members = [
        _member("older-no-mentions", mentions=0, age_days=100),
        _member("hub", mentions=40, age_days=1),
    ]
    assert exact.pick_survivor(members)["id"] == "hub"

    tie = [_member("no-emb", mentions=3, emb=False, age_days=50), _member("emb", mentions=3, age_days=1)]
    assert exact.pick_survivor(tie)["id"] == "emb"
    tie = [_member("young", mentions=3, age_days=1), _member("elder", mentions=3, age_days=50)]
    assert exact.pick_survivor(tie)["id"] == "elder"


def test_exact_cross_type_survivor_uses_mentions_before_dominant_type(exact):
    members = [
        _member("a", etype="CONCEPT", mentions=1),
        _member("b", etype="CONCEPT", mentions=2),
        _member("hub", etype="PRODUCT", mentions=77),
    ]
    assert exact.pick_survivor_cross_type(members)["id"] == "hub"
    tie = [
        _member("lone", etype="PRODUCT", mentions=4, age_days=500),
        _member("c1", etype="CONCEPT", mentions=4, age_days=2),
        _member("c2", etype="CONCEPT", mentions=4, age_days=3),
    ]
    assert exact.pick_survivor_cross_type(tie)["id"] == "c2"


def test_exact_pick_survivor_rejects_empty(exact):
    with pytest.raises(ValueError):
        exact.pick_survivor([])


# ── mention counts come from Memgraph, batched ───────────────────────────────


def test_attach_mention_counts_uses_one_unwind_and_defaults_to_zero():
    session = FakeSession(
        lambda q, p: [{"entity_id": "hub", "mentions": 12}] if "UNWIND $ids" in q else []
    )
    members = [_member("hub"), _member("ghost"), _member("done", mentions=9)]
    for m in members[:2]:
        del m["mentions"]

    entity_audit.attach_mention_counts(session, members)

    assert [m["mentions"] for m in members] == [12, 0, 9]
    (query, params), = session.calls  # a single query; "done" already had a count
    assert "MENTIONS" in query
    assert params["ids"] == ["hub", "ghost"]

    session.calls.clear()
    entity_audit.attach_mention_counts(session, members)
    assert session.calls == []  # all counted: no query at all


def test_merge_cluster_keeps_the_hub_and_absorbs_the_misspelling(semantic, monkeypatch, capsys):
    members = [
        {**_member("typo", name="Open AI", mentions=0), "aliases": ["OpenAI Inc"]},
        {**_member("good", name="OpenAI", mentions=0)},
    ]
    for m in members:
        del m["mentions"]
    session = FakeSession(
        lambda q, p: [
            {"entity_id": "typo", "mentions": 6826},
            {"entity_id": "good", "mentions": 41},
        ]
        if "UNWIND $ids" in q
        else []
    )
    merge_memgraph = MagicMock(return_value=41)
    merge_postgres = MagicMock()
    monkeypatch.setattr(semantic, "merge_memgraph", merge_memgraph)
    monkeypatch.setattr(semantic, "merge_postgres", merge_postgres)
    conn = MagicMock()

    merged = semantic.merge_cluster(conn, session, members, dry_run=False)

    assert merged == 1
    merge_memgraph.assert_called_once_with(session, "typo", ["good"])
    assert merge_postgres.call_args[0][1:3] == ("typo", ["good"])
    conn.commit.assert_called_once()


def test_merge_cluster_dry_run_reads_counts_but_writes_nothing(semantic, monkeypatch, capsys):
    members = [_member("a", name="Acme Corp"), _member("b", name="Acme Corporation")]
    for m in members:
        del m["mentions"]
    session = FakeSession(lambda q, p: [{"entity_id": "b", "mentions": 8}] if "UNWIND $ids" in q else [])
    monkeypatch.setattr(semantic, "merge_memgraph", MagicMock(side_effect=AssertionError("write")))
    monkeypatch.setattr(semantic, "merge_postgres", MagicMock(side_effect=AssertionError("write")))

    assert semantic.merge_cluster(MagicMock(), session, members, dry_run=True) == 0

    out = capsys.readouterr().out
    assert "'Acme Corporation' (8 mention(s))" in out
    assert session.write_queries() == []


def test_merge_group_attaches_counts_before_picking(exact, monkeypatch):
    group = {
        "name": "Acme",
        "entity_type": "ORGANIZATION",
        "members": [
            {"id": "a", "aliases": [], "has_embedding": True, "created_at": NOW - timedelta(days=9)},
            {"id": "b", "aliases": [], "has_embedding": True, "created_at": NOW},
        ],
    }
    session = FakeSession(lambda q, p: [{"entity_id": "b", "mentions": 30}] if "UNWIND $ids" in q else [])
    memgraph = MagicMock(return_value=30)
    monkeypatch.setattr(exact, "merge_memgraph", memgraph)
    monkeypatch.setattr(exact, "merge_postgres", MagicMock())

    exact.merge_group(MagicMock(), session, group, dry_run=False)

    memgraph.assert_called_once_with(session, "b", ["a"])  # the better-connected row survives
