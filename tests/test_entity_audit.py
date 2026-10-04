"""Jev entity-mention audit (`rag.entity_audit`): question construction,
answer aggregation, resumable scoring, the apply phase (drop / retype /
salience) and the no-write dry-run report. Everything runs against recording
doubles; no test touches Postgres, Memgraph or the network."""

import pytest

from audit_fakes import FakeConn, FakeSession
from rag import entity_audit as ea
from rag import jev
from rag.jev import JevError, JevResponse

NIL = "00000000-0000-0000-0000-000000000000"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(jev, "_new_client", boom)
    monkeypatch.setattr("requests.post", boom)
    monkeypatch.setattr(ea.settings, "ENTITY_AUDIT_INVALID_THRESHOLD", 0.2)
    monkeypatch.setattr(ea.settings, "ENTITY_AUDIT_RETYPE_CONFIDENCE", 0.8)
    monkeypatch.setattr(ea.settings, "JEV_MODEL", "typesafe/jev-1.13")


def _choice(winner, p=0.9):
    options = list(ea.ENTITY_TYPES)
    rest = [o for o in options if o != winner]
    probs = {winner: p}
    for option in rest:
        probs[option] = (1 - p) / len(rest)
    return {"type": "choice", "choice": winner, "probabilities": probs, "confidence": p}


def _answers(n, *, is_entity=0.9, etype="ORGANIZATION", score=2.0):
    answers = {}
    for i in range(n):
        answers[f"e{i}_is_entity"] = {"type": "noul", "noul": is_entity}
        answers[f"e{i}_type__a"] = _choice(etype)
        answers[f"e{i}_type__b"] = _choice(etype)
        answers[f"e{i}_salience"] = {"type": "score", "score": score, "confidence": 0.9}
    return answers


def _mention(i, chunk="c1", name=None, etype="CONCEPT"):
    return ea.Mention(chunk_id=chunk, entity_id=f"e{i}", name=name or f"Entity {i}", entity_type=etype)


def _n_entities(questions):
    return sum(1 for key in questions if key.endswith("_is_entity"))


# ── Question construction ────────────────────────────────────────────────────


def test_build_questions_one_noul_choice_pair_and_score_per_entity():
    mentions = [_mention(0, name="LinkedIn"), _mention(1, name='The "Big" Co')]
    questions = ea.build_questions(mentions)

    assert set(questions) == {
        "e0_is_entity", "e0_type__a", "e0_type__b", "e0_salience",
        "e1_is_entity", "e1_type__a", "e1_type__b", "e1_salience",
    }
    noul = questions["e0_is_entity"]
    assert noul["type"] == "noul"
    assert '"LinkedIn"' in noul["instructions"]
    assert noul["criteria"]["true"] and noul["criteria"]["false"]
    # quotes inside a name cannot break out of the quoted label
    assert "'Big'" in questions["e1_is_entity"]["instructions"]

    # type is a Choice over exactly the 7 schema types, asked twice in reversed order
    a, b = questions["e0_type__a"], questions["e0_type__b"]
    assert a["type"] == b["type"] == "choice"
    assert list(a["criteria"]) == list(ea.ENTITY_TYPES)
    assert list(b["criteria"]) == list(reversed(ea.ENTITY_TYPES))
    assert set(ea.ENTITY_TYPES) == {
        "ORGANIZATION", "PERSON", "POLICY", "PRODUCT", "REGULATION", "CONCEPT", "LOCATION",
    }
    assert all(len(d) <= 255 for d in a["criteria"].values())

    # salience is a 4-level ordered Score
    sal = questions["e0_salience"]
    assert sal["type"] == "score"
    assert len(sal["criteria"]) == 4
    assert "passing" in sal["criteria"][0]
    assert "central" in sal["criteria"][3]


def test_split_into_calls_caps_entities_per_call_and_keeps_chunks_apart():
    big = [_mention(i, chunk="big") for i in range(95)]
    small = [_mention(i, chunk="small") for i in range(3)]
    calls = ea.split_into_calls({"big": big, "small": small})

    assert [(chunk, len(group)) for chunk, group in calls] == [
        ("big", 40), ("big", 40), ("big", 15), ("small", 3),
    ]
    assert [m.entity_id for m in calls[2][1]] == [f"e{i}" for i in range(80, 95)]


# ── Answer aggregation ───────────────────────────────────────────────────────


def test_parse_answers_normalises_salience_and_clamps():
    group = [_mention(0), _mention(1), _mention(2), _mention(3)]
    answers = _answers(4)
    for i, raw in enumerate([0.0, 1.0, 2.0, 3.0]):
        answers[f"e{i}_salience"]["score"] = raw
    scores, malformed = ea.parse_answers(group, answers)
    assert malformed == 0
    assert [s.salience for s in scores] == [0.0, 0.3333, 0.6667, 1.0]

    answers = _answers(1)
    answers["e0_salience"]["score"] = 7.5  # defensive clamp
    assert ea.parse_answers([_mention(0)], answers)[0][0].salience == 1.0


def test_parse_answers_averages_reversed_order_type_votes():
    """Jev favours the first option: the `a` order says ORGANIZATION, the
    reversed `b` order says PRODUCT strongly, so the average picks PRODUCT."""
    answers = _answers(1)
    answers["e0_type__a"] = {"choice": "ORGANIZATION", "probabilities": {"ORGANIZATION": 0.6, "PRODUCT": 0.4}}
    answers["e0_type__b"] = {"choice": "PRODUCT", "probabilities": {"ORGANIZATION": 0.2, "PRODUCT": 0.8}}

    (score,), malformed = ea.parse_answers([_mention(0)], answers)

    assert malformed == 0
    assert score.entity_type == "PRODUCT"
    assert score.type_confidence == pytest.approx(0.6)
    assert score.is_entity == pytest.approx(0.9)
    assert (score.chunk_id, score.entity_id) == ("c1", "e0")


def test_parse_answers_tolerates_missing_parts():
    group = [_mention(0), _mention(1), _mention(2)]
    answers = _answers(3)
    del answers["e0_is_entity"]  # unusable: counted malformed, skipped
    del answers["e1_type__a"], answers["e1_type__b"]  # type unusable: kept, type None
    del answers["e2_salience"]  # salience unusable: kept, salience None

    scores, malformed = ea.parse_answers(group, answers)

    assert malformed == 1
    assert [s.entity_id for s in scores] == ["e1", "e2"]
    assert scores[0].entity_type is None and scores[0].type_confidence is None
    assert scores[0].salience == pytest.approx(0.6667)
    assert scores[1].salience is None and scores[1].entity_type == "ORGANIZATION"


# ── Reading mentions and resumability ────────────────────────────────────────


def _mention_records(rows):
    return [
        {"chunk_id": c, "entity_id": e, "name": n, "entity_type": t} for c, e, n, t in rows
    ]


def test_load_chunk_mentions_dedups_per_entity_orders_by_name_and_skips_nameless():
    session = FakeSession(
        lambda q, p: _mention_records(
            [
                ("c1", "e2", "zeta", "CONCEPT"),
                ("c1", "e1", "Alpha", "CONCEPT"),
                ("c1", "e1", "Alpha", "CONCEPT"),
                ("c1", "e3", "  ", "CONCEPT"),
                ("c2", "e1", "Alpha", None),
            ]
        )
    )

    result = ea.load_chunk_mentions(session, ["c1", "c2"])

    assert "UNWIND $chunk_ids" in session.calls[0][0]
    assert "MENTIONS" in session.calls[0][0]
    assert session.calls[0][1] == {"chunk_ids": ["c1", "c2"]}
    assert [m.name for m in result["c1"]] == ["Alpha", "zeta"]
    assert result["c2"][0].entity_type == ""


def test_iter_pending_batches_skips_scored_pairs_unless_rescore():
    session = FakeSession(
        lambda q, p: _mention_records(
            [("c1", "e1", "A", "CONCEPT"), ("c1", "e2", "B", "CONCEPT"), ("c2", "e1", "A", "CONCEPT")]
        )
    )

    def responder(sql, params):
        if "FROM entity_mention_audit" in sql:
            return [("c1", "e1")]
        return []

    conn = FakeConn(responder)

    (batch,) = list(ea.iter_pending_batches(conn, session, ["c1", "c2"]))
    assert batch.mentions_total == 3
    assert batch.mentions_already_scored == 1
    assert {c: [m.entity_id for m in ms] for c, ms in batch.pending.items()} == {
        "c1": ["e2"], "c2": ["e1"],
    }

    conn = FakeConn(responder)
    (batch,) = list(ea.iter_pending_batches(conn, session, ["c1", "c2"], rescore=True))
    assert batch.mentions_already_scored == 0
    assert sum(len(v) for v in batch.pending.values()) == 3
    assert conn.executed == []  # rescore never even looks at the audit table


# ── Scoring ──────────────────────────────────────────────────────────────────


def _scoring_fixtures(*, contents=None, scored=(), mentions=None, chunk_ids=("c1", "c2")):
    contents = contents if contents is not None else {"c1": "text one", "c2": "text two"}
    mentions = mentions if mentions is not None else [
        ("c1", "e1", "Acme", "ORGANIZATION"),
        ("c1", "e2", "AI", "CONCEPT"),
        ("c2", "e1", "Acme", "ORGANIZATION"),
    ]

    def conn_responder(sql, params):
        if "SELECT id::text FROM chunks" in sql:
            return [(c,) for c in chunk_ids]
        if "SELECT id::text, content FROM chunks" in sql:
            wanted = params[0]
            return [(c, contents[c]) for c in wanted if c in contents]
        if "SELECT chunk_id::text, entity_id::text FROM entity_mention_audit" in sql:
            return [pair for pair in scored if pair[0] in params[0]]
        return []

    conn = FakeConn(conn_responder)
    session = FakeSession(
        lambda q, p: _mention_records([m for m in mentions if m[0] in p.get("chunk_ids", [])])
    )
    return conn, session


def _install_jev(monkeypatch, *, fail_chunks=(), cost=0.00002, calls=None):
    """Replace jev.system_one_many with a deterministic fake."""
    calls = calls if calls is not None else []

    def fake(items, *, max_workers=None, with_usage=False, **kwargs):
        assert with_usage is True
        calls.append(list(items))
        out = []
        for state, questions in items:
            if state in fail_chunks:
                out.append(JevError("boom", status_code=500))
            else:
                out.append(
                    JevResponse(
                        answers=_answers(_n_entities(questions)),
                        usage={"cost": cost},
                    )
                )
        return out

    monkeypatch.setattr(jev, "system_one_many", fake)
    return calls


def test_score_mentions_upserts_results_commits_and_reports_stats(monkeypatch):
    calls = _install_jev(monkeypatch)
    conn, session = _scoring_fixtures()
    messages: list[str] = []

    stats = ea.score_mentions(conn, session, 7, progress=messages.append)

    # one Jev call per chunk, state = chunk text, questions for each entity
    assert len(calls) == 1 and len(calls[0]) == 2
    assert [state for state, _ in calls[0]] == ["text one", "text two"]
    assert [_n_entities(q) for _, q in calls[0]] == [2, 1]

    (sql, rows), = conn.executemany_calls
    assert "INSERT INTO entity_mention_audit" in sql
    assert "ON CONFLICT (chunk_id, entity_id) DO UPDATE" in sql
    assert rows == [
        ("c1", "e1", 0.9, "ORGANIZATION", 0.9, 0.6667, "typesafe/jev-1.13"),
        ("c1", "e2", 0.9, "ORGANIZATION", 0.9, 0.6667, "typesafe/jev-1.13"),
        ("c2", "e1", 0.9, "ORGANIZATION", 0.9, 0.6667, "typesafe/jev-1.13"),
    ]
    assert conn.commits >= 2  # after the scope query and after the batch
    assert stats.mentions_scored == 3
    assert stats.calls == 2 and stats.call_errors == 0
    assert stats.cost_usd == pytest.approx(0.00004)
    assert any("mentions scored 3" in m for m in messages)
    # scoring only ever writes the audit table
    assert all("entity_mention_audit" in w for w in conn.write_statements())
    assert session.write_queries() == []


def test_score_mentions_truncates_the_chunk_state(monkeypatch):
    calls = _install_jev(monkeypatch)
    conn, session = _scoring_fixtures(
        contents={"c1": "x" * 9000},
        mentions=[("c1", "e1", "Acme", "ORGANIZATION")],
        chunk_ids=("c1",),
    )

    ea.score_mentions(conn, session, None, progress=lambda m: None)

    assert len(calls[0][0][0]) == ea.STATE_CHARS == 4000


def test_score_mentions_is_resumable_and_rescore_overrides(monkeypatch):
    calls = _install_jev(monkeypatch)
    all_pairs = {("c1", "e1"), ("c1", "e2"), ("c2", "e1")}
    conn, session = _scoring_fixtures(scored=all_pairs)

    stats = ea.score_mentions(conn, session, 7, progress=lambda m: None)

    assert calls == []  # everything already scored: no Jev call at all
    assert conn.executemany_calls == []
    assert stats.mentions_scored == 0 and stats.mentions_already_scored == 3

    # only the unscored pair is sent on a partial run
    conn, session = _scoring_fixtures(scored={("c1", "e1"), ("c2", "e1")})
    ea.score_mentions(conn, session, 7, progress=lambda m: None)
    assert len(calls) == 1 and [_n_entities(q) for _, q in calls[0]] == [1]
    assert [r[:2] for r in conn.executemany_calls[0][1]] == [("c1", "e2")]

    # --rescore re-asks every mention
    calls.clear()
    conn, session = _scoring_fixtures(scored=all_pairs)
    stats = ea.score_mentions(conn, session, 7, rescore=True, progress=lambda m: None)
    assert [_n_entities(q) for _, q in calls[0]] == [2, 1]
    assert stats.mentions_scored == 3


def test_score_mentions_splits_chunks_with_many_entities_into_second_call(monkeypatch):
    calls = _install_jev(monkeypatch)
    mentions = [("c1", f"e{i}", f"Name {i:03d}", "CONCEPT") for i in range(45)]
    conn, session = _scoring_fixtures(contents={"c1": "chunk"}, mentions=mentions, chunk_ids=("c1",))

    stats = ea.score_mentions(conn, session, 7, progress=lambda m: None)

    assert [_n_entities(q) for _, q in calls[0]] == [40, 5]
    assert [state for state, _ in calls[0]] == ["chunk", "chunk"]
    assert stats.calls == 2 and stats.mentions_scored == 45
    assert len(conn.executemany_calls[0][1]) == 45


def test_jev_error_for_one_call_is_counted_not_raised(monkeypatch):
    _install_jev(monkeypatch, fail_chunks={"text one"})
    conn, session = _scoring_fixtures()

    stats = ea.score_mentions(conn, session, 7, progress=lambda m: None)

    assert stats.call_errors == 1
    assert stats.calls == 2
    assert stats.mentions_scored == 1  # chunk c2 still scored
    assert not stats.aborted
    assert stats.first_error == "boom"
    assert [r[:2] for r in conn.executemany_calls[0][1]] == [("c2", "e1")]


def test_scoring_stops_after_consecutive_fully_failed_batches(monkeypatch):
    calls = _install_jev(monkeypatch, fail_chunks={f"text {i}" for i in range(6)})
    ids = tuple(f"c{i}" for i in range(6))
    conn, session = _scoring_fixtures(
        contents={f"c{i}": f"text {i}" for i in range(6)},
        mentions=[(f"c{i}", "e1", "Acme", "ORGANIZATION") for i in range(6)],
        chunk_ids=ids,
    )
    messages: list[str] = []

    stats = ea.score_mentions(conn, session, None, batch_size=1, progress=messages.append)

    assert stats.aborted is True
    assert len(calls) == ea.MAX_CONSECUTIVE_FAILED_BATCHES == 3
    assert stats.call_errors == 3
    assert any("stopping" in m for m in messages)


# ── Apply ────────────────────────────────────────────────────────────────────


def _apply_conn(
    *,
    drop_counts=(),
    retype_rows=(),
    salience_pages=None,
    drop_pairs=(),
    audited=10,
):
    """Postgres double answering the audit-apply queries."""

    def responder(sql, params):
        if "SELECT count(*)" in sql:
            return [(audited,)]
        if "sum(a.type_confidence)" in sql:
            return list(retype_rows)
        if "ORDER BY count(*) DESC LIMIT 15" in sql:
            return [("Foo", "CONCEPT", 2)]
        if "SELECT a.entity_id::text, count(*)" in sql:
            return list(drop_counts)
        if "CASE WHEN a.salience IS NULL" in sql:
            return [(0, 1), (2, 4), (3, 5)]
        if "ORDER BY a.chunk_id, a.entity_id LIMIT" in sql:
            if "a.salience IS NOT NULL" in sql:
                return salience_pages(params) if salience_pages else []
            return list(drop_pairs)
        if sql.startswith("DELETE FROM entity_mention_audit a WHERE NOT EXISTS"):
            return 4
        if sql.startswith("DELETE FROM entity_mention_audit"):
            return len(drop_pairs)
        return []

    return FakeConn(responder)


def test_apply_drops_invalid_mentions_then_deletes_orphan_entities():
    conn = _apply_conn(
        drop_counts=[("e1", 2), ("e2", 1)],
        drop_pairs=[("c1", "e1"), ("c2", "e1"), ("c3", "e2")],
    )

    def graph(query, params):
        if "OPTIONAL MATCH" in query:  # e1 lost every mention; e2 keeps 3; node order irrelevant
            return [{"entity_id": "e2", "mentions": 3}, {"entity_id": "e1", "mentions": 0}]
        return []

    session = FakeSession(graph)
    messages: list[str] = []

    result = ea.apply_audit(conn, session, None, page_size=100, progress=messages.append)

    # a. the three invalid MENTIONS edges are deleted (exact chunk/entity pairs)
    (delete_edges_q, delete_edges_p), = session.queries_matching("DELETE r")
    assert "[r:MENTIONS]" in delete_edges_q
    assert delete_edges_p["pairs"] == [
        {"chunk_id": "c1", "entity_id": "e1"},
        {"chunk_id": "c2", "entity_id": "e1"},
        {"chunk_id": "c3", "entity_id": "e2"},
    ]
    assert result.edges_dropped == 3

    # ...and only the orphan e1 is deleted from BOTH stores (e2 still has mentions)
    (detach_q, detach_p), = session.queries_matching("DETACH DELETE")
    assert detach_p["ids"] == ["e1"]
    (_, pg_params), = conn.sql_matching("DELETE FROM entities WHERE id")
    assert pg_params == (["e1"],)
    (cache_sql, cache_params), = conn.sql_matching("DELETE FROM entity_semantic_edges")
    assert cache_params == (["e1"], ["e1"])
    assert result.deleted_entity_ids == {"e1"}

    # ordering: edges first, then the live in-degree check, then the node delete
    queries = [q for q, _ in session.calls]
    edge_i = next(i for i, q in enumerate(queries) if "DELETE r" in q)
    count_i = next(i for i, q in enumerate(queries) if "OPTIONAL MATCH" in q)
    node_i = next(i for i, q in enumerate(queries) if "DETACH DELETE" in q)
    assert edge_i < count_i < node_i

    # the audit rows of dropped mentions are removed only after the orphans are gone
    audit_delete = conn.sql_matching("DELETE FROM entity_mention_audit WHERE is_entity < %s")
    assert audit_delete and audit_delete[0][1] == (0.2,)
    executed_sql = [sql for sql, _ in conn.executed]
    entities_delete_sql = conn.sql_matching("DELETE FROM entities WHERE id")[0][0]
    assert executed_sql.index(entities_delete_sql) < executed_sql.index(audit_delete[0][0])
    assert result.audit_rows_removed == 3
    assert result.stale_pruned == 4
    assert conn.commits >= 1


def test_apply_scoped_by_since_uses_chunk_age_and_matching_delete():
    conn = _apply_conn(drop_counts=[("e1", 1)], drop_pairs=[("c1", "e1")])
    session = FakeSession(lambda q, p: [{"entity_id": "e1", "mentions": 0}] if "OPTIONAL MATCH" in q else [])

    ea.apply_audit(conn, session, 7, page_size=100, progress=lambda m: None)

    page_sql, page_params = conn.sql_matching("ORDER BY a.chunk_id, a.entity_id LIMIT")[0]
    assert "c.created_at >= now() - make_interval(days => %s)" in page_sql
    assert page_params == (7, 0.2, NIL, NIL, 100)
    scoped_delete = conn.sql_matching("DELETE FROM entity_mention_audit a USING chunks c")
    assert scoped_delete[0][1] == (0.2, 7)


def test_apply_without_invalid_mentions_touches_no_edges_or_entities():
    conn = _apply_conn()
    session = FakeSession()

    result = ea.apply_audit(conn, session, None, progress=lambda m: None)

    assert session.queries_matching("DELETE r") == []
    assert session.queries_matching("DETACH DELETE") == []
    assert conn.sql_matching("DELETE FROM entities WHERE id") == []
    assert result.edges_dropped == 0 and result.deleted_entity_ids == set()


RETYPE_ROWS = [
    # (entity_id, name, current type, voted type, votes, summed type confidence)
    ("x1", "LinkedIn", "EVENT", "ORGANIZATION", 3, 2.64),       # mean .88 -> retype
    ("x2", "Maybe", "ROLE", "PERSON", 2, 1.0),                  # mean .5 -> too unsure
    ("x3", "Split", "DATE", "ORGANIZATION", 1, 0.9),            # majority PRODUCT...
    ("x3", "Split", "DATE", "PRODUCT", 2, 1.2),                 # ...mean .6 -> too unsure
    ("x5", "NoType", None, "LOCATION", 2, 1.8),                 # NULL type counts as off-schema
    ("x6", "Tie", "EVENT", "POLICY", 2, 1.9),                   # vote tie -> higher confidence wins
    ("x6", "Tie", "EVENT", "CONCEPT", 2, 1.7),
]


def test_apply_retypes_only_confident_off_schema_entities():
    conn = _apply_conn(retype_rows=RETYPE_ROWS)
    session = FakeSession()

    result = ea.apply_audit(conn, session, None, progress=lambda m: None)

    # SQL restricts to entities whose current type is NOT one of the 7 schema
    # types and only counts mentions that survive the drop.
    retype_sql, retype_params = conn.sql_matching("sum(a.type_confidence)")[0]
    assert "e.entity_type IS NULL OR e.entity_type <> ALL(%s)" in retype_sql
    assert "a.is_entity >= %s" in retype_sql
    assert retype_params == (0.2, list(ea.ENTITY_TYPES), list(ea.ENTITY_TYPES))

    (_, graph_params), = session.queries_matching("SET e.entity_type")
    assert graph_params["rows"] == [
        {"entity_id": "x1", "entity_type": "ORGANIZATION"},
        {"entity_id": "x5", "entity_type": "LOCATION"},
        {"entity_id": "x6", "entity_type": "POLICY"},
    ]
    (update_sql, update_rows), = [c for c in conn.executemany_calls if "UPDATE entities" in c[0]]
    assert update_rows == [("ORGANIZATION", "x1"), ("LOCATION", "x5"), ("POLICY", "x6")]
    assert result.retyped == 3


def test_apply_writes_salience_into_mentions_confidence_in_keyset_pages():
    def pages(params):
        # params = (invalid_threshold, cursor_chunk, cursor_entity, page_size)
        if params[1] == NIL:
            return [("c1", "e1", 0.0), ("c1", "e2", 1.0)]
        assert (params[1], params[2]) == ("c1", "e2")
        return [("c2", "e1", 0.3333)]

    conn = _apply_conn(salience_pages=pages)
    session = FakeSession()

    result = ea.apply_audit(conn, session, None, page_size=2, progress=lambda m: None)

    writes = session.queries_matching("SET m.confidence = r.salience")
    assert len(writes) == 2
    assert "UNWIND $rows" in writes[0][0]
    assert "[m:MENTIONS]" in writes[0][0]
    assert writes[0][1]["rows"] == [
        {"chunk_id": "c1", "entity_id": "e1", "salience": 0.0},
        {"chunk_id": "c1", "entity_id": "e2", "salience": 1.0},
    ]
    assert writes[1][1]["rows"] == [{"chunk_id": "c2", "entity_id": "e1", "salience": 0.3333}]
    assert result.salience_written == 3
    # only kept (valid) mentions with a salience are written
    sql, _ = conn.sql_matching("a.salience IS NOT NULL")[0]
    assert "a.is_entity >= %s" in sql


def test_prune_stale_audit_rows_removes_rows_with_missing_chunk_or_entity():
    conn = FakeConn(lambda sql, params: 5)
    assert ea.prune_stale_audit_rows(conn) == 5
    sql, _ = conn.executed[0]
    assert "NOT EXISTS (SELECT 1 FROM chunks" in sql and "NOT EXISTS (SELECT 1 FROM entities" in sql


# ── Dry-run report ───────────────────────────────────────────────────────────


def test_dry_run_report_describes_the_plan_and_writes_nothing():
    def conn_responder(sql, params):
        if "SELECT chunk_id::text, entity_id::text FROM entity_mention_audit" in sql:
            return [("c1", "e1")]
        if "SELECT id::text FROM chunks" in sql:
            return [("c1",)]
        if "SELECT count(*)" in sql:
            return [(40,)]
        if "sum(a.type_confidence)" in sql:
            return RETYPE_ROWS[:1]
        if "ORDER BY count(*) DESC LIMIT 15" in sql:
            return [("AI", "CONCEPT", 7)]
        if "SELECT a.entity_id::text, count(*)" in sql:
            return [("e1", 2), ("e2", 1)]
        if "CASE WHEN a.salience IS NULL" in sql:
            return [(0, 10), (2, 20), (3, 5)]
        return []

    def graph(query, params):
        if "UNWIND $chunk_ids" in query:
            return _mention_records(
                [("c1", "e1", "A", "CONCEPT"), ("c1", "e2", "B", "CONCEPT"), ("c1", "e3", "C", "CONCEPT")]
            )
        if "OPTIONAL MATCH" in query:
            # e1 has exactly the 2 mentions being dropped (orphan); e2 keeps others
            return [{"entity_id": "e1", "mentions": 2}, {"entity_id": "e2", "mentions": 3}]
        return []

    conn, session = FakeConn(conn_responder), FakeSession(graph)
    lines: list[str] = []

    plan = ea.dry_run_report(conn, session, 7, progress=lines.append)

    report = "\n".join(lines)
    assert plan.drop_mentions == 3
    assert plan.orphan_entity_ids == ["e1"]
    assert plan.unscored_mentions == 2  # e2, e3 have no audit row yet
    assert "mentions to drop (is_entity < 0.2): 3 across 2 entity/entities" in report
    assert "'AI' [CONCEPT] x7" in report
    assert "entities left with no mentions (would be deleted): 1" in report
    assert "off-schema entities to retype (>= 0.8 mean type confidence): 1" in report
    assert "'LinkedIn' EVENT -> ORGANIZATION (3 mention(s), mean 0.88)" in report
    assert "passing mention (0): 10" in report and "central (3): 5" in report
    assert "in-scope mentions not yet scored: 2" in report

    # no writes anywhere: Postgres only saw SELECTs, Memgraph only reads, no commit
    assert conn.write_statements() == []
    assert all(sql.strip().upper().startswith("SELECT") for sql, _ in conn.executed)
    assert session.write_queries() == []
    assert conn.commits == 0


def test_count_unscored_is_read_only():
    conn = FakeConn(
        lambda sql, params: [("c1",)] if "SELECT id::text FROM chunks" in sql else []
    )
    session = FakeSession(
        lambda q, p: _mention_records([("c1", "e1", "A", "CONCEPT")]) if "UNWIND $chunk_ids" in q else []
    )

    assert ea.count_unscored(conn, session, None) == (1, 1)
    assert conn.write_statements() == [] and session.write_queries() == []
