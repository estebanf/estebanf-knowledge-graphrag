"""Weekly maintenance, Jev phases: flag wiring (everything Jev stays off by
default), the entity-name phase gating and cache invalidation, and the
Jev-confirmed near-duplicate insight band. Offline: Jev, Postgres and Memgraph
are doubles; nothing here runs the script against a live stack."""
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from audit_fakes import FakeConn, FakeDriver, FakeSession, ctx
from rag import jev
from rag import jev_dedup as jd
from rag.jev import JevError, JevResponse

NOW = datetime(2026, 1, 1)


def _load_script(name: str, filename: str):
    path = Path(__file__).parent.parent / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wm(monkeypatch):
    from audit_fakes import forbid_live_stores

    module = _load_script("weekly_maintenance_jev_script", "weekly_maintenance.py")
    # Any phase a test forgot to stub must fail, not reach the live stack.
    forbid_live_stores(monkeypatch, module, module.entity_merge)
    return module


@pytest.fixture(autouse=True)
def _jev_off_and_offline(monkeypatch):
    from rag.config import settings

    def boom(*args, **kwargs):
        raise AssertionError("unexpected network call")

    monkeypatch.setattr(jev, "_new_client", boom)
    monkeypatch.setattr(settings, "MAINTENANCE_JEV_ENABLED", False)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "or-test")


def _parse(wm, *flags):
    return wm.build_parser().parse_args(list(flags))


_NO_OTHER_PHASES = ["--skip-consistency", "--skip-health", "--skip-entity-audit"]


def _stub_phases(wm, monkeypatch):
    entity = MagicMock(return_value=(0, set()))
    insight = MagicMock(return_value=0)
    names = MagicMock(return_value=set())
    monkeypatch.setattr(wm.entity_merge, "run_entity_merge", entity)
    monkeypatch.setattr(wm, "run_insight_merge", insight)
    monkeypatch.setattr(wm, "run_entity_name_correction", names)
    monkeypatch.setattr(wm, "_invalidate_entity_edge_cache", MagicMock())
    return entity, insight, names


# ── Flag wiring: Jev stays off by default ────────────────────────────────────


def test_every_jev_phase_is_off_by_default(wm, monkeypatch, capsys):
    entity, insight, names = _stub_phases(wm, monkeypatch)

    assert wm._run_phases(_parse(wm, "--dry-run", *_NO_OTHER_PHASES)) == 0

    entity.assert_called_once_with(dry_run=True, use_jev=False)
    assert insight.call_args.kwargs["use_jev"] is False
    names.assert_not_called()
    assert "Phase 1b" not in capsys.readouterr().out


def test_jev_flag_turns_on_merge_confirmation_names_and_near_dups(wm, monkeypatch):
    entity, insight, names = _stub_phases(wm, monkeypatch)

    wm._run_phases(_parse(wm, "--dry-run", "--jev", "--name-limit", "7", "--near-dup-limit", "9", *_NO_OTHER_PHASES))

    entity.assert_called_once_with(dry_run=True, use_jev=True)
    assert insight.call_args.kwargs["use_jev"] is True
    assert insight.call_args.kwargs["near_dup_limit"] == 9
    names.assert_called_once_with(dry_run=True, name_limit=7)


def test_defaults_for_the_new_flags(wm):
    args = _parse(wm, "--dry-run")
    assert args.name_limit == 500 and args.near_dup_limit == 2000
    assert args.skip_entity_names is False


def test_setting_enables_the_jev_phases_and_no_jev_overrides_it(wm, monkeypatch):
    entity, insight, names = _stub_phases(wm, monkeypatch)
    monkeypatch.setattr(wm.settings, "MAINTENANCE_JEV_ENABLED", True)

    wm._run_phases(_parse(wm, "--dry-run", *_NO_OTHER_PHASES))
    assert entity.call_args.kwargs["use_jev"] is True and names.call_count == 1

    entity.reset_mock()
    names.reset_mock()
    wm._run_phases(_parse(wm, "--dry-run", "--no-jev", *_NO_OTHER_PHASES))
    assert entity.call_args.kwargs["use_jev"] is False
    names.assert_not_called()


def test_skip_entity_names_skips_only_the_name_phase(wm, monkeypatch):
    entity, insight, names = _stub_phases(wm, monkeypatch)

    wm._run_phases(_parse(wm, "--dry-run", "--jev", "--skip-entity-names", *_NO_OTHER_PHASES))

    names.assert_not_called()
    assert entity.call_args.kwargs["use_jev"] is True  # merge confirmation unaffected
    assert insight.call_args.kwargs["use_jev"] is True


def test_name_phase_runs_after_the_entity_merge_and_before_insights(wm, monkeypatch):
    order = []
    monkeypatch.setattr(
        wm.entity_merge, "run_entity_merge", lambda dry_run, **kw: order.append("merge") or (0, set())
    )
    monkeypatch.setattr(wm, "run_entity_name_correction", lambda **kw: order.append("names") or set())
    monkeypatch.setattr(wm, "run_insight_merge", lambda **kw: order.append("insights") or 0)
    monkeypatch.setattr(wm, "_invalidate_entity_edge_cache", MagicMock())

    wm._run_phases(_parse(wm, "--dry-run", "--jev", *_NO_OTHER_PHASES))

    assert order == ["merge", "names", "insights"]


def test_name_phase_is_refused_while_jobs_are_active_under_execute(wm, monkeypatch, capsys):
    guard_conn = MagicMock()
    monkeypatch.setattr(wm, "get_connection", lambda: ctx(guard_conn))
    monkeypatch.setattr(wm, "count_active_jobs", MagicMock(return_value=2))
    monkeypatch.setattr(wm, "run_entity_audit_scoring", MagicMock(return_value=MagicMock(aborted=False)))
    entity, insight, names = _stub_phases(wm, monkeypatch)

    assert wm.main(["--execute", "--jev"]) == 1

    names.assert_not_called()
    entity.assert_not_called()
    assert "Refusing --execute" in capsys.readouterr().out


def test_name_phase_runs_under_the_lock_when_the_guard_is_clear(wm, monkeypatch):
    order = []
    guard_conn = MagicMock()
    monkeypatch.setattr(wm, "get_connection", lambda: ctx(guard_conn))
    monkeypatch.setattr(wm, "count_active_jobs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "_reap_stale_community_runs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "_count_active_community_runs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "try_acquire_maintenance_lock", lambda c: order.append("lock") or True)
    monkeypatch.setattr(wm, "release_maintenance_lock", lambda c: order.append("unlock"))
    monkeypatch.setattr(wm, "run_entity_audit_scoring", MagicMock(return_value=MagicMock(aborted=False)))
    monkeypatch.setattr(wm, "run_entity_audit_apply", MagicMock())
    monkeypatch.setattr(wm, "run_consistency_sweep", MagicMock(return_value={}))
    monkeypatch.setattr(wm, "run_health_phase", MagicMock())
    entity, insight, names = _stub_phases(wm, monkeypatch)
    names.side_effect = lambda **kw: order.append("names") or set()

    assert wm.main(["--execute", "--jev"]) == 0

    assert order == ["lock", "names", "unlock"]
    names.assert_called_once_with(dry_run=False, name_limit=500)


# ── Phase 1b plumbing ────────────────────────────────────────────────────────


def _name_phase_env(wm, monkeypatch, result):
    conn, driver = MagicMock(), MagicMock()
    monkeypatch.setattr(wm, "get_connection", lambda: ctx(conn))
    monkeypatch.setattr(wm, "get_graph_driver", lambda: ctx(driver))
    run = MagicMock(return_value=result)
    monkeypatch.setattr(wm.entity_names, "run_name_correction", run)
    invalidate, vacuum = MagicMock(), MagicMock()
    monkeypatch.setattr(wm, "_invalidate_entity_edge_cache", invalidate)
    monkeypatch.setattr(wm, "vacuum_analyze_entities", vacuum)
    return run, invalidate, vacuum, conn, driver


def test_name_phase_execute_invalidates_cache_for_every_touched_id_and_vacuums(wm, monkeypatch):
    from rag.entity_names import ApplyResult

    result = ApplyResult(renamed=1, merged=1, touched_ids={"s", "d"}, deleted_ids={"d"})
    run, invalidate, vacuum, conn, driver = _name_phase_env(wm, monkeypatch, result)

    touched = wm.run_entity_name_correction(dry_run=False, name_limit=42)

    assert touched == {"s", "d"}
    invalidate.assert_called_once_with({"s", "d"})
    vacuum.assert_called_once()
    kwargs = run.call_args.kwargs
    assert kwargs["dry_run"] is False and kwargs["limit"] == 42
    assert kwargs["pick_survivor"] is wm.entity_merge.pick_survivor
    assert kwargs["merge_fn"] is wm._merge_entities_into
    assert run.call_args.args[0] is conn


def test_name_phase_rename_only_invalidates_but_does_not_vacuum(wm, monkeypatch):
    from rag.entity_names import ApplyResult

    run, invalidate, vacuum, *_ = _name_phase_env(
        wm, monkeypatch, ApplyResult(renamed=1, touched_ids={"s"})
    )

    wm.run_entity_name_correction(dry_run=False, name_limit=5)

    invalidate.assert_called_once_with({"s"})
    vacuum.assert_not_called()


def test_name_phase_dry_run_and_empty_result_touch_no_cache(wm, monkeypatch):
    from rag.entity_names import ApplyResult

    run, invalidate, vacuum, *_ = _name_phase_env(wm, monkeypatch, ApplyResult())

    assert wm.run_entity_name_correction(dry_run=True, name_limit=5) == set()
    assert wm.run_entity_name_correction(dry_run=False, name_limit=5) == set()

    invalidate.assert_not_called()
    vacuum.assert_not_called()
    assert run.call_args_list[0].kwargs["dry_run"] is True


def test_merge_entities_into_repoints_memgraph_before_postgres(wm, monkeypatch):
    order = []
    monkeypatch.setattr(
        wm.entity_merge, "merge_memgraph", lambda s, sid, dups: order.append("memgraph") or 11
    )
    monkeypatch.setattr(
        wm.entity_merge, "merge_postgres", lambda c, sid, dups, members: order.append("postgres")
    )

    assert wm._merge_entities_into("session", "conn", "survivor", ["dup"], [{"id": "survivor"}]) == 11
    assert order == ["memgraph", "postgres"]


# ── Near-duplicate insights ──────────────────────────────────────────────────


def test_band_query_runs_one_prefilter_pass_and_filters_client_side(wm, monkeypatch):
    """The band reuses the binary-HNSW prefilter + full rerank shape; the lower
    bound is only a client-side cut on the reranked cosines."""
    cur = MagicMock()
    cur.fetchall.return_value = [("c1", 0.97), ("c2", 0.91), ("c3", 0.80)]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    ef = MagicMock()
    monkeypatch.setattr(wm, "set_hnsw_ef_search", ef)

    pairs = wm.find_insight_candidate_pairs(conn, [("p", "[1,0]")], 0.88, 100)

    assert pairs == [("p", "c1", 0.97), ("p", "c2", 0.91)]
    ef.assert_called_once_with(conn, 100)
    sql, params = cur.execute.call_args[0]
    assert "binary_quantize(embedding)::bit(4096) <~> binary_quantize(%s::vector)::bit(4096)" in sql
    assert params == ("p", "[1,0]", 100, "[1,0]", "[1,0]", wm._MERGE_RERANK_LIMIT)


def _insight_env(wm, monkeypatch, probe_pairs, *, reviewed=(), texts=None, jev_fn=None):
    """Wire run_insight_merge to doubles. Returns (conn, merge_calls, find_mock, jev_calls)."""
    def responder(sql, params):
        if "FROM insight_pair_review" in sql:
            return list(reviewed)
        if "FROM insights WHERE id = ANY" in sql:
            return [(i, (texts or {}).get(i, i)) for i in params[0]]  # text defaults to the id
        return []

    conn = FakeConn(responder)
    find = MagicMock(return_value=list(probe_pairs))
    monkeypatch.setattr(wm, "fetch_probe_insights", MagicMock(return_value=[("p", "[1,0]")]))
    monkeypatch.setattr(wm, "find_insight_candidate_pairs", find)
    monkeypatch.setattr(
        wm,
        "fetch_insight_meta",
        lambda c, ids: {i: {"id": i, "created_at": NOW - timedelta(days=ord(i[0]) % 50)} for i in ids},
    )
    monkeypatch.setattr(wm, "get_connection", lambda: ctx(conn))
    driver_session = FakeSession()
    monkeypatch.setattr(wm, "get_graph_driver", lambda: ctx(FakeDriver(driver_session)))
    merges = []
    monkeypatch.setattr(
        wm, "merge_insights_memgraph", lambda s, survivor, dups: merges.append(("memgraph", survivor, list(dups))) or 0
    )
    monkeypatch.setattr(
        wm, "merge_insights_postgres", lambda c, survivor, dups: merges.append(("postgres", survivor, list(dups)))
    )
    monkeypatch.setattr(wm, "vacuum_analyze_insights", MagicMock())
    jev_calls = []

    def default_jev(items, **kwargs):
        jev_calls.extend(items)
        return [JevResponse(answers={"alignment": {"type": "score", "score": 2.0}}) for _ in items]

    monkeypatch.setattr(jev, "system_one_many", jev_fn or default_jev)
    return conn, merges, find, jev_calls


def _score_by_first_char(items, **kwargs):
    """a* pairs same claim (1.9), r* related (1.0), d* different (0.1), e* Jev error."""
    out = []
    for state, _ in items:
        key = state["insight_a"][0]
        if key == "e":
            out.append(JevError("boom", status_code=500))
        else:
            out.append(JevResponse(answers={"alignment": {"type": "score", "score": {"a": 1.9, "r": 1.0, "d": 0.1}[key]}}))
    return out


def test_jev_off_runs_exactly_the_old_flow(wm, monkeypatch):
    conn, merges, find, jev_calls = _insight_env(
        wm, monkeypatch, [("p", "x", 0.99)], jev_fn=MagicMock(side_effect=AssertionError("Jev must stay off"))
    )

    merged = wm.run_insight_merge(dry_run=False, since_days=7, cosine_threshold=0.95)

    find.assert_called_once_with(conn, [("p", "[1,0]")], 0.95, wm.settings.INSIGHT_PREFILTER_CANDIDATES)
    assert merged == 1
    assert conn.sql_matching("insight_pair_review") == []


def test_jev_searches_at_the_low_bound_and_routes_the_band(wm, monkeypatch):
    pairs = [
        ("p", "high", 0.97),  # >= 0.95: merges as before, no Jev call
        ("a1", "a2", 0.93),  # band -> same claim -> merge
        ("r1", "r2", 0.91),  # band -> related -> review
        ("d1", "d2", 0.89),  # band -> different -> dropped
        ("e1", "e2", 0.90),  # band -> Jev error -> unconfirmed
    ]
    conn, merges, find, jev_calls = _insight_env(wm, monkeypatch, pairs, jev_fn=_score_by_first_char)
    judged = []
    inner = jev.system_one_many
    monkeypatch.setattr(
        jev, "system_one_many",
        lambda items, **kw: judged.extend(s["insight_a"] for s, _ in items) or inner(items, **kw),
    )

    merged = wm.run_insight_merge(
        dry_run=False, since_days=7, cosine_threshold=0.95, use_jev=True, near_dup_low=0.88
    )

    assert find.call_args.args[2] == 0.88  # one prefilter pass at the lower bound
    assert sorted(judged) == ["a1", "d1", "e1", "r1"]  # the >= 0.95 pair never reaches Jev
    assert merged == 2  # ("p","high") and ("a1","a2"); related / different / error are not merged
    postgres = [m for m in merges if m[0] == "postgres"]
    merged_ids = {i for m in postgres for i in (m[1], *m[2])}
    assert merged_ids == {"p", "high", "a1", "a2"}
    (sql, rows), = conn.executemany_calls
    assert "INSERT INTO insight_pair_review" in sql and "ON CONFLICT (insight_a, insight_b) DO NOTHING" in sql
    assert rows == [("r1", "r2", 0.91, 1.0)]
    assert conn.commits >= 1


def test_jev_dry_run_judges_and_reports_but_writes_nothing(wm, monkeypatch, capsys):
    pairs = [("a1", "a2", 0.93), ("r1", "r2", 0.91)]
    conn, merges, find, jev_calls = _insight_env(wm, monkeypatch, pairs, jev_fn=_score_by_first_char)
    driver = MagicMock()
    monkeypatch.setattr(wm, "get_graph_driver", driver)

    merged = wm.run_insight_merge(dry_run=True, since_days=7, cosine_threshold=0.95, use_jev=True)

    assert merged == 1  # reported as "would merge"
    assert merges == [] and conn.executemany_calls == [] and conn.write_statements() == []
    driver.assert_not_called()  # no Memgraph session at all
    out = capsys.readouterr().out
    assert "Would add 1 pair(s) to insight_pair_review" in out


def test_band_pairs_are_deduplicated_before_judging(wm, monkeypatch):
    """With --since both members are often probes, so (a, b) and (b, a) both show up."""
    pairs = [("a1", "a2", 0.93), ("a2", "a1", 0.93), ("a1", "a2", 0.92)]
    conn, merges, find, jev_calls = _insight_env(wm, monkeypatch, pairs, jev_fn=_score_by_first_char)
    calls = []
    inner = jev.system_one_many

    def counting(items, **kw):
        calls.extend(items)
        return inner(items, **kw)

    monkeypatch.setattr(jev, "system_one_many", counting)

    wm.run_insight_merge(dry_run=False, since_days=7, cosine_threshold=0.95, use_jev=True)

    assert len(calls) == 1


def test_near_dup_limit_caps_judged_pairs_highest_cosine_first(wm, monkeypatch):
    pairs = [(f"a{i}", f"a{i}x", 0.94 - i * 0.01) for i in range(5)]  # 0.94, 0.93, 0.92, 0.91, 0.90
    conn, merges, find, jev_calls = _insight_env(wm, monkeypatch, pairs, jev_fn=_score_by_first_char)
    judged = []
    inner = jev.system_one_many
    monkeypatch.setattr(jev, "system_one_many", lambda items, **kw: judged.extend(s["insight_a"] for s, _ in items) or inner(items, **kw))

    wm.run_insight_merge(dry_run=True, since_days=7, cosine_threshold=0.95, use_jev=True, near_dup_limit=2)

    assert judged == ["a0", "a1"]


def test_pairs_already_in_the_review_table_are_not_rejudged_or_merged(wm, monkeypatch):
    pairs = [("a1", "a2", 0.93), ("a3", "a4", 0.92)]
    conn, merges, find, jev_calls = _insight_env(
        wm, monkeypatch, pairs, reviewed=[("a2", "a1")], jev_fn=_score_by_first_char
    )
    judged = []
    inner = jev.system_one_many
    monkeypatch.setattr(jev, "system_one_many", lambda items, **kw: judged.extend(s["insight_a"] for s, _ in items) or inner(items, **kw))

    wm.run_insight_merge(dry_run=False, since_days=7, cosine_threshold=0.95, use_jev=True)

    assert judged == ["a3"]
    merged_ids = {i for m in merges for i in (m[1], *m[2])}
    assert merged_ids == {"a3", "a4"}


def test_jev_errors_never_merge_band_pairs(wm, monkeypatch):
    pairs = [("e1", "e2", 0.93), ("e3", "e4", 0.92)]
    conn, merges, find, jev_calls = _insight_env(wm, monkeypatch, pairs, jev_fn=_score_by_first_char)

    assert wm.run_insight_merge(dry_run=False, since_days=7, cosine_threshold=0.95, use_jev=True) == 0
    assert merges == [] and conn.executemany_calls == []


def test_missing_api_key_never_merges_band_pairs_and_makes_no_call(wm, monkeypatch):
    monkeypatch.setattr(wm.settings, "OPENROUTER_API_KEY", "")
    pairs = [("a1", "a2", 0.93)]
    conn, merges, *_ = _insight_env(
        wm, monkeypatch, pairs, jev_fn=MagicMock(side_effect=AssertionError("no call expected"))
    )

    assert wm.run_insight_merge(dry_run=False, since_days=7, cosine_threshold=0.95, use_jev=True) == 0
    assert merges == []
