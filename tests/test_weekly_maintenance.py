"""Tests for scripts/weekly_maintenance.py (U7: weekly corpus maintenance)."""
import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest


def _load_script(name: str, filename: str):
    script_path = Path(__file__).parent.parent / "scripts" / filename
    assert script_path.exists()
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def wm(monkeypatch):
    from audit_fakes import forbid_live_stores

    module = _load_script("weekly_maintenance_script", "weekly_maintenance.py")
    # Any phase a test forgot to stub must fail, not reach the live stack.
    forbid_live_stores(monkeypatch, module, module.entity_merge)
    return module


@pytest.fixture(autouse=True)
def _jev_maintenance_off_by_default(monkeypatch):
    """Never let a developer's real `.env` (MAINTENANCE_JEV_ENABLED, a live
    OPENROUTER_API_KEY) turn the Jev entity audit on inside these tests."""
    from rag.config import settings

    monkeypatch.setattr(settings, "MAINTENANCE_JEV_ENABLED", False)
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", "")


def _conn_ctx(conn):
    """Wrap a MagicMock conn as the object returned by `with get_connection() as conn`."""
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=conn)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _driver_ctx(driver):
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=driver)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


# ── Phase 2 helpers: survivor selection, repointing, self-edges ─────────────


def test_pick_insight_survivor_is_oldest(wm):
    now = datetime(2026, 1, 1)
    members = [
        {"id": "new", "created_at": now},
        {"id": "old", "created_at": now - timedelta(days=10)},
        {"id": "mid", "created_at": now - timedelta(days=5)},
    ]
    survivor = wm.pick_insight_survivor(members)
    assert survivor["id"] == "old"


def test_merge_insights_postgres_repoints_then_deletes(wm):
    conn = MagicMock()
    wm.merge_insights_postgres(conn, "survivor-id", ["dup-1", "dup-2"])

    assert conn.execute.call_count == 2
    update_sql, update_params = conn.execute.call_args_list[0][0]
    assert "UPDATE chunk_insights" in update_sql
    assert update_params == ("survivor-id", ["dup-1", "dup-2"], "survivor-id")

    delete_sql, delete_params = conn.execute.call_args_list[1][0]
    assert "DELETE FROM insights" in delete_sql
    assert delete_params == (["dup-1", "dup-2"],)


def test_merge_insights_postgres_noop_for_no_dups(wm):
    conn = MagicMock()
    wm.merge_insights_postgres(conn, "survivor-id", [])
    conn.execute.assert_not_called()


def test_merge_insights_memgraph_repoints_edges_and_deletes_node(wm):
    session = MagicMock()
    single_result = MagicMock()
    single_result.single.return_value = {"repointed": 2}
    session.run.return_value = single_result

    edges = wm.merge_insights_memgraph(session, "survivor-id", ["dup-1"])

    assert edges == 2
    # 4 calls per dup_id: CONTAINS repoint, RELATED_TO out, RELATED_TO in, DETACH DELETE.
    assert session.run.call_count == 4
    last_call_query = session.run.call_args_list[-1][0][0]
    assert "DETACH DELETE" in last_call_query


def test_merge_insights_memgraph_excludes_survivor_as_self_edge(wm):
    """RELATED_TO re-point queries must exclude `other.insight_id == survivor_id`
    so merging never creates (survivor)-[:RELATED_TO]->(survivor)."""
    session = MagicMock()
    session.run.return_value.single.return_value = None

    wm.merge_insights_memgraph(session, "survivor-id", ["dup-1"])

    related_to_calls = [
        c for c in session.run.call_args_list if "RELATED_TO" in c[0][0]
    ]
    assert len(related_to_calls) == 2
    for c in related_to_calls:
        assert "WHERE other.insight_id <> $survivor_id" in c[0][0]


# ── AE5: seeded duplicate pair merges to oldest ─────────────────────────────


def test_run_insight_merge_ae5_merges_to_oldest(wm, monkeypatch):
    now = datetime(2026, 1, 1)
    monkeypatch.setattr(
        wm, "fetch_probe_insights", MagicMock(return_value=[("new-id", "[1,0]")])
    )
    monkeypatch.setattr(
        wm,
        "find_insight_candidate_pairs",
        MagicMock(return_value=[("new-id", "old-id", 0.99)]),
    )
    monkeypatch.setattr(
        wm,
        "fetch_insight_meta",
        MagicMock(
            return_value={
                "new-id": {"id": "new-id", "created_at": now},
                "old-id": {"id": "old-id", "created_at": now - timedelta(days=3)},
            }
        ),
    )
    conn = MagicMock()
    driver = MagicMock()
    session = MagicMock()
    session.run.return_value.single.return_value = None
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False

    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(conn)))
    monkeypatch.setattr(wm, "get_graph_driver", MagicMock(side_effect=lambda: _driver_ctx(driver)))
    vacuum_mock = MagicMock()
    monkeypatch.setattr(wm, "vacuum_analyze_insights", vacuum_mock)

    merged = wm.run_insight_merge(dry_run=False, since_days=7, cosine_threshold=0.95)

    assert merged == 1
    # survivor is the older row ("old-id"); the newer row ("new-id") is the dup.
    update_call = next(c for c in conn.execute.call_args_list if "UPDATE chunk_insights" in c[0][0])
    assert update_call[0][1] == ("old-id", ["new-id"], "old-id")
    delete_call = next(c for c in conn.execute.call_args_list if "DELETE FROM insights" in c[0][0])
    assert delete_call[0][1] == (["new-id"],)
    # Memgraph DETACH DELETE issued for the loser.
    detach_calls = [c for c in session.run.call_args_list if "DETACH DELETE" in c[0][0]]
    assert len(detach_calls) == 1
    assert detach_calls[0][1]["dup_id"] == "new-id"
    vacuum_mock.assert_called_once()


def test_run_insight_merge_chunk_linked_to_both_survives_once(wm, monkeypatch):
    """The composite-PK-conflict case: a chunk links to both the survivor and
    a loser. merge_insights_postgres must not crash and must leave one row."""
    now = datetime(2026, 1, 1)
    monkeypatch.setattr(wm, "fetch_probe_insights", MagicMock(return_value=[("dup", "[1,0]")]))
    monkeypatch.setattr(
        wm, "find_insight_candidate_pairs", MagicMock(return_value=[("dup", "survivor", 0.99)])
    )
    monkeypatch.setattr(
        wm,
        "fetch_insight_meta",
        MagicMock(
            return_value={
                "dup": {"id": "dup", "created_at": now},
                "survivor": {"id": "survivor", "created_at": now - timedelta(days=1)},
            }
        ),
    )
    conn = MagicMock()
    driver = MagicMock()
    session = MagicMock()
    session.run.return_value.single.return_value = None
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(conn)))
    monkeypatch.setattr(wm, "get_graph_driver", MagicMock(side_effect=lambda: _driver_ctx(driver)))
    monkeypatch.setattr(wm, "vacuum_analyze_insights", MagicMock())

    merged = wm.run_insight_merge(dry_run=False, since_days=7, cosine_threshold=0.95)
    assert merged == 1
    # UPDATE excludes chunks already linked to survivor; DELETE (ON DELETE CASCADE)
    # removes any remaining conflicting chunk_insights row for the loser.
    update_call = next(c for c in conn.execute.call_args_list if "UPDATE chunk_insights" in c[0][0])
    assert "NOT IN" in update_call[0][0]


def test_run_insight_merge_dry_run_writes_nothing(wm, monkeypatch):
    now = datetime(2026, 1, 1)
    monkeypatch.setattr(wm, "fetch_probe_insights", MagicMock(return_value=[("a", "[1,0]")]))
    monkeypatch.setattr(
        wm, "find_insight_candidate_pairs", MagicMock(return_value=[("a", "b", 0.99)])
    )
    monkeypatch.setattr(
        wm,
        "fetch_insight_meta",
        MagicMock(
            return_value={
                "a": {"id": "a", "created_at": now},
                "b": {"id": "b", "created_at": now - timedelta(days=1)},
            }
        ),
    )
    select_conn = MagicMock()
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(select_conn)))
    get_graph_driver_mock = MagicMock()
    monkeypatch.setattr(wm, "get_graph_driver", get_graph_driver_mock)
    vacuum_mock = MagicMock()
    monkeypatch.setattr(wm, "vacuum_analyze_insights", vacuum_mock)

    merged = wm.run_insight_merge(dry_run=True, since_days=7, cosine_threshold=0.95)

    assert merged == 1
    get_graph_driver_mock.assert_not_called()  # no Memgraph session opened at all in dry-run
    vacuum_mock.assert_not_called()
    # Only SELECTs happened on the postgres side (fetch_probe_insights/fetch_insight_meta mocked away,
    # so select_conn.execute should never see an UPDATE/DELETE).
    for c in select_conn.execute.call_args_list:
        sql = c[0][0]
        assert "UPDATE" not in sql and "DELETE" not in sql and "INSERT" not in sql


# ── Scoping: --since vs --full ──────────────────────────────────────────────


def test_fetch_probe_insights_since_scopes_by_created_at(wm):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    wm.fetch_probe_insights(conn, since_days=7)
    sql, params = conn.execute.call_args[0]
    assert "created_at >=" in sql
    assert params == (7,)


def test_fetch_probe_insights_full_has_no_since_filter(wm):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    wm.fetch_probe_insights(conn, since_days=None)
    sql, *rest = conn.execute.call_args[0]
    assert "created_at" not in sql
    assert rest == []


def test_run_phases_full_flag_passes_none_since_days(wm, monkeypatch):
    args = wm.build_parser().parse_args(["--dry-run", "--full", "--skip-entities", "--skip-consistency", "--skip-health"])
    run_insight_merge_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "run_insight_merge", run_insight_merge_mock)

    wm._run_phases(args)

    _, kwargs = run_insight_merge_mock.call_args
    assert kwargs["since_days"] is None


def test_run_phases_default_since_is_seven_days(wm, monkeypatch):
    args = wm.build_parser().parse_args(["--dry-run", "--skip-entities", "--skip-consistency", "--skip-health"])
    run_insight_merge_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "run_insight_merge", run_insight_merge_mock)

    wm._run_phases(args)

    _, kwargs = run_insight_merge_mock.call_args
    assert kwargs["since_days"] == 7


# ── Orphan/consistency sweep ─────────────────────────────────────────────────


def test_consistency_sweep_execute_calls_cleanup_orphan_insights(wm, monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value.run.return_value = []
    driver.session.return_value.__exit__.return_value = False
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(conn)))
    monkeypatch.setattr(wm, "get_graph_driver", MagicMock(side_effect=lambda: _driver_ctx(driver)))
    cleanup_mock = MagicMock()
    monkeypatch.setattr(wm, "_cleanup_orphan_insights", cleanup_mock)

    wm.run_consistency_sweep(dry_run=False)

    cleanup_mock.assert_called_once_with(conn, driver)


def test_consistency_sweep_dry_run_skips_cleanup(wm, monkeypatch):
    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = []
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value.run.return_value = []
    driver.session.return_value.__exit__.return_value = False
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(conn)))
    monkeypatch.setattr(wm, "get_graph_driver", MagicMock(side_effect=lambda: _driver_ctx(driver)))
    cleanup_mock = MagicMock()
    monkeypatch.setattr(wm, "_cleanup_orphan_insights", cleanup_mock)

    wm.run_consistency_sweep(dry_run=True)

    cleanup_mock.assert_not_called()


# ── Phase flags ──────────────────────────────────────────────────────────────


def test_skip_flags_prevent_phase_calls(wm, monkeypatch):
    entity_mock = MagicMock()
    insight_mock = MagicMock(return_value=0)
    consistency_mock = MagicMock(return_value={})
    health_mock = MagicMock()
    monkeypatch.setattr(wm.entity_merge, "run_entity_merge", entity_mock)
    monkeypatch.setattr(wm, "run_insight_merge", insight_mock)
    monkeypatch.setattr(wm, "run_consistency_sweep", consistency_mock)
    monkeypatch.setattr(wm, "run_health_phase", health_mock)

    args = wm.build_parser().parse_args(
        ["--dry-run", "--skip-entities", "--skip-insights", "--skip-consistency", "--skip-health"]
    )
    rc = wm._run_phases(args)

    assert rc == 0
    entity_mock.assert_not_called()
    insight_mock.assert_not_called()
    consistency_mock.assert_not_called()
    health_mock.assert_not_called()


# ── Concurrency guard ────────────────────────────────────────────────────────


def test_execute_aborts_when_active_job_present(wm, monkeypatch):
    guard_conn = MagicMock()
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(guard_conn)))
    monkeypatch.setattr(wm, "count_active_jobs", MagicMock(return_value=1))
    try_lock_mock = MagicMock()
    monkeypatch.setattr(wm, "try_acquire_maintenance_lock", try_lock_mock)
    run_phases_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "_run_phases", run_phases_mock)

    rc = wm.main(["--execute"])

    assert rc == 1
    try_lock_mock.assert_not_called()
    run_phases_mock.assert_not_called()


def test_execute_aborts_when_lock_unavailable(wm, monkeypatch):
    guard_conn = MagicMock()
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(guard_conn)))
    monkeypatch.setattr(wm, "count_active_jobs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "_reap_stale_community_runs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "_count_active_community_runs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "try_acquire_maintenance_lock", MagicMock(return_value=False))
    release_mock = MagicMock()
    monkeypatch.setattr(wm, "release_maintenance_lock", release_mock)
    run_phases_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "_run_phases", run_phases_mock)

    rc = wm.main(["--execute"])

    assert rc == 1
    run_phases_mock.assert_not_called()
    release_mock.assert_not_called()


def test_execute_proceeds_and_releases_lock_when_clear(wm, monkeypatch):
    guard_conn = MagicMock()
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(guard_conn)))
    monkeypatch.setattr(wm, "count_active_jobs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "_reap_stale_community_runs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "_count_active_community_runs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "try_acquire_maintenance_lock", MagicMock(return_value=True))
    release_mock = MagicMock()
    monkeypatch.setattr(wm, "release_maintenance_lock", release_mock)
    run_phases_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "_run_phases", run_phases_mock)

    rc = wm.main(["--execute"])

    assert rc == 0
    run_phases_mock.assert_called_once()
    release_mock.assert_called_once_with(guard_conn)


def test_dry_run_bypasses_guard_even_with_active_jobs(wm, monkeypatch):
    count_active_mock = MagicMock(return_value=5)
    monkeypatch.setattr(wm, "count_active_jobs", count_active_mock)
    run_phases_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "_run_phases", run_phases_mock)

    rc = wm.main(["--dry-run"])

    assert rc == 0
    count_active_mock.assert_not_called()
    run_phases_mock.assert_called_once()


# ── Jev entity audit phase ───────────────────────────────────────────────────

_ONLY_AUDIT = [
    "--skip-entities", "--skip-entity-names", "--skip-insights", "--skip-consistency", "--skip-health",
]


def _parse(wm, *flags):
    return wm.build_parser().parse_args(list(flags))


def _stub_audit(wm, monkeypatch, *, stats=None):
    apply_mock = MagicMock()
    score_mock = MagicMock(return_value=stats if stats is not None else MagicMock(aborted=False))
    monkeypatch.setattr(wm, "run_entity_audit_apply", apply_mock)
    monkeypatch.setattr(wm, "run_entity_audit_scoring", score_mock)
    return apply_mock, score_mock


def _stub_guard_clear(wm, monkeypatch, *, active_jobs=0):
    guard_conn = MagicMock()
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(guard_conn)))
    monkeypatch.setattr(wm, "count_active_jobs", MagicMock(return_value=active_jobs))
    monkeypatch.setattr(wm, "_reap_stale_community_runs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "_count_active_community_runs", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "try_acquire_maintenance_lock", MagicMock(return_value=True))
    monkeypatch.setattr(wm, "release_maintenance_lock", MagicMock())
    return guard_conn


def test_jev_audit_is_off_by_default(wm, monkeypatch, capsys):
    apply_mock, score_mock = _stub_audit(wm, monkeypatch)
    monkeypatch.setattr(wm.entity_merge, "run_entity_merge", MagicMock(return_value=(0, set())))
    monkeypatch.setattr(wm, "_invalidate_entity_edge_cache", MagicMock())
    monkeypatch.setattr(wm, "run_insight_merge", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "run_consistency_sweep", MagicMock(return_value={}))
    monkeypatch.setattr(wm, "run_health_phase", MagicMock())
    _stub_guard_clear(wm, monkeypatch)

    assert wm.main(["--dry-run"]) == 0
    assert wm.main(["--execute"]) == 0

    apply_mock.assert_not_called()
    score_mock.assert_not_called()
    assert "Phase 0" not in capsys.readouterr().out


def test_score_flag_without_jev_is_ignored_with_a_note(wm, monkeypatch, capsys):
    apply_mock, score_mock = _stub_audit(wm, monkeypatch)
    monkeypatch.setattr(wm, "_run_phases", MagicMock(return_value=0))

    assert wm.main(["--dry-run", "--score"]) == 0

    score_mock.assert_not_called()
    apply_mock.assert_not_called()
    assert "Jev maintenance is off" in capsys.readouterr().out


def test_jev_flag_enables_audit_phase_with_the_shared_scope(wm, monkeypatch):
    apply_mock, _ = _stub_audit(wm, monkeypatch)

    assert wm._run_phases(_parse(wm, "--dry-run", "--jev", *_ONLY_AUDIT)) == 0
    apply_mock.assert_called_once_with(dry_run=True, since_days=7, rescore=False)

    apply_mock.reset_mock()
    wm._run_phases(_parse(wm, "--dry-run", "--jev", "--since", "3", "--rescore", *_ONLY_AUDIT))
    apply_mock.assert_called_once_with(dry_run=True, since_days=3, rescore=True)

    apply_mock.reset_mock()
    wm._run_phases(_parse(wm, "--execute", "--jev", "--full", *_ONLY_AUDIT))
    apply_mock.assert_called_once_with(dry_run=False, since_days=None, rescore=False)


def test_setting_enables_audit_and_no_jev_overrides_it(wm, monkeypatch):
    apply_mock, _ = _stub_audit(wm, monkeypatch)
    monkeypatch.setattr(wm.settings, "MAINTENANCE_JEV_ENABLED", True)

    wm._run_phases(_parse(wm, "--dry-run", *_ONLY_AUDIT))
    assert apply_mock.call_count == 1

    apply_mock.reset_mock()
    wm._run_phases(_parse(wm, "--dry-run", "--no-jev", *_ONLY_AUDIT))
    apply_mock.assert_not_called()


def test_jev_and_no_jev_are_mutually_exclusive(wm):
    with pytest.raises(SystemExit):
        _parse(wm, "--dry-run", "--jev", "--no-jev")


def test_skip_entity_audit_skips_scoring(wm, monkeypatch):
    _, score_mock = _stub_audit(wm, monkeypatch)
    _stub_guard_clear(wm, monkeypatch)
    monkeypatch.setattr(wm, "_run_phases", MagicMock(return_value=0))

    assert wm.entity_audit_active(_parse(wm, "--execute", "--jev", "--skip-entity-audit")) is False
    assert wm.main(["--execute", "--jev", "--skip-entity-audit"]) == 0

    score_mock.assert_not_called()


def test_skip_entity_audit_skips_apply(wm, monkeypatch):
    apply_mock, _ = _stub_audit(wm, monkeypatch)

    wm._run_phases(_parse(wm, "--dry-run", "--jev", "--skip-entity-audit", *_ONLY_AUDIT))

    apply_mock.assert_not_called()


def test_audit_apply_runs_before_the_entity_merge(wm, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(wm, "run_entity_audit_apply", lambda **kw: order.append("audit"))
    monkeypatch.setattr(
        wm.entity_merge, "run_entity_merge", lambda dry_run, **kw: order.append("merge") or (0, set())
    )
    args = _parse(
        wm, "--dry-run", "--jev", "--skip-entity-names", "--skip-insights", "--skip-consistency", "--skip-health"
    )

    wm._run_phases(args)

    assert order == ["audit", "merge"]


def test_scoring_default_follows_the_mode(wm, monkeypatch):
    assert wm.should_score(_parse(wm, "--execute")) is True
    assert wm.should_score(_parse(wm, "--dry-run")) is False
    assert wm.should_score(_parse(wm, "--dry-run", "--score")) is True
    assert wm.should_score(_parse(wm, "--execute", "--no-score")) is False


def test_dry_run_does_not_score_by_default_and_skips_the_guard(wm, monkeypatch):
    _, score_mock = _stub_audit(wm, monkeypatch)
    count_active_mock = MagicMock(return_value=9)
    monkeypatch.setattr(wm, "count_active_jobs", count_active_mock)
    run_phases_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "_run_phases", run_phases_mock)

    assert wm.main(["--dry-run", "--jev"]) == 0

    score_mock.assert_not_called()
    count_active_mock.assert_not_called()
    run_phases_mock.assert_called_once()


def test_dry_run_with_score_scores_before_reporting_and_skips_the_guard(wm, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(
        wm, "run_entity_audit_scoring",
        lambda args: order.append("score") or MagicMock(aborted=False),
    )
    count_active_mock = MagicMock(return_value=9)
    monkeypatch.setattr(wm, "count_active_jobs", count_active_mock)
    monkeypatch.setattr(wm, "_run_phases", lambda args: order.append("phases") or 0)

    assert wm.main(["--dry-run", "--jev", "--score"]) == 0

    assert order == ["score", "phases"]
    count_active_mock.assert_not_called()


def test_execute_scores_even_with_active_jobs_but_mutations_stay_refused(wm, monkeypatch, capsys):
    order: list[str] = []
    monkeypatch.setattr(
        wm, "run_entity_audit_scoring",
        lambda args: order.append("score") or MagicMock(aborted=False),
    )
    guard_conn = MagicMock()
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(guard_conn)))
    monkeypatch.setattr(
        wm, "count_active_jobs", lambda conn: order.append("guard") or 2
    )
    try_lock_mock = MagicMock()
    monkeypatch.setattr(wm, "try_acquire_maintenance_lock", try_lock_mock)
    run_phases_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "_run_phases", run_phases_mock)
    apply_mock = MagicMock()
    monkeypatch.setattr(wm, "run_entity_audit_apply", apply_mock)

    rc = wm.main(["--execute", "--jev"])

    assert rc == 1
    assert order == ["score", "guard"]  # scoring ran first, outside the gate
    try_lock_mock.assert_not_called()
    run_phases_mock.assert_not_called()  # no corpus mutation (apply, merges, ...) happened
    apply_mock.assert_not_called()
    assert "Refusing --execute" in capsys.readouterr().out


def test_execute_no_score_skips_scoring_but_still_applies(wm, monkeypatch):
    apply_mock, score_mock = _stub_audit(wm, monkeypatch)
    _stub_guard_clear(wm, monkeypatch)
    monkeypatch.setattr(wm.entity_merge, "run_entity_merge", MagicMock(return_value=(0, set())))
    monkeypatch.setattr(wm, "_invalidate_entity_edge_cache", MagicMock())
    monkeypatch.setattr(wm, "run_insight_merge", MagicMock(return_value=0))
    monkeypatch.setattr(wm, "run_consistency_sweep", MagicMock(return_value={}))
    monkeypatch.setattr(wm, "run_health_phase", MagicMock())
    monkeypatch.setattr(wm, "run_entity_name_correction", MagicMock(return_value=set()))

    assert wm.main(["--execute", "--jev", "--no-score"]) == 0

    score_mock.assert_not_called()
    apply_mock.assert_called_once_with(dry_run=False, since_days=7, rescore=False)


def test_execute_with_jev_scores_then_applies_under_the_lock(wm, monkeypatch):
    order: list[str] = []
    _stub_guard_clear(wm, monkeypatch)
    monkeypatch.setattr(
        wm, "try_acquire_maintenance_lock", lambda conn: order.append("lock") or True
    )
    monkeypatch.setattr(wm, "release_maintenance_lock", lambda conn: order.append("unlock"))
    monkeypatch.setattr(
        wm, "run_entity_audit_scoring",
        lambda args: order.append("score") or MagicMock(aborted=False),
    )
    monkeypatch.setattr(wm, "run_entity_audit_apply", lambda **kw: order.append("apply"))
    monkeypatch.setattr(wm, "_invalidate_entity_edge_cache", MagicMock())
    for name, value in (
        ("run_insight_merge", 0), ("run_consistency_sweep", {}), ("run_health_phase", None),
    ):
        monkeypatch.setattr(wm, name, MagicMock(return_value=value))
    monkeypatch.setattr(wm.entity_merge, "run_entity_merge", MagicMock(return_value=(0, set())))
    monkeypatch.setattr(wm, "run_entity_name_correction", MagicMock(return_value=set()))

    assert wm.main(["--execute", "--jev"]) == 0

    assert order == ["score", "lock", "apply", "unlock"]


def test_scoring_without_api_key_stops_before_any_mutation(wm, monkeypatch, capsys):
    monkeypatch.setattr(wm.settings, "OPENROUTER_API_KEY", "")
    get_conn = MagicMock()
    monkeypatch.setattr(wm, "get_connection", get_conn)
    run_phases_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "_run_phases", run_phases_mock)

    assert wm.main(["--execute", "--jev"]) == 1

    get_conn.assert_not_called()
    run_phases_mock.assert_not_called()
    assert "OPENROUTER_API_KEY" in capsys.readouterr().out


def test_aborted_scoring_still_runs_phases_but_exits_non_zero(wm, monkeypatch):
    _stub_audit(wm, monkeypatch, stats=MagicMock(aborted=True))
    run_phases_mock = MagicMock(return_value=0)
    monkeypatch.setattr(wm, "_run_phases", run_phases_mock)

    assert wm.main(["--dry-run", "--jev", "--score"]) == 1
    run_phases_mock.assert_called_once()


def test_run_entity_audit_apply_execute_invalidates_cache_and_vacuums(wm, monkeypatch):
    from rag.entity_audit import ApplyResult

    conn, driver = MagicMock(), MagicMock()
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(conn)))
    monkeypatch.setattr(wm, "get_graph_driver", MagicMock(side_effect=lambda: _driver_ctx(driver)))
    apply_mock = MagicMock(return_value=ApplyResult(deleted_entity_ids={"e1", "e2"}))
    monkeypatch.setattr(wm.entity_audit, "apply_audit", apply_mock)
    invalidate_mock, vacuum_mock = MagicMock(), MagicMock()
    monkeypatch.setattr(wm, "_invalidate_entity_edge_cache", invalidate_mock)
    monkeypatch.setattr(wm, "vacuum_analyze_entities", vacuum_mock)

    wm.run_entity_audit_apply(dry_run=False, since_days=7, rescore=False)

    session = driver.session.return_value.__enter__.return_value
    apply_mock.assert_called_once_with(conn, session, 7)
    invalidate_mock.assert_called_once_with({"e1", "e2"})
    vacuum_mock.assert_called_once()


def test_run_entity_audit_apply_dry_run_only_reports(wm, monkeypatch):
    conn, driver = MagicMock(), MagicMock()
    monkeypatch.setattr(wm, "get_connection", MagicMock(side_effect=lambda: _conn_ctx(conn)))
    monkeypatch.setattr(wm, "get_graph_driver", MagicMock(side_effect=lambda: _driver_ctx(driver)))
    apply_mock, report_mock = MagicMock(), MagicMock()
    monkeypatch.setattr(wm.entity_audit, "apply_audit", apply_mock)
    monkeypatch.setattr(wm.entity_audit, "dry_run_report", report_mock)
    invalidate_mock = MagicMock()
    monkeypatch.setattr(wm, "_invalidate_entity_edge_cache", invalidate_mock)

    wm.run_entity_audit_apply(dry_run=True, since_days=None, rescore=True)

    session = driver.session.return_value.__enter__.return_value
    report_mock.assert_called_once_with(conn, session, None, rescore=True)
    apply_mock.assert_not_called()
    invalidate_mock.assert_not_called()


def test_dry_run_with_score_writes_only_the_audit_table(wm, monkeypatch):
    """`--dry-run --score` runs the real scoring + report code against
    doubles: the only write anywhere is the upsert into entity_mention_audit."""
    from audit_fakes import FakeConn, FakeDriver, FakeSession, ctx
    from rag import jev
    from rag.jev import JevResponse

    def conn_responder(sql, params):
        if "SELECT id::text FROM chunks" in sql:
            return [("c1",)]
        if "SELECT id::text, content FROM chunks" in sql:
            return [("c1", "Acme ships AI to customers.")]
        if "SELECT count(*)" in sql:
            return [(0,)]
        return []

    def graph(query, params):
        if "UNWIND $chunk_ids" in query:
            return [
                {"chunk_id": "c1", "entity_id": "e1", "name": "Acme", "entity_type": "ORGANIZATION"},
                {"chunk_id": "c1", "entity_id": "e2", "name": "AI", "entity_type": "CONCEPT"},
            ]
        return []

    conn, session = FakeConn(conn_responder), FakeSession(graph)
    monkeypatch.setattr(wm, "get_connection", lambda: ctx(conn))
    monkeypatch.setattr(wm, "get_graph_driver", lambda: ctx(FakeDriver(session)))
    monkeypatch.setattr(wm.settings, "OPENROUTER_API_KEY", "or-test")

    def fake_many(items, **kwargs):
        out = []
        for _state, questions in items:
            n = sum(1 for key in questions if key.endswith("_is_entity"))
            answers = {}
            for i in range(n):
                answers[f"e{i}_is_entity"] = {"noul": 0.1 if i else 0.95}
                choice = {"choice": "ORGANIZATION", "probabilities": {"ORGANIZATION": 0.9, "PRODUCT": 0.1}}
                answers[f"e{i}_type__a"] = choice
                answers[f"e{i}_type__b"] = choice
                answers[f"e{i}_salience"] = {"score": 3.0}
            out.append(JevResponse(answers=answers, usage={"cost": 0.00001}))
        return out

    monkeypatch.setattr(jev, "system_one_many", fake_many)
    monkeypatch.setattr(wm.entity_merge, "run_entity_merge", MagicMock())
    guard = MagicMock()
    monkeypatch.setattr(wm, "count_active_jobs", guard)

    rc = wm.main([
        "--dry-run", "--jev", "--score", "--skip-insights", "--skip-consistency", "--skip-health",
        "--skip-entities", "--skip-entity-names",
    ])

    assert rc == 0
    guard.assert_not_called()
    (sql, rows), = conn.executemany_calls
    assert "INSERT INTO entity_mention_audit" in sql
    assert [(r[1], r[2]) for r in rows] == [("e1", 0.95), ("e2", 0.1)]
    # nothing but that upsert writes to Postgres, and Memgraph is read-only
    assert conn.write_statements() == [sql]
    assert session.write_queries() == []


def test_entity_merge_repoints_mentions_matching_endpoints_only(wm):
    """Once salience makes MENTIONS confidences differ, MERGE-ing on the
    confidence value would create a parallel edge instead of merging."""
    from audit_fakes import FakeSession

    session = FakeSession(lambda q, p: [])
    session.run = lambda query, **params: (
        FakeSession.run(session, query, **params) or MagicMock(single=lambda: {"repointed": 1})
    )

    wm.entity_merge.merge_memgraph(session, "survivor", ["dup"])

    query, params = next(c for c in session.calls if "old:MENTIONS" in c[0])
    assert "MERGE (c)-[m:MENTIONS]->(s) ON CREATE SET m.confidence = old.confidence" in query
    assert "{confidence" not in query
    assert params == {"dup_id": "dup", "survivor_id": "survivor"}


def test_invalidate_entity_edge_cache_commits_its_deletes(wm, monkeypatch):
    """get_connection() is not autocommit and closes without committing, so the
    cache DELETEs are silently rolled back unless explicitly committed."""
    from audit_fakes import FakeConn, ctx

    conn = FakeConn(lambda sql, params: [(2,)] if "SELECT count(*)" in sql else 2)
    monkeypatch.setattr(wm, "get_connection", lambda: ctx(conn))

    wm._invalidate_entity_edge_cache({"e1"})
    assert conn.commits == 1
    assert conn.sql_matching("DELETE FROM entity_semantic_edges")

    conn.commits = 0
    wm._invalidate_entity_edge_cache(set())  # orphan-only sweep
    assert conn.commits == 1
