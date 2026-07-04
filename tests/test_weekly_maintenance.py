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
def wm():
    return _load_script("weekly_maintenance_script", "weekly_maintenance.py")


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
