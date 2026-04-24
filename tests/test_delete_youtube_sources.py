from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _load_script_module():
    script_path = Path(__file__).parent.parent / "scripts" / "delete_youtube_sources.py"
    spec = importlib.util.spec_from_file_location("delete_youtube_sources_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_list_youtube_sources_filters_active_kind_youtube():
    from rag.youtube_cleanup import list_youtube_sources

    conn = MagicMock()
    conn.execute.return_value.fetchall.return_value = [
        ("source-1", "Video 1", "video1.txt"),
    ]

    rows = list_youtube_sources(conn)

    assert rows == [
        {"source_id": "source-1", "name": "Video 1", "file_name": "video1.txt"},
    ]
    sql = conn.execute.call_args.args[0]
    params = conn.execute.call_args.args[1]
    assert "deleted_at IS NULL" in sql
    assert "metadata->>'kind' = 'youtube'" in sql
    assert params == []


def test_purge_youtube_sources_dry_run_does_not_delete(capsys):
    from rag.youtube_cleanup import purge_youtube_sources

    fake_matches = [
        {"source_id": "source-1", "name": "Video 1", "file_name": "video1.txt"},
        {"source_id": "source-2", "name": "Video 2", "file_name": "video2.txt"},
    ]
    conn = MagicMock()
    driver = MagicMock()
    delete_source_artifacts = MagicMock()
    delete_stored_file = MagicMock()

    deleted = purge_youtube_sources(
        conn=conn,
        driver=driver,
        execute=False,
        delete_source_artifacts_fn=delete_source_artifacts,
        delete_stored_file_fn=delete_stored_file,
        matches=fake_matches,
    )

    assert deleted == []
    delete_source_artifacts.assert_not_called()
    delete_stored_file.assert_not_called()
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "source-1" in captured.out
    assert "source-2" in captured.out


def test_purge_youtube_sources_execute_deletes_matches(capsys):
    from rag.youtube_cleanup import purge_youtube_sources

    fake_matches = [
        {"source_id": "source-1", "name": "Video 1", "file_name": "video1.txt"},
        {"source_id": "source-2", "name": "Video 2", "file_name": "video2.txt"},
    ]
    conn = MagicMock()
    driver = MagicMock()
    delete_source_artifacts = MagicMock()
    delete_stored_file = MagicMock()

    deleted = purge_youtube_sources(
        conn=conn,
        driver=driver,
        execute=True,
        delete_source_artifacts_fn=delete_source_artifacts,
        delete_stored_file_fn=delete_stored_file,
        matches=fake_matches,
    )

    assert deleted == ["source-1", "source-2"]
    assert delete_source_artifacts.call_args_list[0].args == (conn, driver, "source-1")
    assert delete_source_artifacts.call_args_list[1].args == (conn, driver, "source-2")
    assert delete_stored_file.call_args_list[0].args == ("source-1",)
    assert delete_stored_file.call_args_list[1].args == ("source-2",)
    assert conn.commit.call_count == 2
    captured = capsys.readouterr()
    assert "Deleted source-1" in captured.out
    assert "Deleted source-2" in captured.out


def test_script_main_requires_execute_for_deletion(monkeypatch):
    module = _load_script_module()
    purge = MagicMock(return_value=[])
    monkeypatch.setattr(module, "run", purge)

    exit_code = module.main([])

    assert exit_code == 0
    assert purge.call_args.kwargs == {
        "execute": False,
        "source_id": None,
        "limit": None,
    }


def test_script_main_passes_execute_and_filters(monkeypatch):
    module = _load_script_module()
    purge = MagicMock(return_value=["source-1"])
    monkeypatch.setattr(module, "run", purge)

    exit_code = module.main(["--execute", "--source-id", "source-1", "--limit", "3"])

    assert exit_code == 0
    assert purge.call_args.kwargs == {
        "execute": True,
        "source_id": "source-1",
        "limit": 3,
    }
