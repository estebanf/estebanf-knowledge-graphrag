"""Tests that merge scripts trigger entities vacuum only after real merges."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

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
def semantic_module():
    return _load_script("merge_semantic_duplicates_script", "merge_semantic_duplicates.py")


@pytest.fixture
def duplicate_module():
    return _load_script("merge_duplicate_entities_script", "merge_duplicate_entities.py")


def _patch_db(module, monkeypatch, conn):
    monkeypatch.setattr(module, "get_connection", MagicMock())
    module.get_connection.return_value.__enter__.return_value = conn
    monkeypatch.setattr(module, "get_graph_driver", MagicMock())
    module.get_graph_driver.return_value.__enter__.return_value = MagicMock()


# ── merge_semantic_duplicates.py ─────────────────────────────────────────────


def test_semantic_execute_with_merge_triggers_vacuum(semantic_module, monkeypatch):
    module = semantic_module
    conn = MagicMock()
    _patch_db(module, monkeypatch, conn)
    monkeypatch.setattr(module, "find_candidate_pairs", MagicMock(return_value=[("a", "b", 1.0)]))
    monkeypatch.setattr(
        module, "fetch_embeddings", MagicMock(return_value={"a": object(), "b": object()})
    )
    monkeypatch.setattr(
        conn,
        "execute",
        MagicMock(
            return_value=MagicMock(
                fetchall=MagicMock(
                    return_value=[
                        ("a", "Acme Corp", "ORG", [], True, None),
                        ("b", "Acme Corporation", "ORG", [], True, None),
                    ]
                )
            )
        ),
    )
    monkeypatch.setattr(module, "merge_cluster", MagicMock(return_value=1))
    vacuum_mock = MagicMock()
    monkeypatch.setattr(module, "vacuum_analyze_entities", vacuum_mock)
    monkeypatch.setattr(sys, "argv", ["merge_semantic_duplicates.py", "--execute"])

    # Force a single 2-member cluster regardless of cosine math.
    monkeypatch.setattr(module.np, "dot", MagicMock(return_value=1.0))

    exit_code = module.main()

    assert exit_code == 0
    vacuum_mock.assert_called_once()


def test_semantic_dry_run_does_not_trigger_vacuum(semantic_module, monkeypatch):
    module = semantic_module
    conn = MagicMock()
    _patch_db(module, monkeypatch, conn)
    monkeypatch.setattr(module, "find_candidate_pairs", MagicMock(return_value=[("a", "b", 1.0)]))
    monkeypatch.setattr(
        module, "fetch_embeddings", MagicMock(return_value={"a": object(), "b": object()})
    )
    monkeypatch.setattr(
        conn,
        "execute",
        MagicMock(
            return_value=MagicMock(
                fetchall=MagicMock(
                    return_value=[
                        ("a", "Acme Corp", "ORG", [], True, None),
                        ("b", "Acme Corporation", "ORG", [], True, None),
                    ]
                )
            )
        ),
    )
    monkeypatch.setattr(module.np, "dot", MagicMock(return_value=1.0))
    vacuum_mock = MagicMock()
    monkeypatch.setattr(module, "vacuum_analyze_entities", vacuum_mock)
    monkeypatch.setattr(sys, "argv", ["merge_semantic_duplicates.py", "--dry-run"])

    exit_code = module.main()

    assert exit_code == 0
    vacuum_mock.assert_not_called()


def test_semantic_zero_clusters_does_not_trigger_vacuum(semantic_module, monkeypatch):
    module = semantic_module
    conn = MagicMock()
    _patch_db(module, monkeypatch, conn)
    monkeypatch.setattr(module, "find_candidate_pairs", MagicMock(return_value=[]))
    monkeypatch.setattr(module, "fetch_embeddings", MagicMock(return_value={}))
    monkeypatch.setattr(
        conn,
        "execute",
        MagicMock(return_value=MagicMock(fetchall=MagicMock(return_value=[]))),
    )
    vacuum_mock = MagicMock()
    monkeypatch.setattr(module, "vacuum_analyze_entities", vacuum_mock)
    monkeypatch.setattr(sys, "argv", ["merge_semantic_duplicates.py", "--execute"])

    exit_code = module.main()

    assert exit_code == 0
    vacuum_mock.assert_not_called()


# ── merge_duplicate_entities.py ──────────────────────────────────────────────


def test_duplicate_entities_execute_with_merge_triggers_vacuum(duplicate_module, monkeypatch):
    module = duplicate_module
    conn = MagicMock()
    _patch_db(module, monkeypatch, conn)
    monkeypatch.setattr(
        module,
        "fetch_duplicate_groups",
        MagicMock(return_value=[{"name": "Acme Corp", "members": [{"id": "a"}, {"id": "b"}]}]),
    )
    monkeypatch.setattr(module, "merge_group", MagicMock(return_value=1))
    vacuum_mock = MagicMock()
    monkeypatch.setattr(module, "vacuum_analyze_entities", vacuum_mock)
    monkeypatch.setattr(sys, "argv", ["merge_duplicate_entities.py", "--execute"])

    exit_code = module.main()

    assert exit_code == 0
    vacuum_mock.assert_called_once()


def test_duplicate_entities_dry_run_does_not_trigger_vacuum(duplicate_module, monkeypatch):
    module = duplicate_module
    conn = MagicMock()
    _patch_db(module, monkeypatch, conn)
    monkeypatch.setattr(
        module,
        "fetch_duplicate_groups",
        MagicMock(
            return_value=[
                {
                    "name": "Acme Corp",
                    "members": [
                        {"id": "a", "has_embedding": True, "created_at": None, "entity_type": "ORG"},
                        {"id": "b", "has_embedding": True, "created_at": None, "entity_type": "ORG"},
                    ],
                }
            ]
        ),
    )
    vacuum_mock = MagicMock()
    monkeypatch.setattr(module, "vacuum_analyze_entities", vacuum_mock)
    monkeypatch.setattr(sys, "argv", ["merge_duplicate_entities.py", "--dry-run"])

    exit_code = module.main()

    assert exit_code == 0
    vacuum_mock.assert_not_called()


def test_duplicate_entities_zero_groups_does_not_trigger_vacuum(duplicate_module, monkeypatch):
    module = duplicate_module
    conn = MagicMock()
    _patch_db(module, monkeypatch, conn)
    monkeypatch.setattr(module, "fetch_duplicate_groups", MagicMock(return_value=[]))
    vacuum_mock = MagicMock()
    monkeypatch.setattr(module, "vacuum_analyze_entities", vacuum_mock)
    monkeypatch.setattr(sys, "argv", ["merge_duplicate_entities.py", "--execute"])

    exit_code = module.main()

    assert exit_code == 0
    vacuum_mock.assert_not_called()
