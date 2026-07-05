import json
from unittest.mock import MagicMock, patch

import pytest


def _make_conn(select_row):
    conn = MagicMock()
    conn.execute.return_value.fetchone.side_effect = select_row
    return conn


def _community(idx: int, label_entities=None):
    return {
        "community_id": str(idx),
        "is_cross_source": False,
        "entities": [{"canonical_name": "Acme Corp"}],
        "contributing_sources": [{"source_id": "s1", "source_name": "Report A"}],
        "chunks": [{"content": "Acme Corp announced a new product.", "source_name": "Report A"}],
    }


# --- generate_theme_report ---

def test_generate_theme_report_includes_evidence_chunks():
    from rag import themes

    run_row = ("run-1", "completed", {"communities": [_community(0)]})
    report_insert_row = ("report-1",)

    with patch("rag.themes.get_connection") as mock_conn:
        conn = mock_conn.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.side_effect = [run_row, report_insert_row]

        with patch(
            "rag.themes._call_llm",
            side_effect=[
                json.dumps({"label": "Acme partnership", "confidence": 4}),
                json.dumps({"buckets": [], "narrative": "n", "cleanup_recommendations": []}),
            ],
        ):
            report_id = themes.generate_theme_report("run-1", model="deepseek-v4-pro")

    assert report_id == "report-1"
    update_call = [
        c for c in conn.execute.call_args_list if "UPDATE theme_reports" in c.args[0]
    ][0]
    status, failed_ids_json, report_json, _report_id = update_call.args[1]
    report = json.loads(report_json)
    assert status == "completed"
    assert json.loads(failed_ids_json) == []
    community = report["communities"][0]
    assert community["evidence_chunks"] == [
        {"content": "Acme Corp announced a new product.", "source": "Report A"}
    ]
    assert community["all_entities"] == ["Acme Corp"]


def test_generate_theme_report_single_community_failure_marks_failed_not_crash():
    """Regression test: the sequential (non-threaded) path used to reference an
    undefined `idx` name when a single-community analysis raised, crashing with
    NameError instead of recording the failure."""
    from rag import themes

    run_row = ("run-1", "completed", {"communities": [_community(0)]})
    report_insert_row = ("report-1",)

    with patch("rag.themes.get_connection") as mock_conn:
        conn = mock_conn.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.side_effect = [run_row, report_insert_row]

        with patch("rag.themes._call_llm", side_effect=RuntimeError("upstream error")):
            report_id = themes.generate_theme_report("run-1", model="deepseek-v4-pro")

    assert report_id == "report-1"
    update_call = [
        c for c in conn.execute.call_args_list if "UPDATE theme_reports" in c.args[0]
    ][0]
    status, failed_ids_json, _report_json, _report_id = update_call.args[1]
    assert status == "failed"
    assert json.loads(failed_ids_json) == [0]


def test_generate_theme_report_raises_on_missing_run():
    from rag import themes

    with patch("rag.themes.get_connection") as mock_conn:
        conn = mock_conn.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.return_value = None
        with pytest.raises(ValueError):
            themes.generate_theme_report("missing-run")


def test_generate_theme_report_raises_on_no_communities():
    from rag import themes

    with patch("rag.themes.get_connection") as mock_conn:
        conn = mock_conn.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.return_value = ("run-1", "completed", {"communities": []})
        with pytest.raises(ValueError):
            themes.generate_theme_report("run-1")
