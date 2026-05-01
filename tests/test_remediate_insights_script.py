import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


def _load_script():
    script_path = Path(__file__).parent.parent / "scripts" / "remediate_insights.py"
    assert script_path.exists()
    spec = importlib.util.spec_from_file_location("remediate_insights_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_remediate_insights_no_pending_sources(capsys):
    module = _load_script()
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value.fetchall.return_value = []
    graph_driver = MagicMock()
    module.get_connection = MagicMock()
    module.get_connection.return_value.__enter__.return_value = conn
    module.get_graph_driver = MagicMock()
    module.get_graph_driver.return_value.__enter__.return_value = graph_driver

    exit_code = module.main(["--batch-size", "5"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Found 0 sources pending insight extraction" in captured.out
    assert "Nothing to do" in captured.out


def test_remediate_insights_source_id_skips_existing_without_force(capsys):
    module = _load_script()
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (3,)
    graph_driver = MagicMock()
    module.get_connection = MagicMock()
    module.get_connection.return_value.__enter__.return_value = conn
    module.get_graph_driver = MagicMock()
    module.get_graph_driver.return_value.__enter__.return_value = graph_driver
    module.extract_and_store_insights = MagicMock()

    exit_code = module.main(["--source-id", "source-1"])

    assert exit_code == 0
    module.extract_and_store_insights.assert_not_called()
    captured = capsys.readouterr()
    assert "already has insight links" in captured.out


def test_remediate_insights_source_id_processes_when_no_existing_links(capsys):
    module = _load_script()
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (0,)
    cur.fetchall.return_value = [("chunk-1", "content")]
    graph_driver = MagicMock()
    module.get_connection = MagicMock()
    module.get_connection.return_value.__enter__.return_value = conn
    module.get_graph_driver = MagicMock()
    module.get_graph_driver.return_value.__enter__.return_value = graph_driver
    def fake_extract(conn_arg, driver_arg, source_id, chunk_rows, progress_callback=None):
        assert progress_callback is not None
        progress_callback("extract_start", {"total": 1, "concurrency": 3})
        progress_callback(
            "extract_chunk",
            {"position": 1, "total": 1, "chunk_id": "chunk-1", "insights": 1},
        )
        progress_callback("store_start", {"total": 1})
        progress_callback(
            "store_chunk",
            {"position": 1, "total": 1, "chunk_id": "chunk-1", "insights": 1},
        )
        progress_callback("store_done", {"total": 1})
        return {"chunks_processed": 1, "insights_extracted": 1, "insights_reused": 0}

    module.extract_and_store_insights = MagicMock(side_effect=fake_extract)

    exit_code = module.main(["--source-id", "source-1"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Loading chunks for source source-1" in captured.out
    assert "Found 1 chunks" in captured.out
    assert "Extracting insights with concurrency 3" in captured.out
    assert "Extracted chunk 1/1" in captured.out
    assert "Storing insights serially" in captured.out
    assert "Stored chunk 1/1" in captured.out
    assert "[OK] source-1" in captured.out


def test_remediate_insights_source_id_force_cleans_before_rebuild():
    module = _load_script()
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = (3,)
    cur.fetchall.return_value = [("chunk-1", "content")]
    graph_driver = MagicMock()
    module.get_connection = MagicMock()
    module.get_connection.return_value.__enter__.return_value = conn
    module.get_graph_driver = MagicMock()
    module.get_graph_driver.return_value.__enter__.return_value = graph_driver
    module.extract_and_store_insights = MagicMock(
        return_value={"chunks_processed": 1, "insights_extracted": 1, "insights_reused": 0}
    )

    exit_code = module.main(["--source-id", "source-1", "--force"])

    assert exit_code == 0
    sql_calls = [call.args[0] for call in cur.execute.call_args_list]
    assert any("DELETE FROM chunk_insights" in sql for sql in sql_calls)
    assert any("DELETE FROM insights" in sql for sql in sql_calls)
    assert graph_driver.session.return_value.__enter__.return_value.run.called
    module.extract_and_store_insights.assert_called_once()
