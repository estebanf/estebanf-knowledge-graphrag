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
