from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import MagicMock


def _load_script_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "delete_legacy_graph_edges.py"
    spec = spec_from_file_location("delete_legacy_graph_edges", script_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_edge_counts_reads_related_to_and_mentioned_in():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False
    session.run.side_effect = [
        [{"count": 12}],
        [{"count": 7}],
    ]

    module = _load_script_module()

    counts = module.collect_edge_counts(driver)

    assert counts == {"RELATED_TO": 12, "MENTIONED_IN": 7}


def test_delete_legacy_edges_executes_for_both_types():
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    driver.session.return_value.__exit__.return_value = False

    module = _load_script_module()

    module.delete_legacy_edges(driver)

    executed = " ".join(call.args[0] for call in session.run.call_args_list)
    assert "RELATED_TO" in executed
    assert "MENTIONED_IN" in executed
