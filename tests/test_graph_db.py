from unittest.mock import patch, MagicMock

import pytest


def test_get_graph_driver_yields_and_closes():
    mock_driver = MagicMock()
    with patch("rag.graph_db.GraphDatabase.driver", return_value=mock_driver) as mock_factory:
        from rag.graph_db import get_graph_driver
        with get_graph_driver() as driver:
            assert driver is mock_driver
        mock_driver.close.assert_called_once()
        call_kwargs = mock_factory.call_args[1]
        assert call_kwargs.get("auth") is None


def test_reconcile_schema_runs_every_statement():
    from rag.graph_db import SCHEMA_STATEMENTS, reconcile_schema

    session = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    reconcile_schema(driver)

    assert session.run.call_count == len(SCHEMA_STATEMENTS)
    executed = [call.args[0] for call in session.run.call_args_list]
    assert executed == SCHEMA_STATEMENTS


def test_reconcile_schema_is_idempotent_across_repeated_calls():
    from rag.graph_db import SCHEMA_STATEMENTS, reconcile_schema

    session = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    reconcile_schema(driver)
    reconcile_schema(driver)

    assert session.run.call_count == 2 * len(SCHEMA_STATEMENTS)


def test_reconcile_schema_surfaces_connection_failure():
    from rag.graph_db import reconcile_schema

    driver = MagicMock()
    driver.session.side_effect = RuntimeError("memgraph unreachable")

    with pytest.raises(RuntimeError, match="memgraph unreachable"):
        reconcile_schema(driver)
