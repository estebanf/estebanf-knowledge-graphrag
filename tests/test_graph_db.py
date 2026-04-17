from unittest.mock import patch, MagicMock


def test_get_graph_driver_yields_and_closes():
    mock_driver = MagicMock()
    with patch("rag.graph_db.GraphDatabase.driver", return_value=mock_driver) as mock_factory:
        from rag.graph_db import get_graph_driver
        with get_graph_driver() as driver:
            assert driver is mock_driver
        mock_driver.close.assert_called_once()
        call_kwargs = mock_factory.call_args[1]
        assert call_kwargs.get("auth") is None
