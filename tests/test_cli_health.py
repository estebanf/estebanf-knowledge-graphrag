from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rag.cli import app


runner = CliRunner()


@patch("rag.cli.get_graph_driver")
@patch("rag.cli.get_connection")
def test_health_command_reports_ready(mock_conn, mock_graph_driver):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    graph_driver = MagicMock()
    mock_graph_driver.return_value.__enter__.return_value = graph_driver
    graph_driver.session.return_value.__enter__.return_value.run.return_value = None

    result = runner.invoke(app, ["health"])

    assert result.exit_code == 0
    assert "ready" in result.output.lower()


@patch("rag.cli.get_graph_driver")
@patch("rag.cli.get_connection")
def test_health_command_reports_failure(mock_conn, mock_graph_driver):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.side_effect = RuntimeError("db down")
    graph_driver = MagicMock()
    mock_graph_driver.return_value.__enter__.return_value = graph_driver

    result = runner.invoke(app, ["health"])

    assert result.exit_code == 1
    assert "db down" in result.output
