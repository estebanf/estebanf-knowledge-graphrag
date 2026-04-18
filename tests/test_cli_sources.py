from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rag.cli import app


runner = CliRunner()


@patch("rag.cli.delete_stored_file")
@patch("rag.ingestion.get_graph_driver")
@patch("rag.cli.get_connection")
def test_sources_delete_hard_removes_chunks_before_jobs(mock_conn, mock_graph_driver, mock_delete_file):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("/tmp/source/file.md",)
    conn.execute.return_value.fetchall.return_value = []

    mock_driver = MagicMock()
    mock_graph_driver.return_value.__enter__.return_value = mock_driver

    result = runner.invoke(app, ["sources", "delete", "source-1", "--hard"])

    assert result.exit_code == 0
    sql_calls = [call.args[0] for call in conn.execute.call_args_list]
    # DELETE order must be: entities → chunks → jobs → sources
    delete_calls = [s for s in sql_calls if s.startswith("DELETE")]
    assert delete_calls[0] == "DELETE FROM entities WHERE source_id = %s"
    assert delete_calls[1] == "DELETE FROM chunks WHERE source_id = %s"
    assert delete_calls[2] == "DELETE FROM jobs WHERE source_id = %s"
    assert delete_calls[3] == "DELETE FROM sources WHERE id = %s"
    mock_delete_file.assert_called_once_with("source-1")
