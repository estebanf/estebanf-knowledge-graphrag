from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rag.cli import app


runner = CliRunner()


@patch("rag.cli.delete_stored_file")
@patch("rag.cli.get_connection")
def test_sources_delete_hard_removes_chunks_before_jobs(mock_conn, mock_delete_file):
    conn = MagicMock()
    mock_conn.return_value.__enter__.return_value = conn
    conn.execute.return_value.fetchone.return_value = ("/tmp/source/file.md",)

    result = runner.invoke(app, ["sources", "delete", "source-1", "--hard"])

    assert result.exit_code == 0
    sql_calls = [call.args[0] for call in conn.execute.call_args_list]
    assert sql_calls[1] == "DELETE FROM entities WHERE source_id = %s"
    assert sql_calls[2] == "DELETE FROM chunks WHERE source_id = %s"
    assert sql_calls[3] == "DELETE FROM jobs WHERE source_id = %s"
    assert sql_calls[4] == "DELETE FROM sources WHERE id = %s"
    mock_delete_file.assert_called_once_with("source-1")
