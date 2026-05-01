import uuid
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rag.cli import app


runner = CliRunner()


@patch("rag.sources.get_source_detail", return_value={"markdown_content": "# Title\n\nBody"})
def test_source_command_prints_markdown_only(mock_get_source_detail):
    result = runner.invoke(app, ["source", "source-1"])

    assert result.exit_code == 0
    assert result.stdout == "# Title\n\nBody\n"
    mock_get_source_detail.assert_called_once()


@patch("rag.sources.get_source_detail", return_value={"markdown_content": None})
def test_source_command_returns_empty_output_for_null_markdown(mock_get_source_detail):
    result = runner.invoke(app, ["source", "source-1"])

    assert result.exit_code == 0
    assert result.stdout == "\n"


@patch("rag.sources.get_source_detail", return_value=None)
def test_source_command_exits_when_source_missing(mock_get_source_detail):
    result = runner.invoke(app, ["source", "missing"])

    assert result.exit_code == 1
    assert "Source not found: missing" in result.stdout


@patch("rag.storage.delete_stored_file")
@patch("rag.cli._get_graph_driver")
@patch("rag.cli._get_connection")
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
    # DELETE order must be: insights join rows/orphans, entities, chunks, jobs, sources
    delete_calls = [s for s in sql_calls if s.startswith("DELETE")]
    assert "DELETE FROM chunk_insights" in delete_calls[0]
    assert "DELETE FROM insights" in delete_calls[1]
    assert delete_calls[2] == "DELETE FROM entities WHERE source_id = %s"
    assert delete_calls[3] == "DELETE FROM chunks WHERE source_id = %s"
    assert delete_calls[4] == "DELETE FROM jobs WHERE source_id = %s"
    assert delete_calls[5] == "DELETE FROM sources WHERE id = %s"
    mock_delete_file.assert_called_once_with("source-1")


def test_sources_insights_prints_table():
    source_id = str(uuid.uuid4())
    fake_rows = [
        (
            uuid.uuid4(),
            "AI reduces operational costs by 30%",
            ["AI Adoption", "Business Outcomes"],
            "2026-01-10 12:00:00",
        ),
    ]
    with patch("rag.cli._get_connection") as mock_conn:
        cur = MagicMock()
        cur.fetchall.return_value = fake_rows
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cur

        result = runner.invoke(app, ["sources", "insights", source_id])

    assert result.exit_code == 0
    assert "AI reduces" in result.output
    assert "Business Outcomes" in result.output


def test_sources_insights_empty():
    with patch("rag.cli._get_connection") as mock_conn:
        cur = MagicMock()
        cur.fetchall.return_value = []
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cur

        result = runner.invoke(app, ["sources", "insights", "some-id"])

    assert result.exit_code == 0
    assert "No insights" in result.output


def test_sources_last_with_integer():
    fake_rows = [(uuid.uuid4(),), (uuid.uuid4(),)]
    with patch("rag.cli._get_connection") as mock_conn:
        cur = MagicMock()
        cur.fetchall.return_value = fake_rows
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cur

        result = runner.invoke(app, ["sources", "last", "5"])

    assert result.exit_code == 0
    assert "LIMIT" in cur.execute.call_args.args[0]
    assert str(fake_rows[0][0]) in result.output


def test_sources_last_with_date_string():
    with patch("rag.cli._get_connection") as mock_conn:
        cur = MagicMock()
        cur.fetchall.return_value = []
        mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value = cur

        result = runner.invoke(app, ["sources", "last", "2026-01-01"])

    assert result.exit_code == 0
    assert "created_at >=" in cur.execute.call_args.args[0]
    assert "No sources" in result.output
