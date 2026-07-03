from unittest.mock import patch

from typer.testing import CliRunner

from rag.cli import app
from rag.prepare import PrepareError, PreparedDocument

runner = CliRunner()


def _prepared(image_count: int = 0) -> PreparedDocument:
    return PreparedDocument(
        markdown="# prepared\n\ncontent",
        images=[],
        original_filename="test.pdf",
        original_extension="pdf",
        original_md5="pdfhash",
        image_count=image_count,
    )


@patch("rag.cli.finalize_markdown", return_value="# prepared\n\ncontent")
@patch("rag.cli.prepare_binary", return_value=_prepared())
@patch("rag.cli.submit_ingestion_job")
def test_ingest_binary_prepares_then_submits_markdown(mock_submit, mock_prepare, mock_final, tmp_path):
    # Direct-DB mode: a PDF is prepared locally into markdown, then submitted with
    # original-source provenance (original hash/filename/type).
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF fake")
    mock_submit.return_value = {"source_id": "s1", "job_id": "j1", "status": "pending"}

    result = runner.invoke(app, ["ingest", str(f)])

    assert result.exit_code == 0, result.output
    assert "j1" in result.output
    mock_prepare.assert_called_once()
    _, kwargs = mock_submit.call_args
    assert kwargs["original_md5"] == "pdfhash"
    assert kwargs["original_file_name"] == "test.pdf"
    assert kwargs["original_file_type"] == "pdf"
    # The submitted file is prepared markdown, not the original binary.
    assert mock_submit.call_args.args[0].suffix == ".md"


@patch("rag.cli.prepare_binary")
@patch("rag.cli.submit_ingestion_job")
def test_ingest_text_file_submits_directly_without_prepare(mock_submit, mock_prepare, tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# notes", encoding="utf-8")
    mock_submit.return_value = {"source_id": "s1", "job_id": "j1", "status": "pending"}

    result = runner.invoke(app, ["ingest", str(f)])

    assert result.exit_code == 0, result.output
    assert "j1" in result.output
    mock_prepare.assert_not_called()
    assert mock_submit.call_args.args[0].name == "notes.md"


@patch("rag.cli.finalize_markdown", return_value="# prepared")
@patch("rag.cli.prepare_binary", return_value=_prepared())
@patch("rag.cli.submit_ingestion_job")
def test_ingest_multiple_files_shows_table(mock_submit, mock_prepare, mock_final, tmp_path):
    f1 = tmp_path / "file1.pdf"
    f2 = tmp_path / "file2.md"
    f1.write_bytes(b"%PDF")
    f2.write_text("md", encoding="utf-8")

    mock_submit.side_effect = [
        {"source_id": "s1", "job_id": "job-aaa", "status": "pending"},
        {"source_id": "s2", "job_id": "job-bbb", "status": "pending"},
    ]

    result = runner.invoke(app, ["ingest", str(f1), str(f2)])

    assert result.exit_code == 0, result.output
    assert "job-aaa" in result.output
    assert "job-bbb" in result.output
    mock_prepare.assert_called_once()  # only the PDF is prepared


@patch("rag.cli.finalize_markdown", return_value="# prepared")
@patch("rag.cli.prepare_binary", return_value=_prepared())
@patch("rag.cli.submit_ingestion_job")
def test_ingest_folder_finds_supported_files(mock_submit, mock_prepare, mock_final, tmp_path):
    (tmp_path / "doc.pdf").write_bytes(b"%PDF")
    (tmp_path / "notes.txt").write_bytes(b"txt content")
    (tmp_path / "slides.pptx").write_bytes(b"pptx content")

    mock_submit.return_value = {"source_id": "sx", "job_id": "jx", "status": "pending"}

    result = runner.invoke(app, ["ingest", str(tmp_path)])

    assert result.exit_code == 0
    assert mock_submit.call_count == 3
    assert mock_prepare.call_count == 2  # pdf + pptx, not the txt


@patch("rag.cli.submit_ingestion_job")
def test_ingest_folder_skips_unsupported(mock_submit, tmp_path):
    (tmp_path / "doc.xyz").write_bytes(b"unsupported content")

    result = runner.invoke(app, ["ingest", str(tmp_path)])

    assert result.exit_code == 0
    assert "No supported files found" in result.output
    mock_submit.assert_not_called()


@patch("rag.cli.submit_ingestion_job")
def test_ingest_folder_empty_dir(mock_submit, tmp_path):
    result = runner.invoke(app, ["ingest", str(tmp_path)])

    assert result.exit_code == 0
    assert "No supported files" in result.output
    mock_submit.assert_not_called()


@patch("rag.cli.prepare_binary", side_effect=PrepareError("docling missing"))
@patch("rag.cli.submit_ingestion_job")
def test_ingest_preparation_failure_queues_no_job(mock_submit, mock_prepare, tmp_path):
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF")

    result = runner.invoke(app, ["ingest", str(f)])

    assert result.exit_code != 0
    assert "Preparation failed" in result.output
    mock_submit.assert_not_called()
