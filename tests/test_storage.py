from pathlib import Path

from rag.storage import store_file, store_markdown_images


def test_store_file_uses_versioned_original_name(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.storage.settings.STORAGE_BASE_PATH", tmp_path)
    source_file = tmp_path / "input.pdf"
    source_file.write_bytes(b"content")

    stored_path = store_file("source-123", source_file, version=3)

    assert stored_path == tmp_path / "source-123" / "3" / "original_input.pdf"
    assert stored_path.exists()


def test_store_markdown_images_copies_referenced_images(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.storage.settings.STORAGE_BASE_PATH", tmp_path)
    img_dir = tmp_path / "docs"
    img_dir.mkdir()
    img_file = img_dir / "chart.png"
    img_file.write_bytes(b"PNG_DATA")
    md_file = img_dir / "report.md"
    md_file.write_text("# Report\n\n![chart](chart.png)\n", encoding="utf-8")

    store_markdown_images("source-abc", md_file, version=1)

    dest = tmp_path / "source-abc" / "1" / "chart.png"
    assert dest.exists()
    assert dest.read_bytes() == b"PNG_DATA"


def test_store_markdown_images_skips_remote_urls(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.storage.settings.STORAGE_BASE_PATH", tmp_path)
    md_file = tmp_path / "report.md"
    md_file.write_text("![x](https://example.com/img.png)\n", encoding="utf-8")

    store_markdown_images("source-abc", md_file, version=1)

    dest = tmp_path / "source-abc" / "1"
    remote_files = list(dest.glob("**/*.png")) if dest.exists() else []
    assert remote_files == []


def test_store_markdown_images_skips_missing_images(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.storage.settings.STORAGE_BASE_PATH", tmp_path)
    md_file = tmp_path / "report.md"
    md_file.write_text("![x](missing.png)\n", encoding="utf-8")

    # Should not raise
    store_markdown_images("source-abc", md_file, version=1)
