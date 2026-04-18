from pathlib import Path

from rag.storage import store_file


def test_store_file_uses_versioned_original_name(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.storage.settings.STORAGE_BASE_PATH", tmp_path)
    source_file = tmp_path / "input.pdf"
    source_file.write_bytes(b"content")

    stored_path = store_file("source-123", source_file, version=3)

    assert stored_path == tmp_path / "source-123" / "3" / "original_input.pdf"
    assert stored_path.exists()
