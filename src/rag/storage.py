import shutil
from pathlib import Path

from rag.config import settings


def store_file(source_id: str, file_path: Path, version: int = 1) -> Path:
    dest_dir = settings.STORAGE_BASE_PATH / source_id / str(version)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"original_{file_path.name}"
    shutil.copy2(file_path, dest)
    return dest


def delete_stored_file(source_id: str) -> None:
    dest_dir = settings.STORAGE_BASE_PATH / source_id
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
