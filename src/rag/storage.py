import re
import shutil
from pathlib import Path

from rag.config import settings

_IMAGE_REF_RE = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
_REMOTE_PREFIXES = ("http://", "https://", "data:")


def store_file(source_id: str, file_path: Path, version: int = 1) -> Path:
    dest_dir = settings.STORAGE_BASE_PATH / source_id / str(version)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"original_{file_path.name}"
    shutil.copy2(file_path, dest)
    return dest


def store_markdown_images(source_id: str, markdown_path: Path, version: int) -> None:
    content = markdown_path.read_text(encoding="utf-8", errors="replace")
    dest_dir = settings.STORAGE_BASE_PATH / source_id / str(version)
    for match in _IMAGE_REF_RE.finditer(content):
        path_str = match.group(1)
        if path_str.startswith(_REMOTE_PREFIXES):
            continue
        img_src = markdown_path.parent / path_str
        if not img_src.exists():
            continue
        img_dest = dest_dir / path_str
        img_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_src, img_dest)


def delete_stored_file(source_id: str) -> None:
    dest_dir = settings.STORAGE_BASE_PATH / source_id
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
