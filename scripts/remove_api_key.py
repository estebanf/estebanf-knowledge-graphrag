#!/usr/bin/env python3
"""Remove an API key block from data/api_keys.toml, rewriting the file IN PLACE.

The backend container bind-mounts api_keys.toml as a single file, so the file's
inode must not change (no rename/replace, no `sed -i`). This script rewrites the
contents through the same open file handle (seek/write/truncate).

Usage:
    python scripts/remove_api_key.py --id smoke-cli [--path data/api_keys.toml]

No-op (exit 0) when the file or the id does not exist. Exit 1 if the result would
not be valid TOML (the file is left untouched in that case).
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

_HEADER_RE = re.compile(r"^\s*\[\[?\s*[A-Za-z0-9_.\"' -]+\s*\]\]?\s*(#.*)?$")
_KEYS_HEADER_RE = re.compile(r"^\s*\[\[\s*keys\s*\]\]\s*(#.*)?$")


def _split_segments(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_keys_block, text) segments at TOML table headers."""
    segments: list[tuple[bool, list[str]]] = [(False, [])]
    for line in text.splitlines(keepends=True):
        if _KEYS_HEADER_RE.match(line):
            segments.append((True, [line]))
        elif _HEADER_RE.match(line):
            segments.append((False, [line]))
        else:
            segments[-1][1].append(line)
    return [(is_keys, "".join(lines)) for is_keys, lines in segments]


def remove_key_block(text: str, key_id: str) -> tuple[str, int]:
    """Return (new_text, removed_count) with every [[keys]] block whose id matches dropped."""
    kept: list[str] = []
    removed = 0
    for is_keys, segment in _split_segments(text):
        if is_keys:
            try:
                parsed = tomllib.loads(segment)
            except tomllib.TOMLDecodeError:
                parsed = {}
            entries = parsed.get("keys") or [{}]
            if entries[0].get("id") == key_id:
                removed += 1
                continue
        kept.append(segment)
    return "".join(kept), removed


def remove_api_key(path: Path, key_id: str) -> int:
    """Remove the block in place, preserving the inode. Returns number of blocks removed."""
    if not path.exists():
        return 0
    with open(path, "r+", encoding="utf-8") as fh:
        original = fh.read()
        new_text, removed = remove_key_block(original, key_id)
        if removed == 0:
            return 0
        parsed = tomllib.loads(new_text)  # raises TOMLDecodeError if we broke the file
        if any(k.get("id") == key_id for k in parsed.get("keys", [])):
            raise ValueError(f"key id {key_id!r} still present after removal")
        fh.seek(0)
        fh.write(new_text)
        fh.truncate()
        fh.flush()
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remove an API key block from api_keys.toml in place.")
    parser.add_argument("--id", required=True, help="The key id to remove.")
    parser.add_argument("--path", default="data/api_keys.toml", type=Path)
    args = parser.parse_args(argv)
    try:
        removed = remove_api_key(args.path, args.id)
    except (tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"refusing to rewrite {args.path}: {exc}", file=sys.stderr)
        return 1
    print(f"removed {removed} key block(s) with id {args.id!r} from {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
