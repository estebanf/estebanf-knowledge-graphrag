#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.youtube_cleanup import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Delete active sources whose metadata kind is youtube."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete matching sources. Default is dry run.",
    )
    parser.add_argument(
        "--source-id",
        default=None,
        help="Limit the purge to a single source UUID.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit how many matching sources to inspect or delete.",
    )
    args = parser.parse_args(argv)

    run(execute=args.execute, source_id=args.source_id, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
