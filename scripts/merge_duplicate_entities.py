#!/usr/bin/env python3
"""Merge duplicate Entity nodes (same canonical_name + entity_type) across Memgraph and PostgreSQL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse

from rag.db import get_connection
from rag.graph_db import get_graph_driver


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge duplicate entities.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Print what would be merged.")
    group.add_argument("--execute", action="store_true", help="Perform the merge.")
    args = parser.parse_args()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
