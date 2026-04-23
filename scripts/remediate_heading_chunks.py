#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.remediation import remediate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-queue completed jobs that produced heading-only chunks."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print affected sources without changing data.")
    parser.add_argument("--source-id", default=None, help="Limit remediation to a single source UUID.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of affected sources.")
    args = parser.parse_args()

    affected = remediate(
        dry_run=args.dry_run,
        only_source_id=args.source_id,
        limit=args.limit,
    )

    if not affected:
        print("No affected sources found.")
        return 0

    prefix = "DRY RUN - " if args.dry_run else ""
    print(f"{prefix}Found {len(affected)} affected source(s):")
    for source in affected:
        print(
            f"  {source.file_name} "
            f"source={source.source_id} "
            f"job={source.job_id} "
            f"heading_only={source.heading_only_chunks}"
        )
    if not args.dry_run:
        print("Remediation complete. Start the worker to process the re-queued jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
