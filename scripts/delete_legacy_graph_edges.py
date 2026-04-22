#!/usr/bin/env python3
import argparse

from rag.graph_db import get_graph_driver


def collect_edge_counts(driver) -> dict[str, int]:
    counts: dict[str, int] = {}
    with driver.session() as session:
        for rel_type in ("RELATED_TO", "MENTIONED_IN"):
            rows = session.run(
                f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS count"
            )
            counts[rel_type] = int(next(iter(rows))["count"])
    return counts


def delete_legacy_edges(driver) -> None:
    with driver.session() as session:
        for rel_type in ("RELATED_TO", "MENTIONED_IN"):
            session.run(f"MATCH ()-[r:{rel_type}]->() DELETE r")


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete deprecated Memgraph edges.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned deletions only.")
    parser.add_argument("--execute", action="store_true", help="Delete the deprecated edges.")
    args = parser.parse_args()

    if args.dry_run == args.execute:
        parser.error("Choose exactly one of --dry-run or --execute.")

    with get_graph_driver() as driver:
        before = collect_edge_counts(driver)
        print(before)
        if args.dry_run:
            return 0
        delete_legacy_edges(driver)
        after = collect_edge_counts(driver)
        print(after)
        if any(after.values()):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
