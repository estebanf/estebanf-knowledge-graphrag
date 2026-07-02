from contextlib import contextmanager

from neo4j import GraphDatabase

from rag.config import settings

# Mirrors scripts/init/memgraph_init.cypher. Kept here (rather than parsed from
# that file at runtime) so schema reconciliation has no packaging/path
# dependency. Every statement must be safe to re-run (CREATE INDEX/CONSTRAINT
# in Memgraph is a no-op when the index/constraint already exists) — do not
# add a statement here that isn't idempotent.
SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT ON (s:Source) ASSERT s.source_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (c:Chunk) ASSERT c.chunk_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (e:Entity) ASSERT e.entity_id IS UNIQUE;",
    "CREATE CONSTRAINT ON (i:Insight) ASSERT i.insight_id IS UNIQUE;",
    "CREATE INDEX ON :Entity(canonical_name);",
    "CREATE INDEX ON :Entity(entity_id);",
    "CREATE INDEX ON :Entity(entity_type);",
    "CREATE INDEX ON :Chunk(chunk_id);",
    "CREATE INDEX ON :Chunk(source_id);",
    "CREATE INDEX ON :Insight(insight_id);",
]


@contextmanager
def get_graph_driver():
    driver = GraphDatabase.driver(settings.MEMGRAPH_URL, auth=None)
    try:
        yield driver
    finally:
        driver.close()


def reconcile_schema(driver) -> None:
    """Idempotently (re-)apply the Memgraph index/constraint statements.

    Guards against schema drift: these statements are normally applied once
    via scripts/init/memgraph_init.cypher, but nothing re-applies them if a
    later statement (e.g. the Insight index) is added after an instance was
    already initialized. Raises on connection/statement failure rather than
    swallowing it, so a broken Memgraph connection is visible at startup.
    """
    with driver.session() as session:
        for statement in SCHEMA_STATEMENTS:
            session.run(statement)
