from contextlib import contextmanager
from typing import Generator

import psycopg

from rag.config import settings


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    conn = psycopg.connect(settings.POSTGRES_URL)
    try:
        yield conn
    finally:
        conn.close()


def vacuum_analyze_entities() -> None:
    """Reclaim dead rows left by entity-merge scripts.

    VACUUM cannot run inside a transaction block, so this opens its own
    autocommit connection rather than reusing a caller's connection.
    """
    conn = psycopg.connect(settings.POSTGRES_URL, autocommit=True)
    try:
        conn.execute("VACUUM (ANALYZE) entities")
    finally:
        conn.close()
