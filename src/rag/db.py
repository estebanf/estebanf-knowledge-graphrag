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
