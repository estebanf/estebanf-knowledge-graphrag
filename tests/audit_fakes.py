"""Offline doubles for a psycopg connection and a Memgraph session.

Used by the entity-audit and weekly-maintenance tests. Nothing here touches a
real database: every statement is recorded and answered by a `responder(sql,
params)` callable that returns rows (a list of tuples/dicts) or, for writes, an
int rowcount.
"""

import re

_WRITE_RE = re.compile(r"^\s*(INSERT|UPDATE|DELETE|TRUNCATE|CREATE|DROP|ALTER|VACUUM)\b", re.I)
_CYPHER_WRITE_RE = re.compile(r"\b(DELETE|DETACH|SET|MERGE|CREATE|REMOVE)\b")


class FakeCursor:
    def __init__(self, rows=None, rowcount=0):
        self._rows = list(rows or [])
        self.rowcount = rowcount

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _CursorCtx:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def executemany(self, sql, rows):
        self._conn.executemany_calls.append((sql, list(rows)))


class FakeConn:
    def __init__(self, responder=None):
        self.responder = responder or (lambda sql, params: [])
        self.executed: list[tuple[str, object]] = []
        self.executemany_calls: list[tuple[str, list]] = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        result = self.responder(sql, params)
        if isinstance(result, int):
            return FakeCursor(rowcount=result)
        rows = result or []
        return FakeCursor(rows=rows, rowcount=len(rows))

    def cursor(self):
        return _CursorCtx(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    # -- helpers for assertions -------------------------------------------
    def sql_matching(self, needle: str) -> list[tuple[str, object]]:
        return [(sql, params) for sql, params in self.executed if needle in sql]

    def write_statements(self) -> list[str]:
        writes = [sql for sql, _ in self.executed if _WRITE_RE.match(sql)]
        writes += [sql for sql, _ in self.executemany_calls]
        return writes


class FakeSession:
    def __init__(self, responder=None):
        self.responder = responder or (lambda query, params: [])
        self.calls: list[tuple[str, dict]] = []

    def run(self, query, **params):
        self.calls.append((query, params))
        return list(self.responder(query, params) or [])

    def queries_matching(self, needle: str) -> list[tuple[str, dict]]:
        return [(q, p) for q, p in self.calls if needle in q]

    def write_queries(self) -> list[str]:
        return [q for q, _ in self.calls if _CYPHER_WRITE_RE.search(q)]


class FakeDriver:
    """`with driver.session() as session` -> the given FakeSession."""

    def __init__(self, session):
        self._session = session

    def session(self):
        outer = self._session

        class _Ctx:
            def __enter__(self_inner):
                return outer

            def __exit__(self_inner, *exc):
                return False

        return _Ctx()


def forbid_live_stores(monkeypatch, *modules):
    """Make any attempt to reach a real Postgres or Memgraph fail loudly.

    Patches `psycopg.connect`, the neo4j driver factory and `rag.db.get_connection` /
    `rag.graph_db.get_graph_driver`, plus the same names on any extra `modules`
    (a script module binds them at import time). A test that needs a connection
    must stub it itself (which overrides this guard); a test that forgets would
    otherwise silently talk to whatever `.env` points at.
    """

    def _no_connection(*args, **kwargs):
        raise AssertionError("test reached a real Postgres connection; stub get_connection")

    def _no_driver(*args, **kwargs):
        raise AssertionError("test reached a real Memgraph driver; stub get_graph_driver")

    import psycopg
    from neo4j import GraphDatabase

    monkeypatch.setattr(psycopg, "connect", _no_connection)
    monkeypatch.setattr(GraphDatabase, "driver", _no_driver)
    monkeypatch.setattr("rag.db.get_connection", _no_connection)
    monkeypatch.setattr("rag.graph_db.get_graph_driver", _no_driver)
    for module in modules:
        if hasattr(module, "get_connection"):
            monkeypatch.setattr(module, "get_connection", _no_connection)
        if hasattr(module, "get_graph_driver"):
            monkeypatch.setattr(module, "get_graph_driver", _no_driver)


def ctx(value):
    """Context manager yielding `value` (stands in for get_connection()/get_graph_driver())."""

    class _Ctx:
        def __enter__(self):
            return value

        def __exit__(self, *exc):
            return False

    return _Ctx()
