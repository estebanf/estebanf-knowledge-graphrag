"""MCP server exposing read-only RAG tools over Streamable HTTP.

Tools wrap the existing REST endpoints' business logic directly so they share
the same code paths as the API:

- ``search`` → hybrid chunk + insight search
- ``retrieve`` → graph-aware retrieval
- ``community`` → community detection (ids / search / retrieve scope modes)
- ``list_sources`` → list recent sources with filters / fuzzy q
- ``source_insights`` → all insights linked to a source

Authentication is shared with the REST API via the same ``KeyStore``; the
``Authorization: Bearer <key>`` header is required on every request to ``/mcp``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP
from starlette.types import ASGIApp, Receive, Scope, Send

from rag.api.auth import KeyStore, get_default_keystore

log = logging.getLogger(__name__)


def _build_server() -> FastMCP:
    mcp = FastMCP(name="rag")
    # Mount-time path is just "/" so we can attach it cleanly under "/mcp".
    mcp.settings.streamable_http_path = "/"

    @mcp.tool()
    def search(query: str, limit: int = 10, min_score: float = 0.7) -> dict:
        """Hybrid search over chunks and insights. Returns a dict with ``chunks`` and ``insights`` ranked lists."""
        from rag.retrieval import hybrid_search

        results = hybrid_search(query, limit=limit, min_score=min_score)
        return {
            "chunks": [
                {
                    "score": r.score,
                    "chunk": r.chunk,
                    "chunk_id": r.chunk_id,
                    "source_id": r.source_id,
                    "source_path": r.source_path,
                    "source_metadata": r.source_metadata,
                }
                for r in results.chunks
            ],
            "insights": [
                {
                    "score": r.score,
                    "insight": r.insight,
                    "insight_id": r.insight_id,
                    "topics": r.topics,
                    "sources": [
                        {
                            "source_id": s.source_id,
                            "source_path": s.source_path,
                            "source_metadata": s.source_metadata,
                        }
                        for s in r.sources
                    ],
                }
                for r in results.insights
            ],
        }

    @mcp.tool()
    def retrieve(
        query: str,
        source_ids: Optional[list[str]] = None,
        filters: Optional[dict[str, str]] = None,
        seed_count: Optional[int] = None,
        result_count: Optional[int] = None,
    ) -> dict:
        """Graph-aware retrieval. Returns ``retrieval_results`` (root + related chunks) and ``insights``."""
        from rag.retrieval import retrieve as do_retrieve

        return do_retrieve(
            query=query,
            source_ids=source_ids or [],
            filters=filters or {},
            seed_count=seed_count,
            result_count=result_count,
        )

    @mcp.tool()
    def community(
        scope_mode: str,
        source_ids: Optional[list[str]] = None,
        criteria: Optional[list[str]] = None,
        filters: Optional[dict[str, str]] = None,
        search_options: Optional[dict[str, Any]] = None,
        retrieve_options: Optional[dict[str, Any]] = None,
        community_options: Optional[dict[str, Any]] = None,
        summarize_model: Optional[str] = None,
        working_set_id: Optional[str] = None,
    ) -> dict:
        """Detect entity communities. ``scope_mode`` must be ``ids`` | ``search`` | ``retrieve`` | ``working_set``. Read-only."""
        from rag.community import detect_communities
        from rag.db import get_connection

        community_options = community_options or {}

        resolved_ids = source_ids or []
        resolved_mode = scope_mode
        if scope_mode == "working_set":
            if not working_set_id:
                raise ValueError("working_set_id is required when scope_mode is 'working_set'")
            import json as _json
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT source_ids FROM working_sets WHERE id = %s",
                    (working_set_id,),
                ).fetchone()
            if row is None:
                raise ValueError(f"working set not found: {working_set_id}")
            source_ids_raw = row[0]
            if isinstance(source_ids_raw, str):
                source_ids_raw = _json.loads(source_ids_raw)
            resolved_ids = list(source_ids_raw) if isinstance(source_ids_raw, list) else []
            resolved_mode = "ids"

        return detect_communities(
            scope_mode=resolved_mode,
            source_ids=resolved_ids,
            criteria=criteria or [],
            filters=filters or {},
            search_options=search_options or {},
            retrieve_options=retrieve_options or {},
            semantic_threshold=community_options.get("semantic_threshold"),
            cutoff=community_options.get("cutoff"),
            min_community_size=community_options.get("min_community_size"),
            top_k_chunks=community_options.get("top_k_chunks"),
            summarize_model=summarize_model,
            cross_source_top_k=community_options.get("cross_source_top_k"),
            max_cross_source_queries=community_options.get("max_cross_source_queries"),
            source_cooc_weight=community_options.get("source_cooc_weight"),
            resolution=community_options.get("resolution"),
        )

    @mcp.tool()
    def list_sources(
        limit: int = 20,
        offset: int = 0,
        metadata: Optional[list[str]] = None,
        q: Optional[str] = None,
    ) -> dict:
        """List recent sources. ``metadata`` entries are ``key:value`` filter strings."""
        from rag.sources import list_recent_sources

        parsed_filters: list[tuple[str, str]] = []
        for item in metadata or []:
            key, sep, value = item.partition(":")
            if sep and key and value:
                parsed_filters.append((key.strip(), value.strip()))
        return list_recent_sources(limit=limit, offset=offset, metadata_filters=parsed_filters, q=q)

    @mcp.tool()
    def source_insights(source_id: str) -> dict:
        """Return all insights linked to chunks of the given source."""
        from rag.sources import list_source_insights

        return {"insights": list_source_insights(source_id)}

    @mcp.tool()
    def list_community_runs(limit: int = 20, offset: int = 0) -> dict:
        """List community detection runs. Returns ``runs`` list with totals. Read-only."""
        from rag.community_runs import list_runs

        return list_runs(limit=limit, offset=offset)

    @mcp.tool()
    def get_community_run(run_id: str) -> dict:
        """Get a single community run by id. Returns the run record. Read-only."""
        from rag.community_runs import get_run

        run = get_run(run_id)
        if run is None:
            raise ValueError(f"community run not found: {run_id}")
        return run

    @mcp.tool()
    def list_theme_reports(limit: int = 20, offset: int = 0) -> dict:
        """List theme reports. Returns ``reports`` list with totals. Read-only."""
        from rag.themes import list_reports

        return list_reports(limit=limit, offset=offset)

    @mcp.tool()
    def get_theme_report(report_id: str) -> dict:
        """Get a single theme report by id. Returns the full report. Read-only."""
        from rag.themes import get_report

        report = get_report(report_id)
        if report is None:
            raise ValueError(f"theme report not found: {report_id}")
        return report

    @mcp.tool()
    def list_answers(limit: int = 20, offset: int = 0) -> dict:
        """List saved answers. Returns ``answers`` list with totals. Read-only."""
        import json as _json
        from rag.db import get_connection

        with get_connection() as conn:
            total = conn.execute("SELECT count(*) FROM saved_answers").fetchone()[0]
            rows = conn.execute(
                """SELECT id, question, answer, model, params, evidence_snapshot, created_at
                   FROM saved_answers ORDER BY created_at DESC LIMIT %s OFFSET %s""",
                (limit, offset),
            ).fetchall()
        return {
            "answers": [_answer_row_to_dict(r, _json) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    @mcp.tool()
    def get_answer(answer_id: str) -> dict:
        """Get a single saved answer by id. Returns the answer record. Read-only."""
        import json as _json
        from rag.db import get_connection

        with get_connection() as conn:
            row = conn.execute(
                """SELECT id, question, answer, model, params, evidence_snapshot, created_at
                   FROM saved_answers WHERE id = %s""",
                (answer_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"answer not found: {answer_id}")
        return _answer_row_to_dict(row, _json)

    @mcp.tool()
    def list_working_sets() -> dict:
        """List working sets. Returns ``working_sets`` list. Read-only."""
        import json as _json
        from rag.db import get_connection

        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, name, source_ids, created_at, updated_at FROM working_sets ORDER BY name"
            ).fetchall()
        return {"working_sets": [_ws_row_to_dict(r, _json) for r in rows]}

    @mcp.tool()
    def get_working_set(ws_id: str) -> dict:
        """Get a single working set by id. Returns the working set record. Read-only."""
        import json as _json
        from rag.db import get_connection

        with get_connection() as conn:
            row = conn.execute(
                "SELECT id, name, source_ids, created_at, updated_at FROM working_sets WHERE id = %s",
                (ws_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f"working set not found: {ws_id}")
        return _ws_row_to_dict(row, _json)

    @mcp.tool()
    def list_metadata_facets() -> dict:
        """List metadata facet counts for kind, author, source, domain across all sources. Read-only."""
        from rag.db import get_connection

        facet_keys = ["kind", "author", "source", "domain"]
        result: dict[str, list[dict]] = {}
        with get_connection() as conn:
            for key in facet_keys:
                rows = conn.execute(
                    """SELECT COALESCE(metadata->>%s, '(none)') AS value, count(*) AS cnt
                       FROM sources WHERE deleted_at IS NULL
                       GROUP BY 1 ORDER BY cnt DESC""",
                    (key,),
                ).fetchall()
                result[key] = [{"value": r[0], "count": r[1]} for r in rows]
        return {"facets": result}

    return mcp


def _answer_row_to_dict(row, json_mod) -> dict:
    evidence = row[5]
    if isinstance(evidence, str):
        try:
            evidence = json_mod.loads(evidence)
        except (json_mod.JSONDecodeError, TypeError):
            evidence = []
    params = row[4]
    if isinstance(params, str):
        try:
            params = json_mod.loads(params)
        except (json_mod.JSONDecodeError, TypeError):
            params = {}
    return {
        "id": str(row[0]),
        "question": row[1],
        "answer": row[2],
        "model": row[3] or "",
        "params": params if isinstance(params, dict) else {},
        "evidence_snapshot": evidence if isinstance(evidence, list) else [],
        "created_at": row[6].isoformat() if row[6] else None,
    }


def _ws_row_to_dict(row, json_mod) -> dict:
    source_ids = row[2]
    if isinstance(source_ids, str):
        try:
            source_ids = json_mod.loads(source_ids)
        except (json_mod.JSONDecodeError, TypeError):
            source_ids = []
    source_ids = source_ids if isinstance(source_ids, list) else []
    return {
        "id": str(row[0]),
        "name": row[1],
        "source_ids": source_ids,
        "source_count": len(source_ids),
        "created_at": row[3].isoformat() if row[3] else None,
        "updated_at": row[4].isoformat() if row[4] else None,
    }


def build_mcp_app(*, keystore: Optional[KeyStore] = None) -> ASGIApp:
    """Build the auth-gated MCP ASGI app to mount under ``/mcp``."""
    mcp = _build_server()
    inner = mcp.streamable_http_app()
    wrapped = BearerAuthMiddleware(inner, keystore or get_default_keystore())
    # Expose the inner app's lifespan so the parent FastAPI app can run it.
    wrapped.lifespan_context = inner.router.lifespan_context  # type: ignore[attr-defined]
    return wrapped


class BearerAuthMiddleware:
    """ASGI middleware: require a valid Bearer token before reaching MCP."""

    def __init__(self, app: ASGIApp, keystore: KeyStore) -> None:
        self.app = app
        self.keystore = keystore

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        auth = headers.get("authorization", "")
        token = ""
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        if not token or self.keystore.lookup(token) is None:
            await self._send_401(send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _send_401(send: Send) -> None:
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": b'{"detail":"missing or invalid api key"}'})
