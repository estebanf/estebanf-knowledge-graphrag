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
    ) -> dict:
        """Detect entity communities. ``scope_mode`` must be ``ids`` | ``search`` | ``retrieve``."""
        from rag.community import detect_communities

        community_options = community_options or {}
        return detect_communities(
            scope_mode=scope_mode,
            source_ids=source_ids or [],
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

    return mcp


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
