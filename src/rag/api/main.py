import asyncio
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.auth import Principal, require_principal
from rag.api.routes.answer import router as answer_router
from rag.api.routes.auth import (
    PostgresUserAuthService,
    build_router as build_auth_router,
    install_default_session_resolver,
)
from rag.api.routes.community import router as community_router
from rag.api.routes.ingest import router as ingest_router
from rag.api.routes.jobs import router as jobs_router
from rag.api.routes.retrieve import router as retrieve_router
from rag.api.routes.search import router as search_router
from rag.api.routes.sources import router as sources_router
from rag.api.routes.workers import router as workers_router
from rag.config import settings
from rag.db import get_connection, prewarm_vector_indexes
from rag.graph_db import get_graph_driver, reconcile_schema

log = logging.getLogger(__name__)

# Stable references so tests can override via app.dependency_overrides.
principal_dep = require_principal()


async def _prewarm_with_retry(attempts: int = 5, base_delay: float = 1.0) -> None:
    """Warm the vector indexes off the startup path, retrying while Postgres
    comes up.

    Runs as a fire-and-forget background task so it never blocks or crashes
    backend startup. `docker compose restart` does not gate on the Postgres
    healthcheck, so the first attempt can hit a not-yet-ready server; back off
    and retry rather than paying the full cold penalty on the first query.
    pg_prewarm's autoprewarm worker covers Postgres-only restarts; this hook
    covers backend restarts and fresh installs, and gives immediate warmth
    without waiting for the autoprewarm dump interval.
    """
    for attempt in range(1, attempts + 1):
        try:
            blocks = await asyncio.to_thread(prewarm_vector_indexes)
            log.info("Prewarmed vector indexes; pg_prewarm block counts: %s", blocks)
            return
        except Exception:
            if attempt == attempts:
                log.exception(
                    "Vector-index prewarm failed after %d attempts; the first "
                    "dense query after this restart will page indexes from disk.",
                    attempts,
                )
                return
            await asyncio.sleep(base_delay * attempt)


def create_app() -> FastAPI:
    mcp_app = None
    if os.environ.get("RAG_DISABLE_MCP") != "1":
        from rag.mcp_server import build_mcp_app

        mcp_app = build_mcp_app()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            with get_graph_driver() as driver:
                reconcile_schema(driver)
        except Exception:
            log.exception(
                "Memgraph schema reconciliation failed at startup; graph-dependent "
                "routes may fail until this is resolved, but the API will still serve "
                "other traffic."
            )

        # Warm the vector indexes in the background so the first dense query
        # after this backend start is fast. Fire-and-forget: keep a reference so
        # it isn't garbage-collected, and cancel it on shutdown.
        prewarm_task = asyncio.create_task(_prewarm_with_retry())

        try:
            if mcp_app is not None and hasattr(mcp_app, "lifespan_context"):
                async with mcp_app.lifespan_context(app):
                    yield
            else:
                yield
        finally:
            prewarm_task.cancel()
            with suppress(asyncio.CancelledError):
                await prewarm_task

    app = FastAPI(title="RAG Explorer API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.RAG_FRONTEND_ORIGIN]
        + [
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        with get_connection() as conn:
            conn.execute("SELECT 1").fetchone()
        with get_graph_driver() as driver:
            with driver.session() as session:
                session.run("RETURN 1")
        return {"status": "ready"}

    # Public auth router (login, logout, /me — /me itself requires auth).
    user_service = PostgresUserAuthService()
    install_default_session_resolver(user_service)
    app.include_router(build_auth_router(user_service))

    # Gated routers — every request must produce a Principal.
    gated = [Depends(principal_dep)]
    app.include_router(search_router, dependencies=gated)
    app.include_router(retrieve_router, dependencies=gated)
    app.include_router(answer_router, dependencies=gated)
    app.include_router(sources_router, dependencies=gated)
    app.include_router(community_router, dependencies=gated)
    app.include_router(ingest_router, dependencies=gated)
    app.include_router(jobs_router, dependencies=gated)
    app.include_router(workers_router, dependencies=gated)

    if os.environ.get("RAG_BYPASS_AUTH") == "1":
        app.dependency_overrides[principal_dep] = lambda: Principal(
            kind="apikey", subject="test-suite", scopes=["read", "ingest", "admin"]
        )

    # Mount the MCP server under /mcp with shared API-key auth.
    if mcp_app is not None:
        app.mount("/mcp", mcp_app)
    return app


app = create_app()
