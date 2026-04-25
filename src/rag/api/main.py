from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.routes.answer import router as answer_router
from rag.api.routes.community import router as community_router
from rag.api.routes.retrieve import router as retrieve_router
from rag.api.routes.search import router as search_router
from rag.api.routes.sources import router as sources_router
from rag.db import get_connection
from rag.graph_db import get_graph_driver


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Explorer API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
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

    app.include_router(search_router)
    app.include_router(retrieve_router)
    app.include_router(answer_router)
    app.include_router(sources_router)
    app.include_router(community_router)
    return app


app = create_app()
