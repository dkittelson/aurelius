"""FastAPI application factory for Aurelius."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.dependencies import AppState, initialize_state, get_app_state
from src.api.routes import graph, predictions, forensic, dashboard, explain
from src.api.schemas import HealthResponse


# ---------------------------------------------------------------------------
# Lifespan — runs once at startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    dataset = os.getenv("DATASET", "elliptic")
    logger.info(f"Starting Aurelius API — loading dataset: {dataset}")
    initialize_state(dataset=dataset)
    yield
    logger.info("Aurelius API shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Aurelius",
        description="Graph-Native Financial Intelligence & Forensic Reasoning",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Allow Vite dev server on any port (5173, 5174, etc.)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(graph.router, prefix="/api/v1")
    app.include_router(predictions.router, prefix="/api/v1")
    app.include_router(forensic.router, prefix="/api/v1")
    app.include_router(dashboard.router, prefix="/api/v1")
    app.include_router(explain.router, prefix="/api/v1")

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health(state: AppState = Depends(get_app_state)):
        return HealthResponse(
            status="ok",
            model_loaded=state.model is not None,
            dataset=state.dataset_name,
        )

    return app


app = create_app()


# ---------------------------------------------------------------------------
# Entry point: python -m src.api.main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
