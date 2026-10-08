"""
backend/app/main.py
--------------------
FastAPI application factory.

Start from the project root with:
    uvicorn backend.app.main:app --reload --port 8000

The lifespan context manager loads the models exactly once when the server
starts and releases them cleanly on shutdown.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.settings import get_settings

# ---------------------------------------------------------------------------
# Make sure the project root is importable as a package root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan: model loading
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load models on startup; nothing special on shutdown."""
    logger.info("Starting MargDrishti backend …")
    from backend.app.services.model_registry import load_all_models
    load_all_models()
    logger.info("Models loaded.  Ready to serve requests.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="MargDrishti API",
        description=(
            "Serves predictions from MargNet (custom CNN) and ResNet50 fine-tuned "
            "on the Indian Traffic SignBoards dataset, and exposes all stored "
            "experiment results for the comparison web app."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ------------------------------------------------------------------ #
    # CORS – allow the Vite dev server to call the API                    #
    # ------------------------------------------------------------------ #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Static file mounts                                                   #
    # ------------------------------------------------------------------ #
    _mount_if_exists(app, "/static/figures", settings.FIGURES_DIR, "figures")
    _mount_if_exists(app, "/static/samples", settings.SAMPLES_DIR, "samples")

    # ------------------------------------------------------------------ #
    # Routers – all endpoints live under /api                             #
    # ------------------------------------------------------------------ #
    from backend.app.routers import health, predict, metrics, dataset

    app.include_router(health.router, prefix="/api", tags=["Health"])
    app.include_router(predict.router, prefix="/api", tags=["Predict"])
    app.include_router(metrics.router, prefix="/api", tags=["Metrics"])
    app.include_router(dataset.router, prefix="/api", tags=["Dataset"])

    return app


def _mount_if_exists(app: FastAPI, path: str, directory: Path, name: str) -> None:
    """Mount a static directory only if it exists (graceful degradation)."""
    if directory.exists():
        app.mount(path, StaticFiles(directory=str(directory)), name=name)
        logger.info("Mounted static files: %s -> %s", path, directory)
    else:
        logger.warning("Static directory not found, skipping mount: %s", directory)


# ---------------------------------------------------------------------------
# Module-level app instance (used by uvicorn)
# ---------------------------------------------------------------------------
app = create_app()
