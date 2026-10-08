"""
backend/app/routers/health.py
------------------------------
GET /api/health
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas import HealthResponse, ModelLoadStatus
from backend.app.services import artifacts, model_registry

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="API health check")
async def health() -> HealthResponse:
    """
    Returns the server status, the device in use, which models are loaded,
    and the timestamp when the artifact manifest was generated.
    """
    manifest = artifacts.load_manifest()
    all_m = model_registry.all_models()

    model_statuses: dict[str, ModelLoadStatus] = {}
    for key, entry in all_m.items():
        model_statuses[key] = ModelLoadStatus(
            available=entry.available,
            error=entry.error,
        )

    # Determine device from any available model
    device_str = "cpu"
    for entry in all_m.values():
        if entry.available and entry.model is not None:
            param = next(iter(entry.model.parameters()), None)
            if param is not None:
                device_str = str(param.device)
            break

    return HealthResponse(
        status="ok",
        device=device_str,
        models=model_statuses,
        manifest_generated_at=manifest.get("generated_at") if manifest else None,
    )
