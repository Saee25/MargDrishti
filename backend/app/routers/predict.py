"""
backend/app/routers/predict.py
-------------------------------
POST /api/predict          - predict from an uploaded image
POST /api/predict/sample   - predict from one of the stored sample crops
GET  /api/classes          - list all K classes
GET  /api/samples          - list all stored sample crops with their true labels
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.app.schemas import (
    ClassEntry,
    ClassesResponse,
    PREDICTION_NOTE,
    PredictResponse,
    SampleEntry,
    SamplePredictResponse,
    SamplesResponse,
)
from backend.app.services import artifacts, model_registry
from backend.app.services.inference import open_and_validate_image, run_predict
from backend.app.settings import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_models(model_param: str) -> list[str]:
    """Parse the 'model' form field into a list of model keys."""
    p = model_param.strip().lower()
    if p == "both":
        return ["margnet", "resnet50"]
    if p in ("margnet", "resnet50"):
        return [p]
    raise HTTPException(status_code=400, detail="model must be 'margnet', 'resnet50', or 'both'")


async def _run_for_keys(
    model_keys: list[str],
    image_bytes: bytes,
    top_k: int,
    do_gradcam: bool,
) -> list:
    """
    For each requested model key, validate that the model is available,
    then dispatch the CPU-bound inference call to a thread-pool executor.
    Returns a list of SingleModelResult objects.
    """
    class_map = artifacts.load_class_map() or []
    loop = asyncio.get_event_loop()
    results = []

    for key in model_keys:
        entry = model_registry.get_model(key)
        if not entry.available:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{key}' is unavailable: {entry.error or 'not loaded'}",
            )
        try:
            from PIL import Image  # type: ignore
            import io
            img = open_and_validate_image(image_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        # Acquire the per-model lock to prevent concurrent forward passes
        # from corrupting the gradient state during GradCAM.
        async with entry.lock:
            result = await loop.run_in_executor(
                None,
                run_predict,
                entry,
                img,
                top_k,
                class_map,
                do_gradcam,
            )
        results.append(result)

    return results


def _check_agreement(results: list) -> Optional[bool]:
    """Return True if all results agree on top-1, False if they disagree, None if <2 results."""
    if len(results) < 2:
        return None
    top1s = [r.top_k[0].class_index for r in results if r.top_k]
    if len(set(top1s)) == 1:
        return True
    return False


# ---------------------------------------------------------------------------
# GET /classes
# ---------------------------------------------------------------------------

@router.get("/classes", response_model=ClassesResponse, summary="List all traffic sign classes")
async def get_classes() -> ClassesResponse:
    """Return the index, slug and display name of all K kept classes."""
    class_map = artifacts.load_class_map()
    if class_map is None:
        return ClassesResponse(count=0, classes=[])
    return ClassesResponse(
        count=len(class_map),
        classes=[ClassEntry(**c) for c in class_map],
    )


# ---------------------------------------------------------------------------
# GET /samples
# ---------------------------------------------------------------------------

@router.get("/samples", response_model=SamplesResponse, summary="List sample crops")
async def get_samples() -> SamplesResponse:
    """
    Return all stored sample crops with their static image URLs and true labels.
    Images are served from /static/samples/.
    """
    index = artifacts.load_samples_index()
    class_map = artifacts.load_class_map() or []
    slug_to_class = {c["slug"]: c for c in class_map}

    if index is None:
        return SamplesResponse(count=0, samples=[])

    sample_files: list[str] = index.get("samples", [])
    entries: list[SampleEntry] = []

    for fname in sorted(sample_files):
        stem = Path(fname).stem          # e.g. "sample_000"
        # Derive true label from filename if it encodes class slug:
        # e.g. "sample_000_stop.jpg" -> slug="stop"
        parts = stem.split("_", maxsplit=2)
        true_cls = None
        if len(parts) >= 3:
            slug_candidate = parts[2]
            true_cls = slug_to_class.get(slug_candidate)

        entries.append(SampleEntry(
            id=stem,
            image_url=f"/static/samples/{fname}",
            true_class_index=true_cls["index"] if true_cls else None,
            true_class_slug=true_cls["slug"] if true_cls else None,
            true_class_display_name=true_cls["display_name"] if true_cls else None,
        ))

    return SamplesResponse(count=len(entries), samples=entries)


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictResponse, summary="Predict from an uploaded image")
async def predict(
    file: Annotated[UploadFile, File(description="Image file (JPEG, PNG, WebP, BMP; max 8 MB)")],
    model: Annotated[str, Form()] = "both",
    top_k: Annotated[int, Form()] = settings.DEFAULT_TOP_K,
    gradcam: Annotated[bool, Form()] = True,
) -> PredictResponse:
    """
    Classify a cropped traffic sign image.

    - **file**: the image to classify (JPEG, PNG, WebP or BMP, max 8 MB)
    - **model**: which model(s) to use: `margnet`, `resnet50`, or `both`
    - **top_k**: how many top predictions to return (default 5)
    - **gradcam**: whether to return a Grad-CAM overlay (default true)
    """
    # --- size check ---
    raw = await file.read()
    if len(raw) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 8 MB limit")

    # --- content-type check ---
    ct = (file.content_type or "").lower()
    if ct not in settings.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ct}'. Accepted: JPEG, PNG, WebP, BMP",
        )

    model_keys = _resolve_models(model)
    top_k = max(1, min(top_k, 20))
    results = await _run_for_keys(model_keys, raw, top_k, gradcam)

    return PredictResponse(
        note=PREDICTION_NOTE,
        results=results,
        agreement=_check_agreement(results),
    )


# ---------------------------------------------------------------------------
# POST /predict/sample
# ---------------------------------------------------------------------------

@router.post(
    "/predict/sample",
    response_model=SamplePredictResponse,
    summary="Predict from a stored sample crop",
)
async def predict_sample(
    sample_id: Annotated[str, Form(description="e.g. 'sample_000'")],
    model: Annotated[str, Form()] = "both",
    top_k: Annotated[int, Form()] = settings.DEFAULT_TOP_K,
    gradcam: Annotated[bool, Form()] = True,
) -> SamplePredictResponse:
    """
    Classify one of the pre-stored sample crops and also return its true label.
    """
    # Locate the sample file
    sample_path = settings.SAMPLES_DIR / f"{sample_id}.jpg"
    if not sample_path.exists():
        # Try without extension in case it was passed with one
        stem = Path(sample_id).stem
        sample_path = settings.SAMPLES_DIR / f"{stem}.jpg"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample '{sample_id}' not found")

    raw = sample_path.read_bytes()
    model_keys = _resolve_models(model)
    top_k = max(1, min(top_k, 20))
    results = await _run_for_keys(model_keys, raw, top_k, gradcam)

    # Determine true label from filename if it encodes the class slug
    class_map = artifacts.load_class_map() or []
    slug_to_class = {c["slug"]: c for c in class_map}
    stem = Path(sample_id).stem
    parts = stem.split("_", maxsplit=2)
    true_cls = slug_to_class.get(parts[2]) if len(parts) >= 3 else None

    return SamplePredictResponse(
        note=PREDICTION_NOTE,
        results=results,
        agreement=_check_agreement(results),
        sample_id=stem,
        true_class_index=true_cls["index"] if true_cls else None,
        true_class_slug=true_cls["slug"] if true_cls else None,
        true_class_display_name=true_cls["display_name"] if true_cls else None,
    )
