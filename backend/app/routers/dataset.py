"""
backend/app/routers/dataset.py
--------------------------------
GET /api/dataset/stats
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas import DatasetStatsResponse
from backend.app.services import artifacts

router = APIRouter()


@router.get(
    "/dataset/stats",
    response_model=DatasetStatsResponse,
    summary="Dataset statistics",
)
async def dataset_stats() -> DatasetStatsResponse:
    """
    Returns the key dataset statistics produced by scripts.compute_stats:
    - How many classes were kept (K) and which were dropped.
    - Training-set per-class sample counts.
    - Training-set mean and std (used for MargNet normalisation).
    - Split sizes (train / valid / test crop counts).
    - Crop-size histogram.
    """
    raw = artifacts.load_dataset_stats()
    if raw is None:
        return DatasetStatsResponse(available=False)

    return DatasetStatsResponse(
        available=True,
        k=raw.get("k"),
        mean=raw.get("mean"),
        std=raw.get("std"),
        class_counts_train=raw.get("class_counts_train"),
        splits=raw.get("splits"),
        crop_size_histogram=raw.get("crop_size_histogram"),
        kept_classes=raw.get("kept_classes"),
        dropped_classes=raw.get("dropped_classes"),
    )
