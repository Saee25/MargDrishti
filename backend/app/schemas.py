"""
backend/app/schemas.py
----------------------
Pydantic response models used by every router.

Design notes
------------
- All fields have explicit type hints so the auto-generated OpenAPI docs
  are self-explanatory.
- Optional fields carry None defaults so the API degrades gracefully when
  an artifact is missing (frontend can check for None and show an empty state).
- The `note` field on every prediction response carries the honest-wording
  disclaimer required by the project spec.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class ModelLoadStatus(BaseModel):
    available: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    device: str
    models: dict[str, ModelLoadStatus]
    manifest_generated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

class ClassEntry(BaseModel):
    index: int
    slug: str
    display_name: str


class ClassesResponse(BaseModel):
    count: int
    classes: list[ClassEntry]


# ---------------------------------------------------------------------------
# Models info
# ---------------------------------------------------------------------------

class LayerDescription(BaseModel):
    name: str
    type: str
    output_shape: str
    parameter_count: int
    description: str


class ModelInfo(BaseModel):
    key: str
    display_name: str
    available: bool
    input_size: int                    # pixels (square)
    parameters: Optional[int] = None
    size_mb: Optional[float] = None
    test_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    cpu_latency_ms: Optional[float] = None  # forward-pass median from benchmark
    pretrained: bool
    layers: list[LayerDescription] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TopKPrediction(BaseModel):
    rank: int
    class_index: int
    slug: str
    display_name: str
    probability: float


class SingleModelResult(BaseModel):
    model_key: str
    top_k: list[TopKPrediction]
    forward_ms: float   # forward pass only
    total_ms: float     # including preprocessing
    gradcam_url: Optional[str] = None   # base64 PNG data URL
    input_preview_64: Optional[str] = None   # 64 px resized input, base64
    input_preview_224: Optional[str] = None  # 224 px resized input, base64


PREDICTION_NOTE = (
    "This model classifies a cropped traffic sign. "
    "Full road photos give unreliable results."
)


class PredictResponse(BaseModel):
    note: str = PREDICTION_NOTE
    results: list[SingleModelResult]
    agreement: Optional[bool] = None   # True if both models agree on top-1


class SamplePredictResponse(PredictResponse):
    sample_id: str
    true_class_index: Optional[int] = None
    true_class_slug: Optional[str] = None
    true_class_display_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

class SampleEntry(BaseModel):
    id: str                        # e.g. "sample_000"
    image_url: str                 # relative URL served by static mount
    true_class_index: Optional[int] = None
    true_class_slug: Optional[str] = None
    true_class_display_name: Optional[str] = None


class SamplesResponse(BaseModel):
    count: int
    samples: list[SampleEntry]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MetricValues(BaseModel):
    accuracy: Optional[float] = None
    top3_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    weighted_f1: Optional[float] = None


class ModelSummaryRow(BaseModel):
    key: str
    display_name: str
    parameters: Optional[int] = None
    size_mb: Optional[float] = None
    test_accuracy: Optional[float] = None
    top3_accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    cpu_latency_ms: Optional[float] = None
    training_time_s: Optional[float] = None


class SummaryResponse(BaseModel):
    rows: list[ModelSummaryRow]
    ratios: dict[str, Any] = Field(default_factory=dict)
    available: bool


class HistoryPoint(BaseModel):
    epoch: int
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    val_macro_f1: Optional[float] = None
    lr: Optional[float] = None


class HistoryResponse(BaseModel):
    model_key: str
    history: list[HistoryPoint]
    available: bool


class ConfusionResponse(BaseModel):
    model_key: str
    matrix: Optional[list[list[int]]] = None
    class_names: Optional[list[str]] = None
    available: bool


class PerClassEntry(BaseModel):
    class_index: int
    slug: str
    display_name: str
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    support: Optional[int] = None


class PerClassResponse(BaseModel):
    model_key: str
    per_class: list[PerClassEntry]
    available: bool


class ComparisonResponse(BaseModel):
    available: bool
    data: Optional[dict[str, Any]] = None


class AblationRow(BaseModel):
    step: str
    description: str
    parameters: Optional[int] = None
    best_epoch: Optional[int] = None
    test_acc: Optional[float] = None
    test_macro_f1: Optional[float] = None
    test_top3_acc: Optional[float] = None
    overfit_gap: Optional[float] = None
    train_time_min: Optional[float] = None
    acc_change_pct_pt: Optional[float] = None
    f1_change_pct_pt: Optional[float] = None


class AblationResponse(BaseModel):
    available: bool
    rows: list[AblationRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DatasetStatsResponse(BaseModel):
    available: bool
    k: Optional[int] = None
    mean: Optional[list[float]] = None
    std: Optional[list[float]] = None
    class_counts_train: Optional[dict[str, int]] = None
    splits: Optional[dict[str, int]] = None
    crop_size_histogram: Optional[dict[str, int]] = None
    kept_classes: Optional[list[str]] = None
    dropped_classes: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

class FigureEntry(BaseModel):
    filename: str
    url: str
    title: str


class FiguresResponse(BaseModel):
    count: int
    figures: list[FigureEntry]
