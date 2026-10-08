"""
backend/app/routers/metrics.py
--------------------------------
All metrics endpoints under /api/metrics/...

GET /api/metrics/summary
GET /api/metrics/{model}/history
GET /api/metrics/{model}/confusion
GET /api/metrics/{model}/per-class
GET /api/metrics/comparison
GET /api/metrics/ablation
GET /api/models
GET /api/figures
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException

from backend.app.schemas import (
    AblationResponse,
    AblationRow,
    ComparisonResponse,
    ConfusionResponse,
    FigureEntry,
    FiguresResponse,
    HistoryPoint,
    HistoryResponse,
    LayerDescription,
    ModelInfo,
    ModelSummaryRow,
    ModelsResponse,
    PerClassEntry,
    PerClassResponse,
    SummaryResponse,
)
from backend.app.services import artifacts, model_registry
from backend.app.settings import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

# Map API model key -> experiment run folder name
_RUN_MAP: dict[str, str] = {
    "margnet": "custom_v5_margnet",
    "resnet50": "resnet50_finetune",
}

# Map benchmark JSON 'model' field -> API key
_BENCH_NAME_MAP: dict[str, str] = {
    "MargNet": "margnet",
    "ResNet50 fine-tuned": "resnet50",
    "ResNet50 frozen": "resnet50_frozen",
}

ModelKey = Literal["margnet", "resnet50"]


def _bench_latency(key: str) -> float | None:
    bench = artifacts.load_benchmark()
    if not bench:
        return None
    for row in bench:
        if _BENCH_NAME_MAP.get(row.get("model", "")) == key:
            return row.get("fwd_bs1_median_ms")
    return None


def _layers_for_key(key: str) -> list[LayerDescription]:
    if key == "margnet":
        raw = artifacts.load_custom_layers()
    else:
        raw = artifacts.load_resnet_layers()
    if not raw:
        return []
    return [LayerDescription(**d) for d in raw]


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get("/models", response_model=ModelsResponse, summary="Per-model info and metrics")
async def get_models() -> ModelsResponse:
    """
    Returns metadata for each model: availability, architecture summary,
    test metrics, CPU latency, and the layer-by-layer description.
    """
    metrics = artifacts.load_metrics() or {}
    infos: list[ModelInfo] = []

    for key, display, pretrained, input_size in [
        ("margnet", "MargNet (Custom CNN)", False, 64),
        ("resnet50", "ResNet50 Fine-tuned", True, 224),
    ]:
        reg = model_registry.get_model(key)
        m = metrics.get(key) or metrics.get("resnet50_finetune" if key == "resnet50" else key, {})
        test_m = m.get("test_metrics", {})

        infos.append(ModelInfo(
            key=key,
            display_name=display,
            available=reg.available,
            input_size=input_size,
            parameters=m.get("parameters"),
            size_mb=m.get("size_mb"),
            test_accuracy=test_m.get("accuracy"),
            macro_f1=test_m.get("macro_f1"),
            cpu_latency_ms=_bench_latency(key),
            pretrained=pretrained,
            layers=_layers_for_key(key),
        ))

    return ModelsResponse(models=infos)


# ---------------------------------------------------------------------------
# GET /metrics/summary
# ---------------------------------------------------------------------------

@router.get("/metrics/summary", response_model=SummaryResponse, summary="Headline comparison table")
async def metrics_summary() -> SummaryResponse:
    """
    The headline side-by-side comparison of all three models (MargNet,
    ResNet50 frozen, ResNet50 fine-tuned) plus the ratio numbers.
    """
    metrics = artifacts.load_metrics()
    if not metrics:
        return SummaryResponse(rows=[], ratios={}, available=False)

    bench = artifacts.load_benchmark() or []
    bench_map: dict[str, dict] = {}
    for row in bench:
        k = _BENCH_NAME_MAP.get(row.get("model", ""))
        if k:
            bench_map[k] = row

    rows: list[ModelSummaryRow] = []
    for key, display in [
        ("margnet", "MargNet"),
        ("resnet50_frozen", "ResNet50 Frozen"),
        ("resnet50_finetune", "ResNet50 Fine-tuned"),
    ]:
        m = metrics.get(key, {})
        test_m = m.get("test_metrics", {})
        bench_key = "resnet50" if "finetune" in key else ("resnet50_frozen" if "frozen" in key else key)
        brow = bench_map.get(bench_key, {})

        rows.append(ModelSummaryRow(
            key=key,
            display_name=display,
            parameters=m.get("parameters"),
            size_mb=m.get("size_mb"),
            test_accuracy=test_m.get("accuracy"),
            top3_accuracy=test_m.get("top3_accuracy"),
            macro_f1=test_m.get("macro_f1"),
            cpu_latency_ms=brow.get("fwd_bs1_median_ms"),
            training_time_s=m.get("total_time_s"),
        ))

    # Compute ratios: MargNet vs ResNet50 fine-tuned
    mn = metrics.get("margnet", {})
    rn = metrics.get("resnet50_finetune", {})
    ratios: dict = {}
    if mn and rn:
        mn_params = mn.get("parameters", 0) or 0
        rn_params = rn.get("parameters", 0) or 1
        mn_size = mn.get("size_mb", 0) or 0
        rn_size = rn.get("size_mb", 0) or 1
        bm = bench_map.get("margnet", {})
        br = bench_map.get("resnet50", {})
        mn_lat = bm.get("fwd_bs1_median_ms", 0) or 0
        rn_lat = br.get("fwd_bs1_median_ms", 0) or 1
        ratios = {
            "parameter_ratio": round(rn_params / mn_params, 1) if mn_params else None,
            "size_ratio_mb": round(rn_size / mn_size, 1) if mn_size else None,
            "latency_ratio": round(rn_lat / mn_lat, 1) if mn_lat else None,
            "acc_gap_pct_pt": round(
                (rn.get("test_metrics", {}).get("accuracy", 0) -
                 mn.get("test_metrics", {}).get("accuracy", 0)) * 100, 2
            ),
        }

    return SummaryResponse(rows=rows, ratios=ratios, available=True)


# ---------------------------------------------------------------------------
# GET /metrics/{model}/history
# ---------------------------------------------------------------------------

@router.get(
    "/metrics/{model}/history",
    response_model=HistoryResponse,
    summary="Training history for a model",
)
async def metrics_history(model: ModelKey) -> HistoryResponse:
    """Return epoch-by-epoch training/validation loss and accuracy."""
    run = _RUN_MAP.get(model)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {model}")

    raw = artifacts.load_run_history(run)
    if raw is None:
        return HistoryResponse(model_key=model, history=[], available=False)

    history: list[HistoryPoint] = []
    for row in raw:
        history.append(HistoryPoint(
            epoch=int(row.get("epoch", 0)),
            train_loss=row.get("train_loss"),
            train_acc=row.get("train_acc"),
            val_loss=row.get("val_loss"),
            val_acc=row.get("val_acc"),
            val_macro_f1=row.get("val_macro_f1"),
            lr=row.get("lr"),
        ))

    return HistoryResponse(model_key=model, history=history, available=True)


# ---------------------------------------------------------------------------
# GET /metrics/{model}/confusion
# ---------------------------------------------------------------------------

@router.get(
    "/metrics/{model}/confusion",
    response_model=ConfusionResponse,
    summary="Confusion matrix for a model",
)
async def metrics_confusion(model: ModelKey) -> ConfusionResponse:
    """Return the confusion matrix as a 2-D list of ints."""
    run = _RUN_MAP.get(model)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {model}")

    raw = artifacts.load_confusion_matrix(run)
    class_map = artifacts.load_class_map() or []
    class_names = [c["display_name"] for c in class_map]

    if raw is None:
        return ConfusionResponse(model_key=model, available=False)

    return ConfusionResponse(
        model_key=model,
        matrix=raw.get("matrix") if isinstance(raw, dict) else raw,
        class_names=class_names,
        available=True,
    )


# ---------------------------------------------------------------------------
# GET /metrics/{model}/per-class
# ---------------------------------------------------------------------------

@router.get(
    "/metrics/{model}/per-class",
    response_model=PerClassResponse,
    summary="Per-class precision/recall/F1 for a model",
)
async def metrics_per_class(model: ModelKey) -> PerClassResponse:
    """Return per-class precision, recall, F1 and support from the test evaluation."""
    run = _RUN_MAP.get(model)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {model}")

    raw = artifacts.load_per_class_report(run)
    class_map = artifacts.load_class_map() or []
    idx_to_cls = {c["index"]: c for c in class_map}

    if raw is None:
        return PerClassResponse(model_key=model, per_class=[], available=False)

    per_class: list[PerClassEntry] = []
    # per_class_report.json format: { "0": {"precision": .., "recall": .., "f1-score": .., "support": ..}, ... }
    for idx_str, vals in raw.items():
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        cls = idx_to_cls.get(idx, {})
        per_class.append(PerClassEntry(
            class_index=idx,
            slug=cls.get("slug", str(idx)),
            display_name=cls.get("display_name", str(idx)),
            precision=vals.get("precision"),
            recall=vals.get("recall"),
            f1=vals.get("f1-score"),
            support=vals.get("support"),
        ))

    per_class.sort(key=lambda e: e.class_index)
    return PerClassResponse(model_key=model, per_class=per_class, available=True)


# ---------------------------------------------------------------------------
# GET /metrics/comparison
# ---------------------------------------------------------------------------

@router.get(
    "/metrics/comparison",
    response_model=ComparisonResponse,
    summary="Agreement, McNemar, per-class differences, calibration",
)
async def metrics_comparison() -> ComparisonResponse:
    """Return the full comparison JSON produced by scripts.compare."""
    data = artifacts.load_comparison()
    if data is None:
        return ComparisonResponse(available=False)
    return ComparisonResponse(available=True, data=data)


# ---------------------------------------------------------------------------
# GET /metrics/ablation
# ---------------------------------------------------------------------------

@router.get(
    "/metrics/ablation",
    response_model=AblationResponse,
    summary="Ablation study table and curves",
)
async def metrics_ablation() -> AblationResponse:
    """Return the step-by-step ablation table (V1 through V5 / MargNet)."""
    raw = artifacts.load_ablation()
    if raw is None:
        return AblationResponse(available=False, rows=[])

    rows = [
        AblationRow(
            step=r.get("step", ""),
            description=r.get("description", ""),
            parameters=r.get("parameters"),
            best_epoch=r.get("best_epoch"),
            test_acc=r.get("test_acc"),
            test_macro_f1=r.get("test_macro_f1"),
            test_top3_acc=r.get("test_top3_acc"),
            overfit_gap=r.get("overfit_gap"),
            train_time_min=r.get("train_time_min"),
            acc_change_pct_pt=r.get("acc_change_pct_pt"),
            f1_change_pct_pt=r.get("f1_change_pct_pt"),
        )
        for r in raw
    ]
    return AblationResponse(available=True, rows=rows)


# ---------------------------------------------------------------------------
# GET /figures
# ---------------------------------------------------------------------------

@router.get("/figures", response_model=FiguresResponse, summary="List available figure files")
async def get_figures() -> FiguresResponse:
    """Return all PNG figures stored in backend/artifacts/figures/."""
    figs_dir = settings.FIGURES_DIR
    if not figs_dir.exists():
        return FiguresResponse(count=0, figures=[])

    # Build human-readable titles from filenames
    def _title(name: str) -> str:
        stem = name.removesuffix(".png")
        return stem.replace("_", " ").replace("-", " ").title()

    entries = [
        FigureEntry(
            filename=p.name,
            url=f"/static/figures/{p.name}",
            title=_title(p.name),
        )
        for p in sorted(figs_dir.glob("*.png"))
    ]
    return FiguresResponse(count=len(entries), figures=entries)
