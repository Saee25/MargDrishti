"""
backend/app/services/model_registry.py
----------------------------------------
Loads model checkpoints at startup and keeps them in memory.

Why a registry?
  The two models (MargNet and ResNet50) are large objects that take several
  seconds to load.  Loading them once at startup and keeping them in eval mode
  means every HTTP request pays only the forward-pass cost.

Graceful degradation
  If a checkpoint file is missing or corrupt, that model's status is set to
  unavailable (available=False) and an error string is stored.  The rest of
  the API keeps working.

Thread safety
  Each model has its own asyncio.Lock.  Inference code acquires the lock
  before doing a forward pass, so two concurrent requests never share a model
  in an undefined state (e.g., one request re-enabling gradients while another
  runs inference_mode).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class LoadedModel:
    """Container for one loaded model and its metadata from the checkpoint."""

    key: str                        # "margnet" or "resnet50"
    model: Optional[torch.nn.Module] = None
    available: bool = False
    error: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Metadata stored inside the checkpoint
    class_names: list[str] = field(default_factory=list)
    img_size: int = 64
    norm_mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    norm_std: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    num_classes: int = 0
    model_name: str = ""
    variant: str = ""


# Module-level registry; populated by load_all_models() called from lifespan.
_registry: dict[str, LoadedModel] = {}


def get_model(key: str) -> LoadedModel:
    """Return the LoadedModel for the given key.  Always succeeds (returns unavailable if missing)."""
    if key not in _registry:
        return LoadedModel(key=key, available=False, error="Model not in registry")
    return _registry[key]


def all_models() -> dict[str, LoadedModel]:
    return _registry


def _choose_device() -> torch.device:
    if settings.USE_GPU and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_single(key: str, pt_path: Path, device: torch.device) -> LoadedModel:
    """
    Load one checkpoint, rebuild the model, put it in eval mode, run one
    warm-up pass.  Returns a LoadedModel with available=True on success or
    available=False with an error string on failure.
    """
    entry = LoadedModel(key=key)

    if not pt_path.exists():
        entry.error = f"Checkpoint not found: {pt_path}"
        logger.warning("Model %s unavailable: %s", key, entry.error)
        return entry

    try:
        # load_checkpoint is the Task-5 helper that rebuilds the model from the
        # checkpoint dict (which embeds model_name, variant, num_classes, etc.)
        from ml.engine.trainer import load_checkpoint  # type: ignore

        model, ckpt = load_checkpoint(str(pt_path), device=str(device))
        model.eval()

        entry.model = model
        entry.class_names = ckpt.get("class_names", [])
        entry.img_size = ckpt.get("img_size", 64)
        entry.norm_mean = ckpt.get("norm_mean", [0.5, 0.5, 0.5])
        entry.norm_std = ckpt.get("norm_std", [0.5, 0.5, 0.5])
        entry.num_classes = ckpt.get("num_classes", len(entry.class_names))
        entry.model_name = ckpt.get("model_name", "")
        entry.variant = ckpt.get("variant", "")

        # Warm-up: one dummy forward pass so the first real request isn't slow.
        _warm_up(model, entry.img_size, device)

        entry.available = True
        logger.info("Model %s loaded successfully (device=%s, classes=%d)", key, device, entry.num_classes)

    except Exception as exc:
        entry.error = str(exc)
        logger.exception("Failed to load model %s: %s", key, exc)

    return entry


def _warm_up(model: torch.nn.Module, img_size: int, device: torch.device) -> None:
    """Run one dummy forward pass to initialise any lazy layers."""
    dummy = torch.zeros(1, 3, img_size, img_size, device=device)
    with torch.inference_mode():
        model(dummy)


def load_all_models() -> None:
    """
    Called once from the FastAPI lifespan context.
    Loads both checkpoints and populates _registry.
    """
    device = _choose_device()
    logger.info("Loading models on device: %s", device)

    _registry["margnet"] = _load_single(
        key="margnet",
        pt_path=settings.MARGNET_PT,
        device=device,
    )
    _registry["resnet50"] = _load_single(
        key="resnet50",
        pt_path=settings.RESNET50_PT,
        device=device,
    )
