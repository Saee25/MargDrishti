"""
backend/app/services/inference.py
----------------------------------
CPU-bound prediction logic, designed to run inside a thread-pool executor so
the async event loop is never blocked.

Key design decisions
--------------------
1. torch.inference_mode() is used for the forward pass (no gradient tape, so
   it is faster and uses less memory than torch.no_grad()).
2. GradCAM re-enters gradient-enabled mode *only* for its own backward pass,
   then immediately removes its hooks.  This is safe because the model is
   acquired under an asyncio.Lock for the duration of the whole predict call.
3. Both the 64 px and 224 px versions of the input are returned as base64 PNG
   data-URLs so the frontend can show what each model actually sees.
4. Timing is measured around the forward pass alone (forward_ms) and around
   the whole preprocessing+forward pipeline (total_ms).
"""

from __future__ import annotations

import base64
import io
import logging
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from backend.app.schemas import SingleModelResult, TopKPrediction
from backend.app.services.model_registry import LoadedModel
from ml.explain.gradcam import GradCAM, apply_colormap_on_image  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image pre-processing helpers
# ---------------------------------------------------------------------------

def _apply_exif_orientation(img: Image.Image) -> Image.Image:
    """Rotate the image to match its EXIF orientation tag (handles phone photos)."""
    try:
        return ImageOps.exif_transpose(img)
    except Exception:
        return img


def open_and_validate_image(data: bytes) -> Image.Image:
    """
    Open raw bytes as a Pillow image.
    - Verifies it is actually an image (Pillow raises if not).
    - Applies EXIF orientation correction.
    - Converts to RGB (drops alpha, handles grayscale).
    Raises ValueError on anything that is not a valid image.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # raises if the file is truncated or not an image
    except Exception as exc:
        raise ValueError(f"Not a valid image: {exc}") from exc

    # Re-open after verify() (verify() moves the internal pointer)
    img = Image.open(io.BytesIO(data))
    img = _apply_exif_orientation(img)
    img = img.convert("RGB")
    return img


def _build_transform(img_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    """Val/test transform: resize then normalise.  No augmentation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def _pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a Pillow image as a base64 data URL."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _resize_preview(img: Image.Image, size: int) -> str:
    """Return a small base64 PNG of the image resized to size×size."""
    preview = img.resize((size, size), Image.Resampling.BILINEAR)
    return _pil_to_base64(preview)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_predict(
    entry: LoadedModel,
    img: Image.Image,
    top_k: int,
    class_map: list[dict],
    do_gradcam: bool,
) -> SingleModelResult:
    """
    Run one forward pass for a single model.  This function is CPU-bound and
    must be called from a thread pool (asyncio.run_in_executor).

    Args:
        entry:      The LoadedModel from the registry.
        img:        Pillow RGB image (already validated and EXIF-corrected).
        top_k:      How many top predictions to return.
        class_map:  List of {index, slug, display_name} dicts from class_map.json.
        do_gradcam: Whether to produce a Grad-CAM overlay.

    Returns:
        SingleModelResult Pydantic model.
    """
    model = entry.model
    transform = _build_transform(entry.img_size, entry.norm_mean, entry.norm_std)

    # Build small dict for fast class lookup
    idx_to_class: dict[int, dict] = {c["index"]: c for c in class_map}

    # ------------------------------------------------------------------ #
    # Preprocessing                                                        #
    # ------------------------------------------------------------------ #
    t_total_start = time.perf_counter()

    tensor = transform(img).unsqueeze(0)  # (1, 3, H, W)

    # ------------------------------------------------------------------ #
    # Forward pass – inference_mode disables gradient bookkeeping          #
    # ------------------------------------------------------------------ #
    t_fwd_start = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)          # (1, K)
    t_fwd_end = time.perf_counter()

    probs = F.softmax(logits, dim=1).squeeze(0)  # (K,)

    top_k_clamped = min(top_k, probs.shape[0])
    values, indices = torch.topk(probs, top_k_clamped)

    top_predictions: list[TopKPrediction] = []
    for rank, (prob_val, class_idx) in enumerate(zip(values.tolist(), indices.tolist())):
        cls = idx_to_class.get(class_idx, {"slug": str(class_idx), "display_name": str(class_idx)})
        top_predictions.append(TopKPrediction(
            rank=rank + 1,
            class_index=class_idx,
            slug=cls.get("slug", ""),
            display_name=cls.get("display_name", ""),
            probability=round(prob_val, 6),
        ))

    forward_ms = (t_fwd_end - t_fwd_start) * 1000.0

    # ------------------------------------------------------------------ #
    # Grad-CAM  (requires gradients, so we exit inference_mode)           #
    # ------------------------------------------------------------------ #
    gradcam_url: Optional[str] = None
    if do_gradcam:
        try:
            top_class_idx = indices[0].item()
            gradcam_url = _compute_gradcam(model, tensor, img, top_class_idx)
        except Exception as exc:
            logger.warning("GradCAM failed for model %s: %s", entry.key, exc)

    t_total_end = time.perf_counter()
    total_ms = (t_total_end - t_total_start) * 1000.0

    # ------------------------------------------------------------------ #
    # Input previews (what each model actually sees)                       #
    # ------------------------------------------------------------------ #
    preview_64 = _resize_preview(img, 64)
    preview_224 = _resize_preview(img, 224)

    return SingleModelResult(
        model_key=entry.key,
        top_k=top_predictions,
        forward_ms=round(forward_ms, 2),
        total_ms=round(total_ms, 2),
        gradcam_url=gradcam_url,
        input_preview_64=preview_64,
        input_preview_224=preview_224,
    )


def _compute_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    original_img: Image.Image,
    target_class: int,
) -> str:
    """
    Generate a Grad-CAM heatmap overlay and return it as a base64 PNG.

    We exit inference_mode (by not being inside the context manager) so that
    PyTorch can build a gradient tape for the backward pass.
    """
    # GradCAM registers forward/backward hooks and runs its own backward pass.
    cam_gen = GradCAM(model)
    try:
        cam = cam_gen.generate(tensor, target_class=target_class)
    finally:
        cam_gen.remove_hooks()

    overlay = apply_colormap_on_image(original_img, cam)
    return _pil_to_base64(overlay)
