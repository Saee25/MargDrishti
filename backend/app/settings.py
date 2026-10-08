"""
backend/app/settings.py
-----------------------
Central configuration for the FastAPI backend.
All paths are derived from the project root so the server can be started
from any working directory as long as the project root is on sys.path.
"""

from pathlib import Path
from functools import lru_cache


def _find_project_root() -> Path:
    """Walk up from this file until we find a directory that contains 'ml/'."""
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "ml").is_dir():
            return parent
    # Fall back to the grandparent of backend/app/
    return Path(__file__).resolve().parent.parent.parent


class Settings:
    """
    Immutable settings read once at import time.
    Change USE_GPU to True in this file if you want GPU inference.
    """

    # --- paths ---
    PROJECT_ROOT: Path = _find_project_root()
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "backend" / "artifacts"
    MODELS_DIR: Path = ARTIFACTS_DIR / "models"
    FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
    SAMPLES_DIR: Path = ARTIFACTS_DIR / "samples"

    # Specific artifact files
    MANIFEST_PATH: Path = ARTIFACTS_DIR / "manifest.json"
    CLASS_MAP_PATH: Path = ARTIFACTS_DIR / "class_map.json"
    METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"
    BENCHMARK_PATH: Path = ARTIFACTS_DIR / "benchmark.json"
    ABLATION_PATH: Path = ARTIFACTS_DIR / "ablation.json"
    DATASET_STATS_PATH: Path = ARTIFACTS_DIR / "dataset_stats.json"
    SAMPLES_INDEX_PATH: Path = SAMPLES_DIR / "samples.json"
    CUSTOM_LAYERS_PATH: Path = ARTIFACTS_DIR / "custom_v5_layers.json"
    RESNET_LAYERS_PATH: Path = ARTIFACTS_DIR / "resnet50_default_layers.json"

    MARGNET_PT: Path = MODELS_DIR / "margnet.pt"
    RESNET50_PT: Path = MODELS_DIR / "resnet50.pt"
    YOLO_PT: Path = MODELS_DIR / "yolo_nano.pt"

    # --- inference limits ---
    MAX_UPLOAD_BYTES: int = 8 * 1024 * 1024  # 8 MB
    ALLOWED_CONTENT_TYPES: frozenset = frozenset(
        {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    )
    DEFAULT_TOP_K: int = 5

    # --- device ---
    USE_GPU: bool = False  # Set True to use CUDA if available

    # --- CORS origins ---
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    return Settings()
