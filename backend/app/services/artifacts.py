"""
backend/app/services/artifacts.py
-----------------------------------
Thin helpers that load the static JSON artifacts from backend/artifacts/.

Each function returns None (or an empty collection) when the file is missing
so every router can produce a meaningful "unavailable" JSON response rather
than a 500 error.  The frontend is designed to check the `available` flag and
show an appropriate empty state with the command that produces that artifact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from backend.app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _load_json(path: Path) -> Optional[Any]:
    """Return parsed JSON or None if the file is missing or malformed."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.debug("Artifact not found: %s", path)
        return None
    except Exception as exc:
        logger.warning("Failed to load artifact %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_manifest() -> Optional[dict]:
    return _load_json(settings.MANIFEST_PATH)


def load_class_map() -> Optional[list[dict]]:
    return _load_json(settings.CLASS_MAP_PATH)


def load_metrics() -> Optional[dict]:
    return _load_json(settings.METRICS_PATH)


def load_benchmark() -> Optional[list[dict]]:
    return _load_json(settings.BENCHMARK_PATH)


def load_ablation() -> Optional[list[dict]]:
    return _load_json(settings.ABLATION_PATH)


def load_dataset_stats() -> Optional[dict]:
    return _load_json(settings.DATASET_STATS_PATH)


def load_samples_index() -> Optional[dict]:
    return _load_json(settings.SAMPLES_INDEX_PATH)


def load_custom_layers() -> Optional[list[dict]]:
    return _load_json(settings.CUSTOM_LAYERS_PATH)


def load_resnet_layers() -> Optional[list[dict]]:
    return _load_json(settings.RESNET_LAYERS_PATH)


def load_run_history(run_name: str) -> Optional[list[dict]]:
    """
    Load the training history CSV for a given run and convert to a list of
    dicts.  Returns None if the file does not exist.

    The history CSV is written by the Trainer and lives at
    experiments/<run_name>/history.csv.
    """
    import csv
    csv_path = settings.PROJECT_ROOT / "experiments" / run_name / "history.csv"
    if not csv_path.exists():
        return None
    try:
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({k: _try_float(v) for k, v in row.items()})
        return rows
    except Exception as exc:
        logger.warning("Failed to load history for %s: %s", run_name, exc)
        return None


def load_confusion_matrix(run_name: str) -> Optional[dict]:
    """Load confusion_matrix.json from the experiment folder."""
    path = settings.PROJECT_ROOT / "experiments" / run_name / "confusion_matrix.json"
    return _load_json(path)


def load_per_class_report(run_name: str) -> Optional[dict]:
    """Load per_class_report.json from the experiment folder."""
    path = settings.PROJECT_ROOT / "experiments" / run_name / "per_class_report.json"
    return _load_json(path)


def load_comparison() -> Optional[dict]:
    """Load comparison_table.json from reports/comparison/."""
    path = settings.PROJECT_ROOT / "reports" / "comparison" / "comparison_table.json"
    return _load_json(path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_float(v: str) -> Any:
    """Convert a CSV cell to float if possible, else keep as string."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


# ---------------------------------------------------------------------------
# Run-name mapping (model key -> experiment folder name)
# ---------------------------------------------------------------------------

RUN_NAME_MAP: dict[str, str] = {
    "margnet": "custom_v5_margnet",
    "resnet50": "resnet50_finetune",
}
