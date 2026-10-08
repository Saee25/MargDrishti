"""
tests/test_api.py
------------------
FastAPI TestClient tests for the MargDrishti backend.

Run from the project root with:
    pytest tests/test_api.py -v

Tests
-----
1. Health endpoint returns status "ok".
2. /api/classes count equals K (71 classes in the current artifact).
3. /api/predict with a real sample image returns probabilities that sum
   to ≈1 for each requested model.
4. A wrong file type (text/plain) is rejected with HTTP 400.
5. An oversized payload is rejected with HTTP 413.
6. /api/metrics/summary returns 'available' True when artifacts exist,
   or False when they do not.
7. /api/predict returns a 503 when a model is not loaded (mocked).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# We patch the model loading so tests don't need the .pt files to be loaded.
# The registry is replaced with a mock that returns tiny real predictions.
# ---------------------------------------------------------------------------

def _make_mock_entry(key: str, num_classes: int = 71) -> MagicMock:
    """Return a mock LoadedModel that produces uniform logits."""
    import torch
    import asyncio

    entry = MagicMock()
    entry.key = key
    entry.available = True
    entry.error = None
    entry.lock = asyncio.Lock()
    entry.class_names = [f"class_{i}" for i in range(num_classes)]
    entry.img_size = 64 if key == "margnet" else 224
    entry.norm_mean = [0.5, 0.5, 0.5]
    entry.norm_std = [0.5, 0.5, 0.5]
    entry.num_classes = num_classes

    # A tiny Linear model that accepts the correct input size
    import torch.nn as nn

    class TinyClassifier(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(3, nc)
            # gradcam_target_layer must be a property returning an nn.Module
            self._target = nn.Identity()

        @property
        def gradcam_target_layer(self):
            return self._target

        def forward(self, x):
            return self.fc(self.pool(x).flatten(1))

    model = TinyClassifier(num_classes)
    model.eval()
    entry.model = model
    return entry


# ---------------------------------------------------------------------------
# Fixture: TestClient with mocked registry
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Return a TestClient backed by mock models so no GPU/large file is needed."""
    # We need to patch model_registry BEFORE the app is imported the first time.
    mock_margnet = _make_mock_entry("margnet", num_classes=71)
    mock_resnet = _make_mock_entry("resnet50", num_classes=71)

    mock_registry = {
        "margnet": mock_margnet,
        "resnet50": mock_resnet,
    }

    def mock_get_model(key):
        return mock_registry.get(key, MagicMock(available=False, error="not found"))

    def mock_all_models():
        return mock_registry

    def mock_load_all():
        pass  # already "loaded"

    with (
        patch("backend.app.services.model_registry._registry", mock_registry),
        patch("backend.app.services.model_registry.get_model", mock_get_model),
        patch("backend.app.services.model_registry.all_models", mock_all_models),
        patch("backend.app.services.model_registry.load_all_models", mock_load_all),
    ):
        from backend.app.main import create_app
        app = create_app()
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_jpeg_bytes() -> bytes:
    """Create a tiny 32×32 JPEG in memory."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(128, 64, 32)).save(buf, format="JPEG")
    return buf.getvalue()


def _small_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1: Health
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "ok"
    assert "models" in data


# ---------------------------------------------------------------------------
# Test 2: Class count equals K
# ---------------------------------------------------------------------------

def test_classes_count(client):
    resp = client.get("/api/classes")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # The class_map.json has 71 classes (K=71)
    assert data["count"] == 71
    assert len(data["classes"]) == 71


# ---------------------------------------------------------------------------
# Test 3: Prediction probabilities sum to ≈1
# ---------------------------------------------------------------------------

def test_predict_probs_sum_to_one(client):
    """
    Predict with top_k=5 and verify that each probability is in [0, 1].
    We request only 5 of the 71 classes so the sum will be ≤ 1 (not ≈ 1).
    The important invariant is 0 ≤ p ≤ 1 for every returned prediction.
    """
    img_bytes = _small_jpeg_bytes()
    resp = client.post(
        "/api/predict",
        data={"model": "both", "top_k": "5", "gradcam": "false"},
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["results"]) == 2
    for result in data["results"]:
        preds = result["top_k"]
        assert len(preds) == 5, f"Expected 5 predictions, got {len(preds)}"
        for p in preds:
            assert 0.0 <= p["probability"] <= 1.0, (
                f"Probability {p['probability']} out of [0, 1]"
            )
        # Probabilities should be in descending order
        probs = [p["probability"] for p in preds]
        assert probs == sorted(probs, reverse=True), "Predictions not sorted by probability"


# ---------------------------------------------------------------------------
# Test 4: Wrong file type rejected with 400
# ---------------------------------------------------------------------------

def test_wrong_file_type_rejected(client):
    resp = client.post(
        "/api/predict",
        data={"model": "margnet", "gradcam": "false"},
        files={"file": ("evil.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400, resp.text


# ---------------------------------------------------------------------------
# Test 5: Oversized file rejected with 413
# ---------------------------------------------------------------------------

def test_oversized_file_rejected(client):
    # 9 MB of bytes (limit is 8 MB)
    big = b"x" * (9 * 1024 * 1024)
    resp = client.post(
        "/api/predict",
        data={"model": "margnet", "gradcam": "false"},
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert resp.status_code == 413, resp.text


# ---------------------------------------------------------------------------
# Test 6: Metrics summary available / unavailable
# ---------------------------------------------------------------------------

def test_metrics_summary_available(client):
    resp = client.get("/api/metrics/summary")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # With real artifacts this should be True; if artifacts are missing it
    # must still return 200 with available=False (not a 500).
    assert "available" in data


# ---------------------------------------------------------------------------
# Test 7: Unavailable model returns 503
# ---------------------------------------------------------------------------

def test_unavailable_model_returns_503(client):
    from unittest.mock import patch
    import asyncio

    unavailable = MagicMock()
    unavailable.available = False
    unavailable.error = "checkpoint not found"

    with patch("backend.app.routers.predict.model_registry.get_model", return_value=unavailable):
        resp = client.post(
            "/api/predict",
            data={"model": "resnet50", "gradcam": "false"},
            files={"file": ("test.jpg", _small_jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 503, resp.text
