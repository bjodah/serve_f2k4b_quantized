"""Integration tests for serve_f2k4b_quantized.

These tests require actual model weights and optionally a CUDA GPU.  They are
skipped automatically when the ``SKIP_HARDWARE_TESTS`` environment variable is
set to any non-empty value (which is the default on the CI server).

Run the full integration suite (on real hardware) with:

    SKIP_HARDWARE_TESTS= pytest tests/test_integration.py -v

Or run *all* tests including the CI-safe subset in one go:

    pytest tests/ -v
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Marker: skip when running in CI / lightweight environment
# ---------------------------------------------------------------------------

_SKIP_HW = os.environ.get("SKIP_HARDWARE_TESTS", "1")
hw_skip = pytest.mark.skipif(
    bool(_SKIP_HW),
    reason=(
        "Hardware integration tests skipped.  Unset SKIP_HARDWARE_TESTS "
        "and ensure model weights are available to run these tests."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_config(tmp_path, response_format="b64_json"):
    import yaml

    cfg = {
        "server": {"host": "127.0.0.1", "port": 8765, "static_dir": str(tmp_path / "static")},
        "model": {"profile": "mixed-q8-int8", "device": "cuda", "compute_dtype": "bfloat16"},
        "api": {"default_response_format": response_format},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg))
    return config_path, cfg


# ---------------------------------------------------------------------------
# Integration: b64_json response
# ---------------------------------------------------------------------------

@hw_skip
def test_integration_b64_response(tmp_path):
    """Full pipeline: load model, run inference, verify base64 PNG is returned."""
    import base64
    import io

    from fastapi.testclient import TestClient
    from PIL import Image

    from serve_f2k4b_quantized import server as server_mod
    from serve_f2k4b_quantized.config import load_config
    from serve_f2k4b_quantized.loader import load_pipeline

    config_path, raw_cfg = _make_test_config(tmp_path)
    cfg = load_config(config_path)

    pipeline = load_pipeline(cfg)
    server_mod._pipeline = pipeline
    server_mod._cfg = cfg

    client = TestClient(server_mod.app)
    payload = {
        "prompt": "a red circle",
        "size": "256x256",
        "num_inference_steps": 1,
        "response_format": "b64_json",
    }
    response = client.post("/v1/images/generations", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert "data" in body
    assert len(body["data"]) == 1
    b64_str = body["data"][0]["b64_json"]
    assert b64_str, "Expected non-empty b64_json"

    # Verify it decodes to a valid PNG
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (256, 256)


# ---------------------------------------------------------------------------
# Integration: url response
# ---------------------------------------------------------------------------

@hw_skip
def test_integration_url_response(tmp_path):
    """Pipeline saves image to disk and returns a URL."""
    from pathlib import Path

    from fastapi.testclient import TestClient

    from serve_f2k4b_quantized import server as server_mod
    from serve_f2k4b_quantized.config import load_config
    from serve_f2k4b_quantized.loader import load_pipeline

    config_path, raw_cfg = _make_test_config(tmp_path, response_format="url")
    cfg = load_config(config_path)

    pipeline = load_pipeline(cfg)
    server_mod._pipeline = pipeline
    server_mod._cfg = cfg

    static_dir = Path(cfg["server"]["static_dir"])
    static_dir.mkdir(parents=True, exist_ok=True)
    # Re-mount static directory for this test
    from fastapi.staticfiles import StaticFiles

    # Remove existing /static mount if present
    routes_to_keep = [r for r in server_mod.app.routes if getattr(r, "path", None) != "/static"]
    server_mod.app.routes = routes_to_keep
    server_mod.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static_test")

    client = TestClient(server_mod.app)
    payload = {
        "prompt": "a blue square",
        "size": "256x256",
        "num_inference_steps": 1,
        "response_format": "url",
    }
    response = client.post("/v1/images/generations", json=payload)
    assert response.status_code == 200, response.text

    body = response.json()
    assert len(body["data"]) == 1
    url = body["data"][0]["url"]
    assert url.startswith("http://"), f"Expected HTTP URL, got: {url}"

    # Verify the file actually exists on disk
    filename = url.split("/static/")[-1]
    assert (static_dir / filename).exists(), f"Image file not found: {static_dir / filename}"
