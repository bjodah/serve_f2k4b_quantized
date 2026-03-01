"""Unit tests for configuration loading — no CUDA, no model downloads required.

These tests run on any machine (CI-safe subset).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# load_config() — default values
# ---------------------------------------------------------------------------

def test_default_config_structure():
    from serve_f2k4b_quantized.config import load_config

    cfg = load_config(None)
    assert "server" in cfg
    assert "model" in cfg
    assert "api" in cfg


def test_default_server_keys():
    from serve_f2k4b_quantized.config import load_config

    cfg = load_config(None)
    assert cfg["server"]["port"] == 8000
    assert cfg["server"]["static_dir"] == "./static"


def test_default_api_format():
    from serve_f2k4b_quantized.config import load_config

    cfg = load_config(None)
    assert cfg["api"]["default_response_format"] == "b64_json"


# ---------------------------------------------------------------------------
# load_config() — YAML file merging
# ---------------------------------------------------------------------------

def test_yaml_override_port():
    from serve_f2k4b_quantized.config import load_config

    data = {"server": {"port": 9999}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(data, fh)
        path = fh.name

    cfg = load_config(path)
    assert cfg["server"]["port"] == 9999
    # Unset keys should still carry their default
    assert cfg["server"]["host"] == "0.0.0.0"


def test_yaml_override_profile():
    from serve_f2k4b_quantized.config import load_config

    data = {"model": {"profile": "aydin99-int8", "device": "cpu"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(data, fh)
        path = fh.name

    cfg = load_config(path)
    assert cfg["model"]["profile"] == "aydin99-int8"


def test_missing_file_raises():
    from serve_f2k4b_quantized.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_profile_raises():
    from serve_f2k4b_quantized.config import load_config

    data = {"model": {"profile": "invalid-profile"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(data, fh)
        path = fh.name

    with pytest.raises(ValueError, match="model.profile"):
        load_config(path)


def test_invalid_device_raises():
    from serve_f2k4b_quantized.config import load_config

    data = {"model": {"device": "tpu"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(data, fh)
        path = fh.name

    with pytest.raises(ValueError, match="model.device"):
        load_config(path)


def test_invalid_response_format_raises():
    from serve_f2k4b_quantized.config import load_config

    data = {"api": {"default_response_format": "gif"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(data, fh)
        path = fh.name

    with pytest.raises(ValueError, match="api.default_response_format"):
        load_config(path)


# ---------------------------------------------------------------------------
# Auto-detect: CPU fall-back
# ---------------------------------------------------------------------------

def test_cpu_forces_float32(monkeypatch):
    """When device is forced to CPU, compute_dtype must become float32."""
    import serve_f2k4b_quantized.config as cfg_mod

    # Make torch.cuda.is_available() return False
    class _FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return False

    monkeypatch.setattr(cfg_mod, "_apply_auto_detect", cfg_mod._apply_auto_detect)

    import tempfile

    data = {"model": {"device": "cuda", "compute_dtype": "bfloat16"}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
        yaml.dump(data, fh)
        path = fh.name

    # Patch torch so CUDA appears unavailable
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_cuda = types.ModuleType("torch.cuda")
    fake_cuda.is_available = lambda: False
    fake_torch.cuda = fake_cuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    cfg = cfg_mod.load_config(path)
    assert cfg["model"]["device"] == "cpu"
    assert cfg["model"]["compute_dtype"] == "float32"


# ---------------------------------------------------------------------------
# Server helpers — parse_size
# ---------------------------------------------------------------------------

def test_parse_size_valid():
    from serve_f2k4b_quantized.server import _parse_size

    w, h = _parse_size("1024x768")
    assert w == 1024
    assert h == 768


def test_parse_size_square():
    from serve_f2k4b_quantized.server import _parse_size

    w, h = _parse_size("512x512")
    assert w == h == 512


def test_parse_size_invalid():
    from serve_f2k4b_quantized.server import _parse_size

    with pytest.raises(ValueError):
        _parse_size("1024")


# ---------------------------------------------------------------------------
# Server helpers — _image_to_b64
# ---------------------------------------------------------------------------

def test_image_to_b64_roundtrip():
    import base64
    import io

    from PIL import Image

    from serve_f2k4b_quantized.server import _image_to_b64

    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    encoded = _image_to_b64(img)
    decoded_bytes = base64.b64decode(encoded)
    recovered = Image.open(io.BytesIO(decoded_bytes))
    assert recovered.size == (8, 8)
