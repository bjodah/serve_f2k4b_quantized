"""Shared pytest fixtures for serve_f2k4b_quantized tests."""

from __future__ import annotations

import pytest


@pytest.fixture()
def default_config():
    """Return the default (no-file) configuration with CPU forced to avoid hardware deps."""
    from serve_f2k4b_quantized.config import load_config

    cfg = load_config(None)
    # Override device to cpu so tests work without CUDA
    cfg["model"]["device"] = "cpu"
    cfg["model"]["compute_dtype"] = "float32"
    return cfg
