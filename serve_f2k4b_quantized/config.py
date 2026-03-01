"""Configuration loading and validation for serve_f2k4b_quantized."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_DEFAULTS: dict[str, Any] = {
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
        "static_dir": "./static",
    },
    "model": {
        "profile": "unsloth-gguf", # Default to GGUF
        "device": "cuda",
        "compute_dtype": "bfloat16",
    },
    "api": {
        "default_response_format": "b64_json",
    },
}

_VALID_PROFILES = {"unsloth-gguf", "aydin99-int8"}
_VALID_DEVICES = {"cuda", "cpu"}
_VALID_DTYPES = {"bfloat16", "float32"}
_VALID_RESPONSE_FORMATS = {"b64_json", "url"}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load configuration from *path* (YAML), merging with defaults.

    Parameters
    ----------
    path:
        Path to a YAML config file.  If *None* (or the file does not exist)
        the built-in defaults are returned as-is.

    Returns
    -------
    dict
        Fully-populated configuration dictionary.

    Raises
    ------
    ValueError
        If any configuration value is invalid.
    """
    import copy

    cfg = copy.deepcopy(_DEFAULTS)

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, user_cfg)

    _validate(cfg)
    _apply_auto_detect(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    profile = cfg["model"]["profile"]
    if profile not in _VALID_PROFILES:
        raise ValueError(
            f"model.profile must be one of {sorted(_VALID_PROFILES)}, got {profile!r}"
        )

    device = cfg["model"]["device"]
    if device not in _VALID_DEVICES:
        raise ValueError(
            f"model.device must be one of {sorted(_VALID_DEVICES)}, got {device!r}"
        )

    dtype = cfg["model"]["compute_dtype"]
    if dtype not in _VALID_DTYPES:
        raise ValueError(
            f"model.compute_dtype must be one of {sorted(_VALID_DTYPES)}, got {dtype!r}"
        )

    fmt = cfg["api"]["default_response_format"]
    if fmt not in _VALID_RESPONSE_FORMATS:
        raise ValueError(
            f"api.default_response_format must be one of "
            f"{sorted(_VALID_RESPONSE_FORMATS)}, got {fmt!r}"
        )


def _apply_auto_detect(cfg: dict) -> None:
    """Adjust device/dtype based on hardware availability."""
    if cfg["model"]["device"] == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                cfg["model"]["device"] = "cpu"
        except ImportError:
            cfg["model"]["device"] = "cpu"

    if cfg["model"]["device"] == "cpu":
        # bfloat16 is typically unsupported on CPU, fall back to float32
        if cfg["model"]["compute_dtype"] == "bfloat16":
            cfg["model"]["compute_dtype"] = "float32"
