"""Model loading logic for serve_f2k4b_quantized.

Supports the ``mixed-q8-int8`` profile which combines:
* A Q8_0 GGUF-quantized FLUX.2-klein Transformer (via Unsloth/Diffusers).
* An Int8-quantized Qwen3 Text Encoder (via optimum-quanto).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Hugging Face repo / file constants
_GGUF_REPO = "unsloth/FLUX.2-klein-base-4B-GGUF"
_GGUF_FILE = "flux-2-klein-base-4b-Q8_0.gguf"
_INT8_TE_REPO = "aydin99/FLUX.2-klein-4B-int8"

# Sub-files that make up the text encoder
_TE_CONFIG_FILE = "text_encoder/config.json"
_TE_MAP_FILE = "text_encoder/model.safetensors.index.json"


def _get_torch_dtype(compute_dtype: str):
    import torch

    return torch.bfloat16 if compute_dtype == "bfloat16" else torch.float32


def _load_transformer(device: str, compute_dtype: str):
    """Download and load the Q8_0 GGUF Transformer."""
    from huggingface_hub import hf_hub_download

    logger.info("Downloading GGUF transformer from %s …", _GGUF_REPO)
    ckpt_path = hf_hub_download(repo_id=_GGUF_REPO, filename=_GGUF_FILE)

    from diffusers import GGUFQuantizationConfig, Flux2Transformer2DModel

    logger.info("Loading GGUF transformer from %s …", ckpt_path)

    if device != "cpu":
        os.environ.setdefault("DIFFUSERS_GGUF_CUDA_KERNELS", "true")

    transformer = Flux2Transformer2DModel.from_single_file(
        ckpt_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=_get_torch_dtype(compute_dtype)),
        torch_dtype=_get_torch_dtype(compute_dtype),
    )
    return transformer


def _load_text_encoder(device: str, compute_dtype: str):
    """Download and load the Int8-quantized text encoder."""
    import json

    import torch
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from optimum.quanto import requantize
    from safetensors.torch import load_file
    from transformers import Qwen3ForCausalLM

    logger.info("Downloading Int8 text encoder config from %s …", _INT8_TE_REPO)
    config_path = hf_hub_download(repo_id=_INT8_TE_REPO, filename=_TE_CONFIG_FILE)
    map_path = hf_hub_download(repo_id=_INT8_TE_REPO, filename=_TE_MAP_FILE)

    with open(config_path) as fh:
        te_config_dict = json.load(fh)

    from transformers import AutoConfig

    te_config = AutoConfig.for_model(**te_config_dict)

    logger.info("Initialising empty text encoder skeleton …")
    with init_empty_weights():
        text_encoder = Qwen3ForCausalLM(te_config)

    # Load quantized weight map
    with open(map_path) as fh:
        index = json.load(fh)
    weight_map: dict[str, str] = index["weight_map"]

    # Collect unique shard filenames
    shard_files = sorted(set(weight_map.values()))
    state_dict: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        logger.info("Downloading text encoder shard %s …", shard)
        shard_path = hf_hub_download(
            repo_id=_INT8_TE_REPO,
            filename=f"text_encoder/{shard}",
        )
        state_dict.update(load_file(shard_path, device="cpu"))

    logger.info("Requantising text encoder weights …")
    requantize(text_encoder, state_dict, device=device)

    dtype = _get_torch_dtype(compute_dtype)
    text_encoder = text_encoder.to(dtype=dtype)
    return text_encoder


def load_pipeline(cfg: dict[str, Any]):
    """Build and return the Flux2KleinPipeline from *cfg*.

    Parameters
    ----------
    cfg:
        Parsed configuration dictionary as returned by
        :func:`serve_f2k4b_quantized.config.load_config`.
    """
    profile = cfg["model"]["profile"]
    device = cfg["model"]["device"]
    compute_dtype = cfg["model"]["compute_dtype"]

    if profile == "mixed-q8-int8":
        return _load_mixed_q8_int8(device, compute_dtype)
    else:
        raise ValueError(f"Unsupported model profile: {profile!r}")


def _load_mixed_q8_int8(device: str, compute_dtype: str):
    """Load the mixed Q8/Int8 pipeline."""
    from diffusers import Flux2KleinPipeline

    logger.info("Loading Flux2KleinPipeline scaffold (no weights) …")
    # Load the pipeline structure without transformer / text_encoder weights
    pipeline = Flux2KleinPipeline.from_pretrained(
        "aydin99/FLUX.2-klein-4B-int8",
        transformer=None,
        text_encoder=None,
        torch_dtype=_get_torch_dtype(compute_dtype),
    )

    transformer = _load_transformer(device, compute_dtype)
    text_encoder = _load_text_encoder(device, compute_dtype)

    pipeline.transformer = transformer
    pipeline.text_encoder = text_encoder

    logger.info("Moving pipeline to device %r …", device)
    pipeline = pipeline.to(device)
    return pipeline
