"""Model loading logic for serve_f2k4b_quantized."""

from __future__ import annotations
import logging
import os
import json
from typing import Any

logger = logging.getLogger(__name__)

_BASE_REPO = "black-forest-labs/FLUX.2-klein-4B"
_GGUF_REPO = "unsloth/FLUX.2-klein-base-4B-GGUF"
_GGUF_FILE = "flux-2-klein-base-4b-Q8_0.gguf"
_INT8_REPO = "aydin99/FLUX.2-klein-4B-int8"

def _get_torch_dtype(compute_dtype: str):
    import torch
    return torch.bfloat16 if compute_dtype == "bfloat16" else torch.float32

def _load_text_encoder(device: str, compute_dtype: str):
    """Downloads and loads the Int8-quantized Qwen3 text encoder from aydin99."""
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from optimum.quanto import requantize
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen3ForCausalLM

    logger.info("Downloading Int8 text encoder from %s …", _INT8_REPO)
    config_path = hf_hub_download(repo_id=_INT8_REPO, filename="text_encoder/config.json")
    map_path = hf_hub_download(repo_id=_INT8_REPO, filename="text_encoder/quanto_qmap.json")
    weights_path = hf_hub_download(repo_id=_INT8_REPO, filename="text_encoder/model.safetensors")

    # Load structure
    te_config = AutoConfig.from_pretrained(config_path, trust_remote_code=True)
    with init_empty_weights():
        text_encoder = Qwen3ForCausalLM(te_config)

    # Requantize
    with open(map_path, "r") as fh:
        qmap = json.load(fh)
    state_dict = load_file(weights_path, device="cpu")
    
    logger.info("Requantising text encoder weights …")
    requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)

    dtype = _get_torch_dtype(compute_dtype)
    return text_encoder.to(device=device, dtype=dtype)

def _load_unsloth_transformer(device: str, compute_dtype: str):
    """Loads the Q8 GGUF Transformer."""
    from diffusers import GGUFQuantizationConfig, Flux2Transformer2DModel
    from huggingface_hub import hf_hub_download

    logger.info("Downloading GGUF transformer from %s …", _GGUF_REPO)
    ckpt_path = hf_hub_download(repo_id=_GGUF_REPO, filename=_GGUF_FILE)

    if device != "cpu":
        os.environ.setdefault("DIFFUSERS_GGUF_CUDA_KERNELS", "true")

    dtype = _get_torch_dtype(compute_dtype)
    transformer = Flux2Transformer2DModel.from_single_file(
        ckpt_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        config=_BASE_REPO,  # see https://github.com/huggingface/diffusers/issues/13001#issuecomment-3803406937
        subfolder="transformer",  # same issue
        torch_dtype=dtype,
    )
    return transformer

def _load_aydin99_transformer(device: str, compute_dtype: str):
    """Loads the Int8 Transformer using the custom wrapper."""
    from huggingface_hub import snapshot_download
    from .quanto_wrapper import QuantizedFlux2Transformer2DModel

    logger.info("Downloading Int8 transformer from %s …", _INT8_REPO)
    model_path = snapshot_download(_INT8_REPO)
    
    qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(model_path)
    # The pipeline expects the unwrapped standard diffusers model class
    return qtransformer._wrapped

def load_pipeline(cfg: dict[str, Any]):
    from diffusers import Flux2KleinPipeline

    profile = cfg["model"]["profile"]
    device = cfg["model"]["device"]
    compute_dtype = cfg["model"]["compute_dtype"]
    dtype = _get_torch_dtype(compute_dtype)

    logger.info("Loading Flux2KleinPipeline scaffold from %s …", _BASE_REPO)
    pipeline = Flux2KleinPipeline.from_pretrained(
        _BASE_REPO,
        transformer=None,
        text_encoder=None,
        torch_dtype=dtype,
    )

    # Always use the Int8 text encoder to keep VRAM strictly under the 24GB limit.
    pipeline.text_encoder = _load_text_encoder(device, compute_dtype)

    if profile == "unsloth-gguf":
        pipeline.transformer = _load_unsloth_transformer(device, compute_dtype)
    elif profile == "aydin99-int8":
        pipeline.transformer = _load_aydin99_transformer(device, compute_dtype)
    else:
        raise ValueError(f"Unsupported model profile: {profile!r}")

    logger.info("Moving pipeline to device %r …", device)
    pipeline = pipeline.to(device)
    return pipeline
