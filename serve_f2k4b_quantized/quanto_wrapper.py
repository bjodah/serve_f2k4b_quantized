"""Optimum-quanto wrapper for FLUX.2-klein Transformer."""
import json
import os
from pathlib import Path
from typing import Any, List, Optional, Union

from diffusers.models.model_loading_utils import load_state_dict
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    _get_checkpoint_shard_files,
    is_accelerate_available,
)
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from huggingface_hub import ModelHubMixin, snapshot_download
from optimum.quanto import freeze, qtype, quantization_map, quantize, requantize, Optimizer
from optimum.quanto.models.shared_dict import ShardedStateDict

class QuantizedDiffusersModel(ModelHubMixin):
    BASE_NAME = "quanto"
    base_class = None

    def __init__(self, model: ModelMixin):
        if not isinstance(model, ModelMixin) or len(quantization_map(model)) == 0:
            raise ValueError("The source model must be a quantized diffusers model.")
        self._wrapped = model

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = self.__dict__["_wrapped"]
            return getattr(wrapped, name)

    def forward(self, *args, **kwargs):
        return self._wrapped.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self._wrapped.forward(*args, **kwargs)

    @staticmethod
    def _qmap_name():
        return f"{QuantizedDiffusersModel.BASE_NAME}_qmap.json"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        if cls.base_class is None:
            raise ValueError("The `base_class` attribute needs to be configured.")

        if not is_accelerate_available():
            raise ValueError("Reloading a quantized diffusers model requires the accelerate library.")
        from accelerate import init_empty_weights

        working_dir = (
            pretrained_model_name_or_path if os.path.isdir(pretrained_model_name_or_path)
            else snapshot_download(pretrained_model_name_or_path, **kwargs)
        )

        qmap_path = os.path.join(working_dir, cls._qmap_name())
        model_config_path = os.path.join(working_dir, CONFIG_NAME)

        with open(qmap_path, "r", encoding="utf-8") as f:
            qmap = json.load(f)

        config = cls.base_class.load_config(pretrained_model_name_or_path, **kwargs)
        with init_empty_weights():
            model = cls.base_class.from_config(config)

        checkpoint_file = os.path.join(working_dir, SAFETENSORS_WEIGHTS_NAME)
        state_dict = load_state_dict(checkpoint_file)

        requantize(model, state_dict=state_dict, quantization_map=qmap)
        model.eval()
        return cls(model)

class QuantizedFlux2Transformer2DModel(QuantizedDiffusersModel):
    """Quantized FLUX.2 Transformer model."""
    base_class = Flux2Transformer2DModel
