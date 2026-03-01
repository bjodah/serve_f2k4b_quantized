# serve_f2k4b_quantized

A minimal Python package that serves the **FLUX.2-klein-4B** image-generation
model through an **OpenAI-compatible REST API** (FastAPI).

> **GENERATIVE AI DISCLAIMER**: The code in this repository was mostly written by LLMs
> (but with plenty of guidance from yours truly).

It combines:
* A **Q8_0 GGUF**-quantized Transformer (Unsloth) loaded via Diffusers.
* An **Int8**-quantized Text Encoder (optimum-quanto).

This keeps peak VRAM usage well within the 24 GB budget of an RTX 3090, and
degrades gracefully to CPU-only execution when no GPU is present.

---

## Table of Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [CUDA Setup](#cuda-setup)
4. [Running the Server](#running-the-server)
5. [CPU Deployment](#cpu-deployment)
6. [Open-WebUI Integration](#open-webui-integration)
7. [API Reference](#api-reference)
8. [Testing](#testing)

---

## Requirements

| Component | Version |
|-----------|---------|
| Python    | ≥ 3.10  |
| uv        | any recent |
| CUDA driver | ≥ 12.8 (for GPU) |
| GPU VRAM  | ≥ 20 GB recommended |

---

## Installation

> **Note:** This project uses [`uv`](https://github.com/astral-sh/uv) as its
> package manager.  Install it with `pip install uv` or follow the
> [official instructions](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# 1. Clone the repository
git clone https://github.com/bjodah/serve_f2k4b_quantized.git
cd serve_f2k4b_quantized

# 2. Install all dependencies (including the git-based Diffusers)
uv sync

# 3. (Optional) install test dependencies
uv sync --extra test
```

`uv` automatically picks up the custom PyTorch CUDA 12.8 index from
`pyproject.toml`, so no extra index configuration is required.

---

## CUDA Setup

1. Ensure your NVIDIA driver supports **CUDA 12.8** or later.
2. Enable the fast Diffusers GGUF CUDA kernels:
   ```bash
   export DIFFUSERS_GGUF_CUDA_KERNELS=true
   ```
   You can also set this permanently in your shell profile.

3. Verify your environment:
   ```bash
   uv run python scripts/minimal_cublas_gemm.py
   ```
   Expected output ends with `cuBLAS GEMM OK ✓`.

---

## Running the Server

```bash
# 1. Copy the example configuration and edit it
cp config.example.yaml config.yaml
$EDITOR config.yaml

# 2. Launch the server
uv run serve-f2k4b --config config.yaml
```

Or use the convenience helper script:
```bash
bash scripts/launch.sh
```

The server listens on `http://0.0.0.0:8000` by default.

### Key Configuration Options (`config.yaml`)

```yaml
server:
  host: "0.0.0.0"   # bind address
  port: 8000         # TCP port
  static_dir: "./static"   # where URL-format images are saved

model:
  profile: "mixed-q8-int8"  # quantization profile
  device: "cuda"             # "cuda" or "cpu"
  compute_dtype: "bfloat16"  # "bfloat16" (GPU) or "float32" (CPU)

api:
  default_response_format: "b64_json"  # "b64_json" or "url"
```

---

## CPU Deployment

To run entirely on CPU (no GPU required), set the following in `config.yaml`:

```yaml
model:
  device: "cpu"
  compute_dtype: "float32"
```

`compute_dtype` is automatically forced to `"float32"` when `device == "cpu"`,
even if you accidentally leave it as `"bfloat16"`.

> **Warning:** CPU-only inference is extremely slow for diffusion models.
> Expect several minutes per image.

---

## Open-WebUI Integration

Set these environment variables in your Open-WebUI instance to point it at the
local server:

```bash
IMAGES_OPENAI_API_BASE_URL=http://127.0.0.1:8000/v1
IMAGES_OPENAI_API_KEY=dummy-key
```

No changes to Open-WebUI's source code are required — it uses the standard
OpenAI image generation API format.

---

## Manually querying the endpoint

```console
curl -s -X POST http://127.0.0.1:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic city skyline at sunset in a cyberpunk style",
    "size": "1024x1024",
    "num_inference_steps": 20,
    "response_format": "b64_json"
  }' | jq -r '.data[0].b64_json' | base64 --decode > generated_image.png
```

---

## API Reference

### `POST /v1/images/generations`

**Request body** (JSON):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | string | — | Image generation prompt |
| `model` | string | `null` | Ignored (accepted for compatibility) |
| `n` | int | `1` | Number of images to generate |
| `size` | string | `"1024x1024"` | Output resolution, e.g. `"512x512"` |
| `response_format` | string | config default | `"b64_json"` or `"url"` |
| `num_inference_steps` | int | `20` | Diffusion steps (lower = faster, lower quality) |

**Response** (JSON, OpenAI-compatible):

```json
{
  "created": 1710000000,
  "data": [
    { "b64_json": "<base64-encoded PNG>" }
  ]
}
```

---

## Testing

### CI-safe subset (no GPU, no model downloads)

These tests validate configuration parsing and pure-Python utilities.
They run on any machine:

```bash
SKIP_HARDWARE_TESTS=1 pytest tests/test_config.py -v
```

Or simply:

```bash
pytest tests/test_config.py -v
```

(`SKIP_HARDWARE_TESTS` defaults to `"1"` in the integration test file.)

### Full integration tests (requires RTX 3090 + model weights)

```bash
SKIP_HARDWARE_TESTS= pytest tests/test_integration.py -v
```

### Run everything

```bash
pytest tests/ -v
```
