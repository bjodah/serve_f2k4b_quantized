#!/usr/bin/env bash
# launch.sh — Helper script for starting serve_f2k4b_quantized
#
# Usage:
#   bash scripts/launch.sh [--config PATH_TO_CONFIG]
#
# The script assumes `uv` is available on PATH and that the project has been
# installed with `uv sync`.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${REPO_ROOT}/config.yaml"

# Allow overriding the config path via the first argument
if [[ "${1:-}" == "--config" && -n "${2:-}" ]]; then
    CONFIG="$2"
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo "  Copy config.example.yaml to config.yaml and edit it first:"
    echo "    cp config.example.yaml config.yaml"
    exit 1
fi

# Optional: enable Diffusers CUDA kernels for the GGUF transformer
export DIFFUSERS_GGUF_CUDA_KERNELS="${DIFFUSERS_GGUF_CUDA_KERNELS:-true}"

echo "Starting serve_f2k4b_quantized with config: $CONFIG"
uv run serve-f2k4b --config "$CONFIG"
