"""FastAPI application for serve_f2k4b_quantized.

Exposes an OpenAI-compatible ``POST /v1/images/generations`` endpoint backed
by the FLUX.2-klein-4B quantized pipeline.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state — populated at startup
# ---------------------------------------------------------------------------
_pipeline: Any = None
_cfg: dict = {}


# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ImageGenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = Field(default=None, description="Ignored; accepted for compatibility.")
    n: int = Field(default=1, ge=1, le=10)
    size: str = Field(default="1024x1024", description='e.g. "1024x1024"')
    response_format: Optional[str] = Field(
        default=None,
        description='"b64_json" or "url". Defaults to api.default_response_format.',
    )
    num_inference_steps: int = Field(default=20, ge=1)


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="serve_f2k4b_quantized", version="0.1.0")

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(exc),
                "type": exc.__class__.__name__
            }
        }
    )


def _parse_size(size_str: str) -> tuple[int, int]:
    parts = size_str.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size string: {size_str!r}.  Expected 'WxH'.")
    width, height = int(parts[0]), int(parts[1])
    return width, height


def _image_to_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.post("/v1/images/generations", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest):
    import time

    global _pipeline, _cfg

    logger.info("images/generations prompt_len=%d size=%s steps=%d n=%d fmt=%s",
                req_id, len(request.prompt), request.size, request.num_inference_steps,
                request.n, request.response_format)

    response_format = request.response_format or _cfg.get("api", {}).get(
        "default_response_format", "b64_json"
    )

    width, height = _parse_size(request.size)

    data: list[ImageData] = []
    for _ in range(request.n):
        result = _pipeline(
            prompt=request.prompt,
            width=width,
            height=height,
            num_inference_steps=request.num_inference_steps,
        )
        image = result.images[0]

        if response_format == "b64_json":
            data.append(ImageData(b64_json=_image_to_b64(image)))
        else:
            static_dir = Path(_cfg.get("server", {}).get("static_dir", "./static"))
            static_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{uuid.uuid4()}.png"
            image.save(static_dir / filename)
            host = _cfg.get("server", {}).get("host", "0.0.0.0")
            port = _cfg.get("server", {}).get("port", 8000)
            data.append(ImageData(url=f"http://{host}:{port}/static/{filename}"))

    return ImageGenerationResponse(created=int(time.time()), data=data)

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "ok",
        # Optional: return a bit of useful info about the running server
        "profile": _cfg.get("model", {}).get("profile", "unknown"),
        "device": _cfg.get("model", {}).get("device", "unknown")
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Serve FLUX.2-klein-4B via OpenAI-compatible API")
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: use built-in defaults)",
    )
    # Add new arguments for host and port
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override the listening address (e.g., 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override the listening port (e.g., 9000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="Logging level: debug, info, warn, error",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from serve_f2k4b_quantized.config import load_config
    from serve_f2k4b_quantized.loader import load_pipeline

    global _pipeline, _cfg
    _cfg = load_config(args.config)

    # Apply command-line overrides if they were provided
    if args.host is not None:
        _cfg["server"]["host"] = args.host
    if args.port is not None:
        _cfg["server"]["port"] = args.port
    if args.log_level is not None:
        _cfg["server"]["log_level"] = args.log_level

    logger.info("Configuration loaded: %s", _cfg)

    # Mount static files if static_dir exists or will be created
    static_dir = Path(_cfg["server"]["static_dir"])
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    logger.info("Loading pipeline …")
    _pipeline = load_pipeline(_cfg)
    logger.info("Pipeline ready.")

    uvicorn.run(
        app,
        host=_cfg["server"]["host"],
        port=_cfg["server"]["port"],
        log_level=_["server"].get("log_level", "info")
    )

if __name__ == "__main__":
    main()
