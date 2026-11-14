import time
from pathlib import Path
import sys
import os
import gc

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(thread_env, "2")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.metrics import evaluate_images
from backend.noise import NOISE_FUNCTIONS, add_noise
from backend.svd_denoise import auto_k_values, denoise_svd, factorize_svd, reconstruct_from_factors
from backend.utils import (
    ImageValidationError,
    data_url_to_bytes,
    decode_image_bytes,
    image_to_data_url,
    rank_warnings,
)


app = FastAPI(
    title="SVD Image Denoising API",
    description="Image upload, classical noise generation, SVD denoising, and PSNR/SSIM evaluation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EvaluateRequest(BaseModel):
    original: str
    denoised: str


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


async def _read_request_image(request: Request):
    body = await request.body()
    try:
        return decode_image_bytes(body)
    except ImageValidationError as exc:
        raise _bad_request(str(exc)) from exc


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(request: Request):
    image, metadata = await _read_request_image(request)
    return {
        "image": image_to_data_url(image),
        "metadata": metadata,
        "message": "Image uploaded successfully.",
    }


@app.post("/add-noise")
async def add_noise_endpoint(
    request: Request,
    noise_type: str = Query(..., description="gaussian, salt_pepper, or speckle"),
    intensity: float = Query(..., ge=0.0, le=1.0),
):
    image, metadata = await _read_request_image(request)
    start = time.perf_counter()
    try:
        noisy = add_noise(image, noise_type, intensity)
    except (ValueError, ImageValidationError) as exc:
        raise _bad_request(str(exc)) from exc
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    return {
        "image": image_to_data_url(noisy),
        "metadata": metadata,
        "noise_type": noise_type,
        "intensity": intensity,
        "processing_time_ms": elapsed_ms,
    }


@app.post("/denoise")
async def denoise_endpoint(
    request: Request,
    k: int = Query(..., ge=1, description="Top-k singular values to keep."),
):
    image, metadata = await _read_request_image(request)
    start = time.perf_counter()
    try:
        denoised, denoise_metadata = denoise_svd(image, k)
    except (ValueError, ImageValidationError) as exc:
        raise _bad_request(str(exc)) from exc
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    return {
        "image": image_to_data_url(denoised),
        "metadata": metadata,
        "denoise": denoise_metadata,
        "processing_time_ms": elapsed_ms,
    }


@app.post("/auto-denoise")
async def auto_denoise_endpoint(
    request: Request,
    noise_type: str = Query(..., description="gaussian, salt_pepper, or speckle"),
    intensity: float = Query(..., ge=0.0, le=1.0),
):
    original, metadata = await _read_request_image(request)
    start = time.perf_counter()
    try:
        noisy = add_noise(original, noise_type, intensity)
        noisy_metrics = evaluate_images(original, noisy)
        factors = factorize_svd(noisy)
        k_values = auto_k_values(factors, target_energy=0.92, count=5)

        results = []
        for rank in k_values:
            denoised = reconstruct_from_factors(factors, rank)
            metrics = evaluate_images(original, denoised)
            results.append(
                {
                    "k": rank,
                    "image": image_to_data_url(denoised),
                    "metrics": metrics,
                    "warnings": rank_warnings(rank, denoised),
                }
            )
            del denoised
        del factors
        gc.collect()
    except (ValueError, ImageValidationError) as exc:
        raise _bad_request(str(exc)) from exc

    best_index = max(range(len(results)), key=lambda index: results[index]["metrics"]["ssim"]) if results else 0
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 2)
    return {
        "metadata": metadata,
        "noisy_image": image_to_data_url(noisy),
        "noisy_metrics": noisy_metrics,
        "results": results,
        "best_index": best_index,
        "best_k": results[best_index]["k"] if results else None,
        "selection": {
            "strategy": "fixed_rank_percentages",
            "percentages": [20, 30, 40, 50, 60],
            "k_values": k_values,
        },
        "processing_time_ms": elapsed_ms,
    }


@app.post("/evaluate")
async def evaluate(payload: EvaluateRequest):
    try:
        original, _ = decode_image_bytes(data_url_to_bytes(payload.original))
        denoised, _ = decode_image_bytes(data_url_to_bytes(payload.denoised))
        metrics = evaluate_images(original, denoised)
    except ImageValidationError as exc:
        raise _bad_request(str(exc)) from exc
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    return metrics


@app.get("/noise-types")
async def noise_types():
    return {"noise_types": sorted(NOISE_FUNCTIONS)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
