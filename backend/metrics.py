from typing import Dict

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _align_images(original: np.ndarray, denoised: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if original.ndim != denoised.ndim:
        if original.ndim == 2 and denoised.ndim == 3:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
        elif original.ndim == 3 and denoised.ndim == 2:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)

    if original.shape[:2] != denoised.shape[:2]:
        height, width = original.shape[:2]
        denoised = cv2.resize(denoised, (width, height), interpolation=cv2.INTER_AREA)
    return original, denoised


def calculate_psnr(original: np.ndarray, denoised: np.ndarray) -> float:
    original, denoised = _align_images(original, denoised)
    diff = original.astype(np.float64) - denoised.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 99.0
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    return round(float(psnr), 4)


def calculate_ssim(original: np.ndarray, denoised: np.ndarray) -> float:
    original, denoised = _align_images(original, denoised)
    min_dimension = min(original.shape[:2])
    if min_dimension < 3:
        return 1.0 if np.array_equal(original, denoised) else 0.0

    win_size = min(7, min_dimension)
    if win_size % 2 == 0:
        win_size -= 1

    channel_axis = -1 if original.ndim == 3 else None
    value = structural_similarity(
        original,
        denoised,
        channel_axis=channel_axis,
        data_range=255,
        win_size=win_size,
    )
    return round(float(np.clip(value, 0.0, 1.0)), 4)


def evaluate_images(original: np.ndarray, denoised: np.ndarray) -> Dict[str, float]:
    return {
        "psnr": calculate_psnr(original, denoised),
        "ssim": calculate_ssim(original, denoised),
    }
