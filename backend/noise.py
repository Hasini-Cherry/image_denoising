from typing import Callable

import numpy as np

from backend.utils import ensure_uint8, validate_intensity


SUPPORTED_NOISES = {"gaussian", "salt_pepper", "speckle"}


def add_gaussian_noise(image: np.ndarray, intensity: float, rng: np.random.Generator | None = None) -> np.ndarray:
    intensity = validate_intensity(intensity)
    if intensity == 0:
        return image.copy()
    generator = rng or np.random.default_rng()
    sigma = 75.0 * intensity
    noise = generator.normal(0.0, sigma, size=image.shape)
    return ensure_uint8(image.astype(np.float32) + noise)


def add_salt_pepper_noise(image: np.ndarray, intensity: float, rng: np.random.Generator | None = None) -> np.ndarray:
    intensity = validate_intensity(intensity)
    if intensity == 0:
        return image.copy()
    generator = rng or np.random.default_rng()
    noisy = image.copy()
    amount = min(0.5, 0.3 * intensity)
    random_map = generator.random(image.shape[:2])
    salt = random_map < amount / 2.0
    pepper = (random_map >= amount / 2.0) & (random_map < amount)
    if image.ndim == 2:
        noisy[salt] = 255
        noisy[pepper] = 0
    else:
        noisy[salt, :] = 255
        noisy[pepper, :] = 0
    return ensure_uint8(noisy)


def add_speckle_noise(image: np.ndarray, intensity: float, rng: np.random.Generator | None = None) -> np.ndarray:
    intensity = validate_intensity(intensity)
    if intensity == 0:
        return image.copy()
    generator = rng or np.random.default_rng()
    image_float = image.astype(np.float32)
    multiplicative_noise = generator.normal(0.0, 0.65 * intensity, size=image.shape)
    return ensure_uint8(image_float + image_float * multiplicative_noise)


NOISE_FUNCTIONS: dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "gaussian": add_gaussian_noise,
    "salt_pepper": add_salt_pepper_noise,
    "speckle": add_speckle_noise,
}


def add_noise(image: np.ndarray, noise_type: str, intensity: float) -> np.ndarray:
    normalized_name = noise_type.strip().lower().replace("-", "_")
    if normalized_name not in NOISE_FUNCTIONS:
        supported = ", ".join(sorted(NOISE_FUNCTIONS))
        raise ValueError(f"Unsupported noise type. Choose one of: {supported}.")
    return NOISE_FUNCTIONS[normalized_name](image, intensity)
