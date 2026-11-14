import base64
from io import BytesIO
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_DIMENSION = 512
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "PNG"}


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot be processed safely."""


def _resample_filter() -> int:
    return getattr(Image, "Resampling", Image).LANCZOS


def ensure_uint8(image: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(clean, 0, 255).astype(np.uint8)


def resize_pil_image(image: Image.Image, max_dimension: int = MAX_DIMENSION) -> Tuple[Image.Image, bool]:
    width, height = image.size
    largest = max(width, height)
    if largest <= max_dimension:
        return image, False

    scale = max_dimension / float(largest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, _resample_filter()), True


def decode_image_bytes(data: bytes, max_dimension: int = MAX_DIMENSION) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not data:
        raise ImageValidationError("No image data was provided.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageValidationError("Image is too large. Maximum upload size is 16 MB.")

    try:
        with Image.open(BytesIO(data)) as pil_image:
            if pil_image.format not in ALLOWED_FORMATS:
                raise ImageValidationError("Only JPG, JPEG, and PNG images are supported.")

            pil_image = ImageOps.exif_transpose(pil_image)
            original_width, original_height = pil_image.size

            if pil_image.mode == "L":
                normalized = pil_image
            elif pil_image.mode in {"RGB", "RGBA", "P", "LA"}:
                normalized = pil_image.convert("RGB")
            else:
                normalized = pil_image.convert("RGB")

            normalized, resized = resize_pil_image(normalized, max_dimension=max_dimension)
            array = np.asarray(normalized)
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("The uploaded file is not a valid image.") from exc

    if array.ndim not in {2, 3}:
        raise ImageValidationError("Unsupported image shape.")
    if min(array.shape[:2]) < 2:
        raise ImageValidationError("Image is too small to process.")

    height, width = array.shape[:2]
    return ensure_uint8(array), {
        "width": width,
        "height": height,
        "channels": 1 if array.ndim == 2 else array.shape[2],
        "original_width": original_width,
        "original_height": original_height,
        "resized": resized,
        "max_rank": min(height, width),
    }


def image_to_png_bytes(image: np.ndarray) -> bytes:
    image = ensure_uint8(image)
    mode = "L" if image.ndim == 2 else "RGB"
    pil_image = Image.fromarray(image, mode=mode)
    output = BytesIO()
    pil_image.save(output, format="PNG", compress_level=1)
    return output.getvalue()


def image_to_data_url(image: np.ndarray) -> str:
    encoded = base64.b64encode(image_to_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def data_url_to_bytes(data_url: str) -> bytes:
    if not isinstance(data_url, str) or "," not in data_url:
        raise ImageValidationError("Expected a valid image data URL.")
    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image/"):
        raise ImageValidationError("Expected an image data URL.")
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ImageValidationError("Image data URL is not valid base64.") from exc


def validate_intensity(intensity: float) -> float:
    try:
        value = float(intensity)
    except (TypeError, ValueError) as exc:
        raise ImageValidationError("Noise intensity must be a number from 0 to 1.") from exc
    if not 0.0 <= value <= 1.0:
        raise ImageValidationError("Noise intensity must be between 0 and 1.")
    return value


def validate_rank(k: int, image: np.ndarray) -> int:
    min_dimension = min(image.shape[:2])
    try:
        rank = int(k)
    except (TypeError, ValueError) as exc:
        raise ImageValidationError("SVD rank k must be an integer.") from exc
    if rank < 1 or rank > min_dimension:
        raise ImageValidationError(f"SVD rank k must satisfy 1 <= k <= {min_dimension}.")
    return rank


def rank_warnings(k: int, image: np.ndarray) -> list[str]:
    min_dimension = min(image.shape[:2])
    warnings: list[str] = []
    if k <= max(2, int(0.04 * min_dimension)):
        warnings.append("k is very low; the output may look blurry or distorted.")
    if k >= max(1, int(0.85 * min_dimension)):
        warnings.append("k is very high; denoising may be weak because most detail is preserved.")
    return warnings


def image_metadata(image: np.ndarray, resized: bool = False, original_size: tuple[int, int] | None = None) -> Dict[str, Any]:
    height, width = image.shape[:2]
    metadata: Dict[str, Any] = {
        "width": width,
        "height": height,
        "channels": 1 if image.ndim == 2 else image.shape[2],
        "resized": resized,
        "max_rank": min(height, width),
    }
    if original_size:
        metadata["original_width"] = original_size[0]
        metadata["original_height"] = original_size[1]
    return metadata
