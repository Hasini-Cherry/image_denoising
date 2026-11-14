from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from backend.utils import ensure_uint8, rank_warnings, validate_rank


@dataclass(frozen=True)
class SVDFactors:
    image_shape: tuple[int, ...]
    channels: tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], ...]

    @property
    def max_rank(self) -> int:
        return min(self.image_shape[:2])

    @property
    def primary_singular_values(self) -> np.ndarray:
        if not self.channels:
            return np.array([], dtype=np.float32)
        return self.channels[0][1]


def _factorize_channel(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.ascontiguousarray(channel, dtype=np.float32)
    return np.linalg.svd(matrix, full_matrices=False)


def factorize_svd(image: np.ndarray) -> SVDFactors:
    if image.ndim == 2:
        channels = (_factorize_channel(image),)
    elif image.ndim == 3 and image.shape[2] in {3, 4}:
        channels = tuple(_factorize_channel(image[:, :, channel_index]) for channel_index in range(3))
    else:
        raise ValueError("SVD denoising supports grayscale and RGB images only.")
    return SVDFactors(image_shape=tuple(image.shape), channels=channels)


def reconstruct_from_factors(factors: SVDFactors, k: int) -> np.ndarray:
    min_dimension = factors.max_rank
    rank = int(np.clip(k, 1, min_dimension))
    reconstructed_channels = []
    for u, singular_values, vt in factors.channels:
        channel = (u[:, :rank] * singular_values[:rank]) @ vt[:rank, :]
        reconstructed_channels.append(channel)

    if len(reconstructed_channels) == 1 and len(factors.image_shape) == 2:
        return ensure_uint8(reconstructed_channels[0])
    return ensure_uint8(np.stack(reconstructed_channels, axis=2))


def denoise_svd(image: np.ndarray, k: int) -> Tuple[np.ndarray, Dict[str, object]]:
    rank = validate_rank(k, image)
    factors = factorize_svd(image)
    output = reconstruct_from_factors(factors, rank)
    singular_curve = _compact_singular_values(factors.primary_singular_values)
    metadata: Dict[str, object] = {
        "k": rank,
        "warnings": rank_warnings(rank, image),
        "singular_values": singular_curve,
    }
    return output, metadata


def select_energy_rank(factors: SVDFactors, target_energy: float = 0.92) -> int:
    target = float(np.clip(target_energy, 0.9, 0.95))
    if not factors.channels:
        return 1

    energy = np.zeros(factors.max_rank, dtype=np.float64)
    for _, singular_values, _ in factors.channels:
        energy += singular_values.astype(np.float64) ** 2

    total = float(np.sum(energy))
    if total <= 1e-12:
        return 1

    cumulative = np.cumsum(energy) / total
    return int(np.searchsorted(cumulative, target, side="left") + 1)


def auto_k_values(factors: SVDFactors, target_energy: float = 0.92, count: int = 5) -> List[int]:
    max_rank = factors.max_rank
    percentages = [0.2, 0.3, 0.4, 0.5, 0.6]
    values: list[int] = []
    for percentage in percentages[:count]:
        rank = int(np.clip(round(max_rank * percentage), 1, max_rank))
        if rank not in values:
            values.append(rank)

    candidate = 1
    while len(values) < count:
        if candidate not in values:
            values.append(candidate)
        candidate += 1

    return values[:count]


def reconstruct_many_from_factors(factors: SVDFactors, k_values: Sequence[int]) -> List[Tuple[int, np.ndarray, Dict[str, object]]]:
    outputs: list[Tuple[int, np.ndarray, Dict[str, object]]] = []
    dummy_shape = np.zeros(factors.image_shape[:2], dtype=np.uint8)
    singular_curve = _compact_singular_values(factors.primary_singular_values)
    for k in k_values:
        rank = int(np.clip(k, 1, factors.max_rank))
        image = reconstruct_from_factors(factors, rank)
        outputs.append(
            (
                rank,
                image,
                {
                    "k": rank,
                    "warnings": rank_warnings(rank, dummy_shape),
                    "singular_values": singular_curve,
                },
            )
        )
    return outputs


def _compact_singular_values(values: np.ndarray, max_points: int = 96) -> List[float]:
    if values.size == 0:
        return []
    if values.size <= max_points:
        selected = values
    else:
        indexes = np.linspace(0, values.size - 1, max_points).astype(int)
        selected = values[indexes]
    max_value = float(np.max(selected))
    if max_value <= 0:
        return [0.0 for _ in selected]
    return [round(float(value / max_value), 6) for value in selected]
