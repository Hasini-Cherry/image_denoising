import sys
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.metrics import evaluate_images
from backend.noise import NOISE_FUNCTIONS, add_noise
from backend.svd_denoise import auto_k_values, factorize_svd, reconstruct_many_from_factors, denoise_svd
from backend.utils import ImageValidationError, decode_image_bytes, image_to_png_bytes


def sample_rgb(height: int = 96, width: int = 128) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    red = np.tile(x, (height, 1))
    green = np.tile(y, (1, width))
    blue = (0.55 * red + 0.45 * green)
    return np.stack([red, green, blue], axis=2).astype(np.uint8)


def make_png_bytes(image: np.ndarray) -> bytes:
    return image_to_png_bytes(image)


class ImageDenoisingTests(unittest.TestCase):
    def test_invalid_upload_is_rejected(self):
        with self.assertRaises(ImageValidationError):
            decode_image_bytes(b"this is not a png or jpg")

    def test_large_image_is_resized(self):
        large = Image.new("RGB", (1800, 1200), (80, 140, 210))
        buffer = BytesIO()
        large.save(buffer, format="PNG")
        image, metadata = decode_image_bytes(buffer.getvalue())
        self.assertTrue(metadata["resized"])
        self.assertLessEqual(max(image.shape[:2]), 1024)
        self.assertEqual(metadata["max_rank"], min(image.shape[:2]))

    def test_all_noise_models_return_valid_images(self):
        image = sample_rgb()
        for noise_type in NOISE_FUNCTIONS:
            with self.subTest(noise_type=noise_type):
                noisy = add_noise(image, noise_type, 0.5)
                self.assertEqual(noisy.shape, image.shape)
                self.assertEqual(noisy.dtype, np.uint8)
                self.assertGreaterEqual(noisy.min(), 0)
                self.assertLessEqual(noisy.max(), 255)

    def test_zero_intensity_leaves_image_unchanged(self):
        image = sample_rgb()
        for noise_type in NOISE_FUNCTIONS:
            with self.subTest(noise_type=noise_type):
                noisy = add_noise(image, noise_type, 0.0)
                self.assertTrue(np.array_equal(noisy, image))

    def test_low_k_changes_image_more_than_high_k(self):
        image = sample_rgb()
        noisy = add_noise(image, "gaussian", 0.2)
        low_rank, low_meta = denoise_svd(noisy, 1)
        high_rank, high_meta = denoise_svd(noisy, min(noisy.shape[:2]))
        low_delta = np.mean((low_rank.astype(np.float32) - noisy.astype(np.float32)) ** 2)
        high_delta = np.mean((high_rank.astype(np.float32) - noisy.astype(np.float32)) ** 2)
        self.assertGreater(low_delta, high_delta)
        self.assertTrue(low_meta["warnings"])
        self.assertTrue(high_meta["warnings"])

    def test_metrics_are_stable(self):
        image = sample_rgb()
        noisy = add_noise(image, "speckle", 0.15)
        denoised, _ = denoise_svd(noisy, max(2, min(noisy.shape[:2]) // 6))
        metrics = evaluate_images(image, denoised)
        self.assertIn("psnr", metrics)
        self.assertIn("ssim", metrics)
        self.assertGreaterEqual(metrics["psnr"], 0)
        self.assertGreaterEqual(metrics["ssim"], 0)
        self.assertLessEqual(metrics["ssim"], 1)

    def test_auto_mode_generates_five_ranked_outputs_from_one_factorization(self):
        image = sample_rgb()
        noisy = add_noise(image, "gaussian", 0.35)
        factors = factorize_svd(noisy)
        k_values = auto_k_values(factors)
        outputs = reconstruct_many_from_factors(factors, k_values)

        self.assertEqual(len(k_values), 5)
        self.assertEqual(len(outputs), 5)
        self.assertTrue(all(1 <= k <= min(noisy.shape[:2]) for k in k_values))
        self.assertEqual(sorted(k_values), k_values)
        for rank, output, metadata in outputs:
            self.assertEqual(output.shape, image.shape)
            self.assertEqual(output.dtype, np.uint8)
            self.assertEqual(metadata["k"], rank)


if __name__ == "__main__":
    unittest.main(verbosity=2)
