# SVD Denoiser

Production-ready full-stack web app for adding classical image noise and denoising images with Singular Value Decomposition (SVD). The backend is FastAPI, NumPy, OpenCV, scikit-image, and Pillow. The frontend is a static HTML/CSS/JS app with a cyberpunk dark theme, light mode, manual k control, and automatic energy-based k selection.

## Project Overview

The app lets you upload a JPG, JPEG, or PNG image, resize oversized images safely, add one of three noise models, denoise the noisy image with reduced SVD, and evaluate the final denoised result against the original image using PSNR and SSIM.

Supported noise models:

- Gaussian noise
- Salt-and-pepper noise
- Speckle noise

Core outputs:

- Original image
- Noisy image
- Manual mode denoised image
- Auto mode grid of 5 denoised images
- PSNR and SSIM for noisy image
- PSNR and SSIM for every denoised image
- Backend processing time in milliseconds
- PSNR/SSIM chart by k in auto mode

## How SVD Works

SVD breaks an image matrix into three parts:

```text
A = U * Sigma * V^T
```

`Sigma` contains singular values. The largest singular values describe the strongest structure in the image. Smaller values often contain fine detail and noise. By reconstructing the image with only the top `k` singular values, the app keeps the dominant image structure and suppresses part of the noise.

For RGB images, the backend processes each color channel separately. Grayscale images are processed directly as one matrix.

## What k Means

`k` is the SVD rank, or the number of singular values used during reconstruction.

- Very low `k`: stronger compression, often blurry or distorted.
- Medium `k`: useful denoising range for many images.
- Very high `k`: image remains close to the noisy input, so denoising may be weak.

The backend enforces:

```text
1 <= k <= min(height, width)
```

The UI warns when `k` is very low or very high.

## Manual and Auto Modes

Manual mode lets you choose `k` directly with a slider. The UI shows a dynamic recommendation based on noise intensity:

- Low noise: 15% to 30% of the minimum image dimension
- Medium noise: 10% to 20%
- High noise: 5% to 15%

Auto mode computes SVD once on the noisy image, then reconstructs five outputs using k values at 20%, 30%, 40%, 50%, and 60% of the minimum image dimension. The best output is highlighted by highest SSIM.

## Setup

1. Open a terminal in this project folder.
2. Install dependencies once in your normal Python installation:

```bash
pip install -r requirements.txt
```

## Run Backend

From the project root:

```powershell
cd backend
python main.py
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Open Frontend

Open this file in a browser:

```text
frontend/index.html
```

The frontend calls the API at:

```text
http://127.0.0.1:8000
```

## API Endpoints

The image endpoints accept raw JPG/PNG bytes in the request body. This keeps `requirements.txt` exactly to the requested dependencies and avoids adding multipart parsing packages.

- `POST /upload`: validates and resizes an image, returns a PNG data URL preview.
- `POST /add-noise?noise_type=gaussian&intensity=0.25`: returns a noisy image.
- `POST /denoise?k=40`: returns an SVD-denoised image.
- `POST /auto-denoise?noise_type=gaussian&intensity=0.25`: adds noise, computes noisy metrics, computes SVD once, returns five denoised outputs at 20%, 30%, 40%, 50%, and 60% k values with PSNR/SSIM.
- `POST /evaluate`: accepts JSON with original and denoised image data URLs, returns PSNR and SSIM.

## Example Results

Typical behavior on a 512x512 image:

- Add Gaussian noise with intensity around `0.20` to `0.35`.
- Choose `k` around `10%` to `25%` of the minimum image dimension.
- The denoised image should look smoother than the noisy image.
- PSNR and SSIM should usually improve when `k` is tuned well for the image and noise level.

## Test Cases

Run:

```bash
python backend/test_cases.py
```

Covered cases:

- Invalid upload bytes are rejected.
- Very large image is resized to a maximum dimension of 512 pixels to keep SVD memory and CPU usage practical.
- Very low `k` produces a strongly compressed/blurry output.
- Very high `k` reconstructs close to the noisy input.
- Auto mode returns 5 valid ranked outputs from one reusable SVD factorization.
- All three noise types return valid `uint8` images without overflow or underflow.
- PSNR and SSIM calculations remain numerically stable.

## Limitations

- SVD denoising is global and can blur fine textures when `k` is too low.
- Very high `k` preserves noise because it keeps most singular values.
- SVD on very large images can be expensive, so uploads are resized to 512 pixels on the longest side.
- Metrics compare original vs noisy and original vs each denoised image.
- Classical SVD denoising is not a replacement for modern learned denoisers, but it is transparent and mathematically interpretable.
