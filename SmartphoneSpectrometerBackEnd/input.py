"""
Input handling: validate uploads, crop, perform pixel -> wavelength mapping,
and extract both luminance and per-channel RGB column-mean spectra.

The spectrometer image is a 2D capture of light dispersed by a diffraction
grating. The horizontal axis maps to wavelength inside the user-supplied crop.
Vertical extent is collapsed by taking column means.
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from fastapi import UploadFile, HTTPException


ALLOWED_PRESETS = {"general", "chlorophyll", "bilirubin"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}

# CONFIG: visible-range pixel -> wavelength mapping.
# Linear interpolation across the cropped horizontal extent.
# Replace with two-point or polynomial calibration coefficients per device
# when known (e.g. from mercury lamp or laser-line calibration).
DEFAULT_LAMBDA_MIN_NM = 380.0
DEFAULT_LAMBDA_MAX_NM = 750.0


# ---------- pixel <-> wavelength ----------

def pixels_to_wavelengths(
    n_pixels: int,
    lambda_min: float = DEFAULT_LAMBDA_MIN_NM,
    lambda_max: float = DEFAULT_LAMBDA_MAX_NM,
) -> np.ndarray:
    """Linear pixel -> wavelength mapping across n_pixels columns."""
    if n_pixels < 2:
        raise HTTPException(status_code=400, detail="Cropped width must be at least 2 pixels.")
    return np.linspace(lambda_min, lambda_max, n_pixels)


# ---------- validators ----------

def validate_preset(preset: str) -> str:
    p = (preset or "").lower().strip()
    if p not in ALLOWED_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid preset '{preset}'. Allowed: {sorted(ALLOWED_PRESETS)}",
        )
    return p


def validate_concentration(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail=f"Concentration must be numeric or null, got {value!r}",
        )


# ---------- image helpers ----------

async def _load_image(upload: UploadFile, label: str) -> Image.Image:
    if upload is None:
        raise HTTPException(status_code=400, detail=f"Missing {label} image.")
    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type for {label}: {upload.content_type}. Expected JPEG or PNG.",
        )
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{label} image is empty.")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode {label} image: {e}")
    return img


def _crop(img: Image.Image, crop: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = crop
    w, h = img.size
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise HTTPException(
            status_code=400,
            detail=f"Crop {crop} out of bounds for image size {img.size}.",
        )
    return img.crop((x1, y1, x2, y2))


def _luminance_spectrum(img: Image.Image) -> np.ndarray:
    """Convert to grayscale, then take column means -> 1D spectrum of length crop_width."""
    gray = np.asarray(img.convert("L"), dtype=np.float64)
    return gray.mean(axis=0)


def _rgb_spectrum(img: Image.Image) -> np.ndarray:
    """Per-channel column means. Returns shape (3, crop_width) for R, G, B."""
    arr = np.asarray(img, dtype=np.float64)  # shape (H, W, 3)
    return arr.mean(axis=0).T  # (3, W)


# ---------- main entrypoint ----------

async def handle_upload(
    sample: UploadFile,
    white: UploadFile,
    dark: UploadFile,
    preset: str,
    concentration,
    crop: Tuple[int, int, int, int],
) -> dict:
    """Validate inputs and return cropped luminance + RGB spectra plus wavelength axis."""
    preset = validate_preset(preset)
    conc = validate_concentration(concentration)

    s_img = await _load_image(sample, "sample")
    w_img = await _load_image(white, "white reference")
    d_img = await _load_image(dark, "dark reference")

    s_crop = _crop(s_img, crop)
    w_crop = _crop(w_img, crop)
    d_crop = _crop(d_img, crop)

    if not (s_crop.size == w_crop.size == d_crop.size):
        raise HTTPException(
            status_code=400,
            detail="Cropped sample / white / dark sizes must all match.",
        )

    n_px = s_crop.size[0]
    wavelengths = pixels_to_wavelengths(n_px)

    return {
        "preset": preset,
        "concentration": conc,
        "wavelengths": wavelengths,
        "sample_lum": _luminance_spectrum(s_crop),
        "white_lum": _luminance_spectrum(w_crop),
        "dark_lum": _luminance_spectrum(d_crop),
        "sample_rgb": _rgb_spectrum(s_crop),
        "white_rgb": _rgb_spectrum(w_crop),
        "dark_rgb": _rgb_spectrum(d_crop),
    }