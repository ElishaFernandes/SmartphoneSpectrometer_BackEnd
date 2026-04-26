"""Shared test fixtures: synthetic spectrum images and path setup."""
import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

# make the project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _png_bytes_from_array(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_sample_image():
    """200 x 400 image with three Gaussian-ish bumps roughly mimicking RGB peaks."""
    h, w = 50, 400
    arr = np.zeros((h, w, 3), dtype=np.float64)
    x = np.arange(w)
    arr[:, :, 0] = 220 * np.exp(-((x - 280) ** 2) / (2 * 60 ** 2))   # red ~ 600 nm
    arr[:, :, 1] = 220 * np.exp(-((x - 180) ** 2) / (2 * 60 ** 2))   # green ~ 530 nm
    arr[:, :, 2] = 220 * np.exp(-((x - 60) ** 2) / (2 * 60 ** 2))    # blue ~ 440 nm
    return _png_bytes_from_array(arr)


@pytest.fixture
def synthetic_white_image():
    return _png_bytes_from_array(np.full((50, 400, 3), 250))


@pytest.fixture
def synthetic_dark_image():
    return _png_bytes_from_array(np.full((50, 400, 3), 5))


@pytest.fixture
def crop_box():
    """A safely-inside crop for the 50x400 synthetic images."""
    return {"crop_x1": 10, "crop_y1": 5, "crop_x2": 390, "crop_y2": 45}