import os

os.environ["DATABASE_URL"] = "sqlite:///./test_main.db"

import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

import main
from database import init_db

init_db()
client = TestClient(main.app)


def _png_bytes(rgb_value: int) -> io.BytesIO:
    arr = np.full((50, 200, 3), rgb_value, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture(scope="module", autouse=True)
def _cleanup():
    yield
    if os.path.exists("./test_main.db"):
        os.remove("./test_main.db")


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_analyse_endpoint_general_preset():
    files = {
        "sample": ("s.png", _png_bytes(120), "image/png"),
        "white": ("w.png", _png_bytes(250), "image/png"),
        "dark": ("d.png", _png_bytes(5), "image/png"),
    }
    data = {
        "preset": "general",
        "concentration": 1.5,
        "crop_x1": 10, "crop_y1": 5, "crop_x2": 190, "crop_y2": 45,
    }
    r = client.post("/analyse", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "session_id" in body
    assert "result" in body
    assert "first_derivative" in body["result"]
    assert "second_derivative" in body["result"]
    assert "ratios" in body["result"]


def test_analyse_invalid_preset_rejected():
    files = {
        "sample": ("s.png", _png_bytes(120), "image/png"),
        "white": ("w.png", _png_bytes(250), "image/png"),
        "dark": ("d.png", _png_bytes(5), "image/png"),
    }
    data = {
        "preset": "ultraviolet",
        "crop_x1": 10, "crop_y1": 5, "crop_x2": 190, "crop_y2": 45,
    }
    r = client.post("/analyse", files=files, data=data)
    assert r.status_code == 400


def test_compare_too_few_sessions():
    r = client.post("/compare", json={"session_ids": ["only-one"]})
    assert r.status_code == 400