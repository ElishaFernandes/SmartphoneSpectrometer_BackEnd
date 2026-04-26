"""
benchmark.py — wall-clock runtime measurement for the spectrometer backend.

Runs two passes:

  [1] End-to-end HTTP round-trip via FastAPI TestClient.
      This is the figure to quote as "total runtime per analysis":
      multipart parse -> validation -> image decode -> analysis -> DB write
      -> JSON serialisation -> response.

  [2] Per-stage internal timing (HTTP layer bypassed).
      Identifies which stage dominates the runtime — useful for arguing
      *which* parts would benefit from a port to a faster language.

Usage (from inside the spectrometer_backend folder):
    python benchmark.py

Outputs benchmark_results.csv alongside this file.
"""
from __future__ import annotations

import os
os.environ.setdefault("DATABASE_URL", "sqlite:///./bench.db")

import io
import csv
import time
import statistics
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

import main
from database import init_db, SessionLocal, create_session_record
from input import (
    pixels_to_wavelengths,
    _luminance_spectrum,
    _rgb_spectrum,
    _crop,
)
from analysis import (
    compute_reflectance,
    channel_ratios,
    peak_values,
    integrated_reflectance,
)
import preprocessing as pp


N_RUNS = 50      # measured iterations
WARMUP = 5       # discarded (interpreter / cache warm-up)
IMG_W, IMG_H = 800, 100


def _png_bytes(value: int) -> bytes:
    arr = np.full((IMG_H, IMG_W, 3), value, dtype=np.uint8)
    x = np.arange(IMG_W)
    bump = (30 * np.exp(-((x - IMG_W // 2) ** 2) / (2 * 60 ** 2))).astype(np.int32)
    arr[:, :, 1] = np.clip(arr[:, :, 1].astype(np.int32) + bump, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- pass 1: full HTTP round-trip ----------

def benchmark_http():
    init_db()
    client = TestClient(main.app)
    sample = _png_bytes(120)
    white = _png_bytes(245)
    dark = _png_bytes(5)

    times_ms = []
    for i in range(N_RUNS + WARMUP):
        files = {
            "sample": ("s.png", io.BytesIO(sample), "image/png"),
            "white":  ("w.png", io.BytesIO(white),  "image/png"),
            "dark":   ("d.png", io.BytesIO(dark),   "image/png"),
        }
        data = {
            "preset": "general",
            "crop_x1": 10, "crop_y1": 10, "crop_x2": 790, "crop_y2": 90,
        }
        t0 = time.perf_counter()
        r = client.post("/analyse", files=files, data=data)
        t1 = time.perf_counter()
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        if i >= WARMUP:
            times_ms.append((t1 - t0) * 1000)
    return times_ms


# ---------- pass 2: per-stage internal timing ----------

def benchmark_stages():
    init_db()
    db = SessionLocal()
    sample = _png_bytes(120)
    white = _png_bytes(245)
    dark = _png_bytes(5)
    crop = (10, 10, 790, 90)

    stages = {k: [] for k in [
        "image_decode_and_crop",
        "spectrum_extract",
        "reflectance",
        "preprocessing_pipeline",
        "ratios_peaks_integral",
        "db_write",
        "TOTAL",
    ]}

    for i in range(N_RUNS + WARMUP):
        t0 = time.perf_counter()
        s_img = Image.open(io.BytesIO(sample)).convert("RGB")
        w_img = Image.open(io.BytesIO(white)).convert("RGB")
        d_img = Image.open(io.BytesIO(dark)).convert("RGB")
        s_c = _crop(s_img, crop)
        w_c = _crop(w_img, crop)
        d_c = _crop(d_img, crop)
        t1 = time.perf_counter()

        s_lum = _luminance_spectrum(s_c)
        w_lum = _luminance_spectrum(w_c)
        d_lum = _luminance_spectrum(d_c)
        s_rgb = _rgb_spectrum(s_c)
        w_rgb = _rgb_spectrum(w_c)
        d_rgb = _rgb_spectrum(d_c)
        wls = pixels_to_wavelengths(s_c.size[0])
        t2 = time.perf_counter()

        R_raw = compute_reflectance(s_lum, w_lum, d_lum)
        t3 = time.perf_counter()

        pre = pp.run_preset_pipeline("general", wls, R_raw)
        R = pre["reflectance"]
        t4 = time.perf_counter()

        ratios = channel_ratios(s_rgb, w_rgb, d_rgb)
        peaks = peak_values(wls, R)
        integ = integrated_reflectance(wls, R)
        t5 = time.perf_counter()

        result = {
            "wavelengths_nm": wls.tolist(),
            "reflectance": R.tolist(),
            "first_derivative": pre["d1"].tolist(),
            "second_derivative": pre["d2"].tolist(),
            "ratios": ratios,
            "peaks": peaks,
            "integrated_reflectance": integ,
        }
        create_session_record(db, preset="general", concentration=None, result=result)
        t6 = time.perf_counter()

        if i >= WARMUP:
            stages["image_decode_and_crop"].append((t1 - t0) * 1000)
            stages["spectrum_extract"].append((t2 - t1) * 1000)
            stages["reflectance"].append((t3 - t2) * 1000)
            stages["preprocessing_pipeline"].append((t4 - t3) * 1000)
            stages["ratios_peaks_integral"].append((t5 - t4) * 1000)
            stages["db_write"].append((t6 - t5) * 1000)
            stages["TOTAL"].append((t6 - t0) * 1000)
    db.close()
    return stages


# ---------- reporting ----------

def _summary(label: str, vals: list[float]) -> tuple[float, float]:
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {label:<28} {m:>9.2f} ms   ± {s:>6.2f} ms")
    return m, s


def main_bench():
    print(f"Spectrometer backend runtime benchmark")
    print(f"  measured runs: {N_RUNS}    warm-up runs: {WARMUP}")
    print(f"  synthetic image size: {IMG_W} x {IMG_H} px (cropped to 780 x 80)\n")

    print("[1] End-to-end HTTP round-trip  (POST /analyse, general preset):")
    http_times = benchmark_http()
    http_m, http_s = _summary("total per request", http_times)

    print("\n[2] Per-stage internal timing  (HTTP layer bypassed):")
    stage_times = benchmark_stages()
    rows = []
    for k, vals in stage_times.items():
        m, s = _summary(k, vals)
        rows.append({"stage": k, "mean_ms": round(m, 3),
                     "std_ms": round(s, 3), "n": len(vals)})
    rows.append({"stage": "http_round_trip", "mean_ms": round(http_m, 3),
                 "std_ms": round(http_s, 3), "n": len(http_times)})

    out = Path(__file__).parent / "benchmark_results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage", "mean_ms", "std_ms", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {out.resolve()}")


if __name__ == "__main__":
    main_bench()
