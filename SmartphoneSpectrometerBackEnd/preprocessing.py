"""
Preprocessing pipelines per preset.

Simple filters (MAF, MPF, SNV, MSC, min-max, Z-score, SGF, SGD, Wiener) are
implemented properly via numpy/scipy. Advanced baseline-correction methods
(airPLS, arPLS) and the Grünwald-Letnikov fractional derivative (GL-FOD) are
PLACEHOLDERS that preserve array shape so the pipeline runs end-to-end.
Replace them with full implementations before quantitative interpretation.

Each preset pipeline operates on a single signal (the reflectance curve) and
returns a dict with the smoothed reflectance plus any derivatives the preset
specifies.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter, wiener, medfilt


# CONFIG: filter parameters tuned for typical Samsung S24+ class crop widths
# (~256-1024 px). Adjustable per device.
SGF_WINDOW = 11
SGF_POLYORDER = 3
MAF_WINDOW = 5
MPF_WINDOW = 5
WIENER_WINDOW = 5


# ===========================================================================
# Simple filters - real implementations
# ===========================================================================

def maf(x: np.ndarray, w: int = MAF_WINDOW) -> np.ndarray:
    """Moving average filter with same-length output."""
    if w < 2:
        return x.copy()
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def mpf(x: np.ndarray, w: int = MPF_WINDOW) -> np.ndarray:
    """Median filter (window forced to odd)."""
    if w % 2 == 0:
        w += 1
    return medfilt(x, kernel_size=w)


def snv(x: np.ndarray) -> np.ndarray:
    """Standard Normal Variate normalisation."""
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-12:
        return x - mu
    return (x - mu) / sd


def msc(x: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """
    Multiplicative Scatter Correction.
    Uses self-mean as reference when none is provided (degenerate but
    keeps the pipeline runnable). For real use, pass a population mean.
    """
    if ref is None:
        ref = np.full_like(x, float(np.mean(x)))
    if np.std(ref) < 1e-12:
        return x.copy()
    a, b = np.polyfit(ref, x, 1)
    if abs(a) < 1e-12:
        return x.copy()
    return (x - b) / a


def min_max(x: np.ndarray) -> np.ndarray:
    """Min-max normalisation to [0, 1]."""
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def z_score(x: np.ndarray) -> np.ndarray:
    """Z-score normalisation (mean 0, std 1)."""
    return snv(x)


def sgf(x: np.ndarray, window: int = SGF_WINDOW, polyorder: int = SGF_POLYORDER) -> np.ndarray:
    """Savitzky-Golay smoothing."""
    if len(x) <= polyorder:
        return x.copy()
    if window > len(x):
        window = len(x) if len(x) % 2 == 1 else len(x) - 1
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        polyorder = max(1, window - 1)
    return savgol_filter(x, window_length=window, polyorder=polyorder)


def sgd(x: np.ndarray, order: int = 1, window: int = SGF_WINDOW, polyorder: int = SGF_POLYORDER) -> np.ndarray:
    """Savitzky-Golay derivative of given order."""
    if len(x) <= polyorder:
        if order == 1:
            return np.gradient(x)
        return np.gradient(np.gradient(x))
    if window > len(x):
        window = len(x) if len(x) % 2 == 1 else len(x) - 1
    if window % 2 == 0:
        window += 1
    if window <= polyorder:
        polyorder = max(order, window - 1)
    return savgol_filter(x, window_length=window, polyorder=polyorder, deriv=order)


def wiener_filter(x: np.ndarray, window: int = WIENER_WINDOW) -> np.ndarray:
    """Wiener filter (scipy.signal.wiener)."""
    return wiener(x, mysize=window)


# ===========================================================================
# Placeholder advanced methods - shape-preserving stubs
# ===========================================================================

def airpls(x: np.ndarray, lam: float = 1e6, order: int = 1, max_iter: int = 15) -> np.ndarray:
    """
    PLACEHOLDER: airPLS adaptive iteratively reweighted penalised least squares.
    Stub: subtract the mean as a crude baseline removal so the pipeline runs.
    TODO: replace with full airPLS implementation.
    """
    return x - np.mean(x)


def arpls(x: np.ndarray, lam: float = 1e5, ratio: float = 0.05, max_iter: int = 50) -> np.ndarray:
    """
    PLACEHOLDER: asymmetrically reweighted penalised least squares.
    Stub: linear-detrend so the pipeline runs.
    TODO: replace with full arPLS implementation.
    """
    n = len(x)
    if n < 2:
        return x.copy()
    t = np.arange(n)
    a, b = np.polyfit(t, x, 1)
    return x - (a * t + b)


def gl_fod(x: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    PLACEHOLDER: Grünwald-Letnikov fractional-order derivative.
    Stub: convex blend between the signal and its 1st derivative.
    TODO: replace with proper GL coefficients (binomial-series formulation).
    """
    d1 = np.gradient(x)
    return (1 - alpha) * x + alpha * d1


# ===========================================================================
# Preset pipelines - operate on the reflectance curve R(λ)
# ===========================================================================

def _general_pipe(R: np.ndarray) -> dict:
    """MAF -> airPLS -> SNV -> min-max -> SGF -> SGD 1st + 2nd"""
    s = maf(R)
    s = airpls(s)
    s = snv(s)
    s = min_max(s)
    smoothed = sgf(s)
    return {
        "reflectance": smoothed,
        "d1": sgd(smoothed, order=1),
        "d2": sgd(smoothed, order=2),
    }


def _chlorophyll_pipe(R: np.ndarray) -> dict:
    """MPF -> airPLS -> SNV -> SGF -> GL-FOD α=0.3"""
    s = mpf(R)
    s = airpls(s)
    s = snv(s)
    smoothed = sgf(s)
    return {
        "reflectance": smoothed,
        "fractional_d_0_3": gl_fod(smoothed, alpha=0.3),
    }


def _bilirubin_pipe(R: np.ndarray) -> dict:
    """MAF -> arPLS -> SNV + MSC -> Wiener -> Z-score -> SGF -> SGD 1st"""
    s = maf(R)
    s = arpls(s)
    s = snv(s)
    s = msc(s)
    s = wiener_filter(s)
    s = z_score(s)
    smoothed = sgf(s)
    return {
        "reflectance": smoothed,
        "d1": sgd(smoothed, order=1),
    }


def run_preset_pipeline(preset: str, wavelengths: np.ndarray, reflectance: np.ndarray) -> dict:
    """Dispatch to the correct preset pipeline. Wavelengths kept in signature for future use."""
    pipes = {
        "general": _general_pipe,
        "chlorophyll": _chlorophyll_pipe,
        "bilirubin": _bilirubin_pipe,
    }
    if preset not in pipes:
        raise ValueError(f"Unknown preset: {preset}")
    out = pipes[preset](np.asarray(reflectance, dtype=np.float64))
    # safety: replace any NaN / inf with zeros so downstream code is finite
    for k, v in out.items():
        out[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return out