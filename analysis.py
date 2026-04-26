"""
Core analytical calculations.

- Reflectance R(λ) = (I_sample - I_dark) / (I_white - I_dark) (luminance-based)
- 1st and 2nd derivatives via Savitzky-Golay
- R/G, R/B, G/B, G/R band ratios from per-channel reflectance
- Chlorophyll proxy classification (low / moderate / high) from G/R ratio
- Peak detection (scipy.find_peaks with prominence)
- Integrated reflectance (trapezoidal)
- Bilirubin band stats: 600-740 nm depression, 750-850 nm elevation when present
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

import preprocessing as pp


# CONFIG: Chlorophyll proxy thresholds on the visible-band G/R ratio.
# These are PLACEHOLDERS. Replace with experimentally calibrated cutoffs
# (e.g. against ECOSIS LOPEX93 chlorophyll content measurements) before
# any quantitative interpretation.
CHLOROPHYLL_THRESHOLDS = {
    "low_max": 1.0,       # G/R <= 1.0      -> low
    "moderate_max": 1.5,  # 1.0 < G/R <= 1.5 -> moderate
                          # G/R > 1.5        -> high
}

CHLOROPHYLL_NOTE = (
    "Placeholder thresholds for prototype demonstration only. "
    "Replace with experimentally calibrated values before quantitative use."
)

# Bilirubin spectral feature bands. The elevation band sits in NIR; on a
# visible-only spectrometer it will be reported as unavailable.
BILIRUBIN_DEPRESSION_BAND = (600.0, 740.0)
BILIRUBIN_ELEVATION_BAND = (750.0, 850.0)

EPS = 1e-9


# ---------- reflectance and derivatives ----------

def compute_reflectance(I_sample: np.ndarray,
                        I_white: np.ndarray,
                        I_dark: np.ndarray) -> np.ndarray:
    """
    Reflectance with epsilon-guarded denominator and physical clipping.
    Returns finite values even when white == dark (degenerate denominator).
    """
    I_sample = np.asarray(I_sample, dtype=np.float64)
    I_white = np.asarray(I_white, dtype=np.float64)
    I_dark = np.asarray(I_dark, dtype=np.float64)
    num = I_sample - I_dark
    den = I_white - I_dark
    safe_den = np.where(np.abs(den) < EPS, EPS, den)
    R = num / safe_den
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    # physical reflectance is in [0, 1]; clip slightly wider for numerical noise
    return np.clip(R, -1.0, 2.0)


def first_derivative(R: np.ndarray) -> np.ndarray:
    return pp.sgd(R, order=1)


def second_derivative(R: np.ndarray) -> np.ndarray:
    return pp.sgd(R, order=2)


# ---------- channel ratios ----------

def channel_ratios(sample_rgb: np.ndarray,
                   white_rgb: np.ndarray,
                   dark_rgb: np.ndarray) -> dict:
    """
    Compute per-channel mean reflectance and the ratios R/G, R/B, G/B, G/R.
    Inputs are arrays of shape (3, n_pixels) for R, G, B respectively.
    """
    R_R = compute_reflectance(sample_rgb[0], white_rgb[0], dark_rgb[0]).mean()
    R_G = compute_reflectance(sample_rgb[1], white_rgb[1], dark_rgb[1]).mean()
    R_B = compute_reflectance(sample_rgb[2], white_rgb[2], dark_rgb[2]).mean()

    def _safe_div(a, b):
        return float(a / b) if abs(b) > EPS else 0.0

    return {
        "R_mean": float(R_R),
        "G_mean": float(R_G),
        "B_mean": float(R_B),
        "R_over_G": _safe_div(R_R, R_G),
        "R_over_B": _safe_div(R_R, R_B),
        "G_over_B": _safe_div(R_G, R_B),
        "G_over_R": _safe_div(R_G, R_R),  # chlorophyll proxy
    }


def chlorophyll_class(g_over_r: float) -> str:
    if g_over_r <= CHLOROPHYLL_THRESHOLDS["low_max"]:
        return "low"
    if g_over_r <= CHLOROPHYLL_THRESHOLDS["moderate_max"]:
        return "moderate"
    return "high"


# ---------- peaks and integration ----------

def peak_values(wavelengths: np.ndarray, R: np.ndarray, n_top: int = 5) -> list:
    """Top-n prominent peaks by prominence, each with wavelength and reflectance."""
    R = np.asarray(R)
    if len(R) < 3:
        return []
    peaks, props = find_peaks(R, prominence=0.005)
    if len(peaks) == 0:
        return []
    proms = props["prominences"]
    order = np.argsort(proms)[::-1][:n_top]
    return [
        {
            "wavelength_nm": float(wavelengths[peaks[i]]),
            "reflectance": float(R[peaks[i]]),
            "prominence": float(proms[i]),
        }
        for i in order
    ]


def integrated_reflectance(wavelengths: np.ndarray, R: np.ndarray,
                           band: tuple | None = None) -> float:
    """Trapezoidal integral of reflectance, optionally over a wavelength band."""
    wavelengths = np.asarray(wavelengths)
    R = np.asarray(R)
    if band is None:
        return float(trapezoid(R, wavelengths))
    lo, hi = band
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    if mask.sum() < 2:
        return 0.0
    return float(trapezoid(R[mask], wavelengths[mask]))


def _band_stats(wavelengths: np.ndarray, R: np.ndarray, band: tuple) -> dict:
    """Reflectance statistics inside a wavelength band; reports availability."""
    lo, hi = band
    mask = (wavelengths >= lo) & (wavelengths <= hi)
    if mask.sum() < 2:
        return {"available": False, "band_nm": list(band)}
    Rb = R[mask]
    return {
        "available": True,
        "band_nm": list(band),
        "mean_reflectance": float(Rb.mean()),
        "min_reflectance": float(Rb.min()),
        "max_reflectance": float(Rb.max()),
        "integrated": float(trapezoid(Rb, wavelengths[mask])),
    }


# ---------- top-level orchestrator ----------

def analyse_spectrum(preset: str,
                     wavelengths: np.ndarray,
                     sample_lum: np.ndarray,
                     white_lum: np.ndarray,
                     dark_lum: np.ndarray,
                     sample_rgb: np.ndarray,
                     white_rgb: np.ndarray,
                     dark_rgb: np.ndarray) -> dict:
    """
    Full per-preset analysis. Returns a JSON-serialisable dict suitable for
    storing and for sending to the Flutter frontend.
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)

    # 1. raw reflectance from luminance
    R_raw = compute_reflectance(sample_lum, white_lum, dark_lum)

    # 2. preset preprocessing on the reflectance curve
    pre = pp.run_preset_pipeline(preset, wavelengths, R_raw)
    R = pre["reflectance"]

    # 3. ratios from raw RGB intensities (not preprocessed)
    ratios = channel_ratios(sample_rgb, white_rgb, dark_rgb)

    out = {
        "wavelengths_nm": wavelengths.tolist(),
        "reflectance": R.tolist(),
        "ratios": ratios,
        "peaks": peak_values(wavelengths, R),
        "integrated_reflectance": integrated_reflectance(wavelengths, R),
    }

    if preset == "general":
        out["first_derivative"] = pre["d1"].tolist()
        out["second_derivative"] = pre["d2"].tolist()

    elif preset == "chlorophyll":
        out["fractional_derivative_alpha_0_3"] = pre["fractional_d_0_3"].tolist()
        out["chlorophyll_proxy"] = {
            "g_over_r": ratios["G_over_R"],
            "level": chlorophyll_class(ratios["G_over_R"]),
            "thresholds": CHLOROPHYLL_THRESHOLDS,
            "note": CHLOROPHYLL_NOTE,
        }

    elif preset == "bilirubin":
        out["first_derivative"] = pre["d1"].tolist()
        out["bilirubin_features"] = {
            "depression_600_740_nm": _band_stats(wavelengths, R, BILIRUBIN_DEPRESSION_BAND),
            "elevation_750_850_nm": _band_stats(wavelengths, R, BILIRUBIN_ELEVATION_BAND),
        }

    return out