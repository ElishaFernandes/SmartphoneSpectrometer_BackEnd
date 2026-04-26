import numpy as np
import analysis as an


def _synth_spectra(n=300):
    wl = np.linspace(380, 750, n)
    I_w = np.full(n, 1000.0)
    I_d = np.full(n, 50.0)
    I_s = 200 + 400 * np.exp(-((wl - 550) ** 2) / (2 * 30 ** 2))
    return wl, I_s, I_w, I_d


def test_reflectance_finite_and_clipped():
    wl, I_s, I_w, I_d = _synth_spectra()
    R = an.compute_reflectance(I_s, I_w, I_d)
    assert R.shape == I_s.shape
    assert np.all(np.isfinite(R))
    assert np.all(R >= -1.0) and np.all(R <= 2.0)


def test_reflectance_no_div_zero_when_white_equals_dark():
    n = 100
    I_s = np.linspace(50.0, 200.0, n)
    I_w = np.full(n, 100.0)
    I_d = np.full(n, 100.0)  # denominator zero
    R = an.compute_reflectance(I_s, I_w, I_d)
    assert np.all(np.isfinite(R))


def test_chlorophyll_classification_thresholds():
    assert an.chlorophyll_class(0.5) == "low"
    assert an.chlorophyll_class(1.0) == "low"   # boundary
    assert an.chlorophyll_class(1.2) == "moderate"
    assert an.chlorophyll_class(1.5) == "moderate"  # boundary
    assert an.chlorophyll_class(2.0) == "high"


def test_integrated_reflectance_known_box():
    # area under unit-height box of width 10 = 10
    wl = np.linspace(0, 10, 1000)
    R = np.ones_like(wl)
    assert abs(an.integrated_reflectance(wl, R) - 10.0) < 1e-6


def test_integrated_reflectance_in_band():
    # Trapezoidal integration on a discrete grid cannot hit band edges
    # exactly. Tolerance is set to ~2x step size to allow for the
    # boundary half-step at each end of the masked window.
    wl = np.linspace(0, 10, 1000)
    R = np.ones_like(wl)
    step = wl[1] - wl[0]
    val = an.integrated_reflectance(wl, R, band=(2, 5))
    assert abs(val - 3.0) < 2 * step


def test_peak_values_find_synthetic_peak():
    wl = np.linspace(380, 750, 300)
    R = np.exp(-((wl - 550) ** 2) / (2 * 20 ** 2))
    peaks = an.peak_values(wl, R)
    assert len(peaks) >= 1
    assert abs(peaks[0]["wavelength_nm"] - 550) < 5


def test_analyse_general_returns_required_fields():
    wl, I_s, I_w, I_d = _synth_spectra()
    rgb_s = np.stack([I_s, I_s, I_s])
    rgb_w = np.stack([I_w, I_w, I_w])
    rgb_d = np.stack([I_d, I_d, I_d])
    out = an.analyse_spectrum("general", wl, I_s, I_w, I_d, rgb_s, rgb_w, rgb_d)
    for k in ("wavelengths_nm", "reflectance", "first_derivative",
              "second_derivative", "ratios", "peaks", "integrated_reflectance"):
        assert k in out


def test_analyse_chlorophyll_returns_proxy():
    wl, I_s, I_w, I_d = _synth_spectra()
    rgb_s = np.stack([I_s, I_s * 1.2, I_s * 0.8])
    rgb_w = np.stack([I_w, I_w, I_w])
    rgb_d = np.stack([I_d, I_d, I_d])
    out = an.analyse_spectrum("chlorophyll", wl, I_s, I_w, I_d, rgb_s, rgb_w, rgb_d)
    assert "fractional_derivative_alpha_0_3" in out
    assert "chlorophyll_proxy" in out
    assert out["chlorophyll_proxy"]["level"] in {"low", "moderate", "high"}


def test_analyse_bilirubin_returns_band_features():
    wl, I_s, I_w, I_d = _synth_spectra()
    rgb_s = np.stack([I_s, I_s, I_s])
    rgb_w = np.stack([I_w, I_w, I_w])
    rgb_d = np.stack([I_d, I_d, I_d])
    out = an.analyse_spectrum("bilirubin", wl, I_s, I_w, I_d, rgb_s, rgb_w, rgb_d)
    assert "first_derivative" in out
    feats = out["bilirubin_features"]
    assert feats["depression_600_740_nm"]["available"] is True
    # 750-850 nm band exceeds visible range, expected unavailable
    assert feats["elevation_750_850_nm"]["available"] is False