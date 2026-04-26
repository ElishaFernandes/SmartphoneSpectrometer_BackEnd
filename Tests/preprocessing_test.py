import numpy as np
import preprocessing as pp


def test_simple_filters_preserve_length_and_finite():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 100) + 0.05 * rng.standard_normal(100)
    for fn in [pp.maf, pp.mpf, pp.snv, pp.min_max, pp.z_score, pp.sgf, pp.wiener_filter]:
        y = fn(x)
        assert y.shape == x.shape, f"{fn.__name__} changed shape"
        assert np.all(np.isfinite(y)), f"{fn.__name__} produced non-finite values"


def test_derivatives_preserve_length():
    x = np.sin(np.linspace(0, 6.28, 200))
    assert pp.sgd(x, order=1).shape == x.shape
    assert pp.sgd(x, order=2).shape == x.shape


def test_general_pipeline_outputs():
    R = np.linspace(0.1, 0.9, 256)
    out = pp.run_preset_pipeline("general", np.linspace(380, 750, 256), R)
    assert set(out.keys()) == {"reflectance", "d1", "d2"}
    for v in out.values():
        assert v.shape == R.shape
        assert np.all(np.isfinite(v))


def test_chlorophyll_pipeline_outputs():
    R = np.linspace(0.1, 0.9, 256)
    out = pp.run_preset_pipeline("chlorophyll", np.linspace(380, 750, 256), R)
    assert set(out.keys()) == {"reflectance", "fractional_d_0_3"}
    for v in out.values():
        assert v.shape == R.shape


def test_bilirubin_pipeline_outputs():
    R = np.linspace(0.1, 0.9, 256)
    out = pp.run_preset_pipeline("bilirubin", np.linspace(380, 750, 256), R)
    assert set(out.keys()) == {"reflectance", "d1"}
    for v in out.values():
        assert v.shape == R.shape


def test_unknown_preset_raises():
    R = np.linspace(0.1, 0.9, 64)
    try:
        pp.run_preset_pipeline("ultraviolet", np.linspace(380, 750, 64), R)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown preset")


def test_pipelines_handle_short_input():
    R = np.linspace(0.0, 1.0, 8)  # shorter than default SGF window
    for preset in ("general", "chlorophyll", "bilirubin"):
        out = pp.run_preset_pipeline(preset, np.linspace(380, 750, 8), R)
        for v in out.values():
            assert v.shape == R.shape
            assert np.all(np.isfinite(v))