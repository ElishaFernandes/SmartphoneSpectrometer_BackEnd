import os

os.environ["DATABASE_URL"] = "sqlite:///./test_compare.db"

import pytest

from database import init_db, SessionLocal, create_session_record
from comparison import compare_sessions, run_calibration


@pytest.fixture(scope="module", autouse=True)
def _setup_db():
    init_db()
    yield
    if os.path.exists("./test_compare.db"):
        os.remove("./test_compare.db")


@pytest.fixture
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _result(integrated, peak=0.5):
    return {
        "wavelengths_nm": [400.0, 500.0, 600.0],
        "reflectance": [0.1, 0.5, 0.3],
        "ratios": {"R_over_G": 0.6, "R_over_B": 0.5,
                   "G_over_B": 0.8, "G_over_R": 1.6},
        "peaks": [{"wavelength_nm": 500.0, "reflectance": peak}],
        "integrated_reflectance": integrated,
    }


def test_reject_fewer_than_two(db):
    sid = create_session_record(db, "general", 1.0, _result(50))
    with pytest.raises(ValueError):
        compare_sessions(db, [sid])


def test_compare_two_sessions(db):
    s1 = create_session_record(db, "general", 1.0, _result(50))
    s2 = create_session_record(db, "general", 2.0, _result(60))
    out = compare_sessions(db, [s1, s2])
    assert out["n_sessions"] == 2
    assert len(out["overlay"]) == 2
    assert "ratio_overlay" in out


def test_calibration_requires_concentrations(db):
    s1 = create_session_record(db, "general", None, _result(40))
    s2 = create_session_record(db, "general", 2.0, _result(80))
    with pytest.raises(ValueError):
        run_calibration(db, [s1, s2])  # only one labelled session


def test_calibration_returns_slope_intercept_r_squared(db):
    # perfect linear data: y = 0.02 * x + 0
    s1 = create_session_record(db, "general", 1.0, _result(50))
    s2 = create_session_record(db, "general", 2.0, _result(100))
    s3 = create_session_record(db, "general", 3.0, _result(150))
    out = run_calibration(db, [s1, s2, s3])
    assert "slope" in out and "intercept" in out and "r_squared" in out
    assert out["r_squared"] > 0.999
    assert out["n_points"] == 3


def test_calibration_with_peak_metric(db):
    # peaks must differ across sessions, otherwise scipy's linregress
    # correctly refuses (cannot fit a non-vertical line through identical x).
    s1 = create_session_record(db, "general", 1.0, _result(50, peak=0.3))
    s2 = create_session_record(db, "general", 2.0, _result(100, peak=0.6))
    out = run_calibration(db, [s1, s2], x_metric="peak_reflectance")
    assert "slope" in out