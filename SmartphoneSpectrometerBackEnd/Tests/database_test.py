import os

# point at an isolated test DB before importing the module
os.environ["DATABASE_URL"] = "sqlite:///./test_spectrometer.db"

import pytest

from database import (
    init_db, SessionLocal,
    create_session_record, get_session_record,
    list_session_records, delete_session_record,
)


@pytest.fixture(scope="module", autouse=True)
def _setup_db():
    init_db()
    yield
    if os.path.exists("./test_spectrometer.db"):
        os.remove("./test_spectrometer.db")


@pytest.fixture
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _fake_result():
    return {
        "wavelengths_nm": [400.0, 500.0, 600.0],
        "reflectance": [0.1, 0.5, 0.3],
        "first_derivative": [0.0, 0.4, -0.2],
        "second_derivative": [0.0, 0.0, -0.6],
        "ratios": {"R_over_G": 0.6, "G_over_R": 1.6},
        "peaks": [{"wavelength_nm": 500.0, "reflectance": 0.5, "prominence": 0.4}],
        "integrated_reflectance": 90.0,
    }


def test_create_and_get(db):
    sid = create_session_record(db, "general", 1.5, _fake_result())
    rec = get_session_record(db, sid)
    assert rec is not None
    assert rec["preset"] == "general"
    assert rec["concentration"] == 1.5
    assert rec["wavelength_data"] == [400.0, 500.0, 600.0]
    assert rec["reflectance_data"] == [0.1, 0.5, 0.3]


def test_concentration_can_be_null(db):
    sid = create_session_record(db, "chlorophyll", None, _fake_result())
    rec = get_session_record(db, sid)
    assert rec["concentration"] is None


def test_list_and_delete(db):
    sid = create_session_record(db, "bilirubin", 2.0, _fake_result())
    all_recs = list_session_records(db)
    assert any(r["id"] == sid for r in all_recs)
    assert delete_session_record(db, sid) is True
    assert get_session_record(db, sid) is None
    assert delete_session_record(db, "nonexistent-id") is False


def test_arrays_are_recoverable_intact(db):
    sid = create_session_record(db, "general", 1.0, _fake_result())
    rec = get_session_record(db, sid)
    assert rec["wavelength_data"] == [400.0, 500.0, 600.0]
    assert rec["reflectance_data"] == [0.1, 0.5, 0.3]
    assert rec["derivative_data"] == [0.0, 0.4, -0.2]
    assert rec["second_derivative_data"] == [0.0, 0.0, -0.6]
