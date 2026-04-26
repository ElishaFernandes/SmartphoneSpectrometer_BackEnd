import pytest
from fastapi import HTTPException

from input import (
    validate_preset, validate_concentration,
    pixels_to_wavelengths, ALLOWED_PRESETS,
)


def test_valid_presets_accepted():
    for p in ALLOWED_PRESETS:
        assert validate_preset(p) == p
        assert validate_preset(p.upper()) == p  # case-insensitive


def test_invalid_preset_rejected():
    with pytest.raises(HTTPException) as exc:
        validate_preset("ultraviolet")
    assert exc.value.status_code == 400


def test_concentration_numeric_accepted():
    assert validate_concentration(1.5) == 1.5
    assert validate_concentration("2.3") == 2.3
    assert validate_concentration(None) is None
    assert validate_concentration("") is None


def test_concentration_text_rejected():
    with pytest.raises(HTTPException) as exc:
        validate_concentration("not a number")
    assert exc.value.status_code == 400


def test_pixels_to_wavelengths_visible_range():
    wl = pixels_to_wavelengths(400)
    assert wl[0] == 380.0
    assert wl[-1] == 750.0
    assert len(wl) == 400


def test_pixels_to_wavelengths_too_short():
    with pytest.raises(HTTPException):
        pixels_to_wavelengths(1)