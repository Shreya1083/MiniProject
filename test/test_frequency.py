"""Automated tests for the FFT frequency-analysis module (Phase 2).

Run with:  .venv/Scripts/python.exe -m pytest test/test_frequency.py -v

These tests cover BOTH valid and invalid inputs as required by the project
plan.  Invalid inputs (None, wrong type, wrong shape, empty array, missing
file, corrupt file) must raise ``FrequencyError`` and never crash the program.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make ``src`` importable when the test is run from the project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.frequency import (  # noqa: E402
    FrequencyError,
    analyze_image,
    analyze_image_file,
    compute_magnitude_spectrum,
    normalize_for_display,
    to_grayscale,
)


# --------------------------------------------------------------------------- #
# Fixtures: build small, deterministic images so tests do not depend on the
# internet or large datasets.
# --------------------------------------------------------------------------- #
@pytest.fixture
def synthetic_rgb() -> np.ndarray:
    """A 64x64 RGB image with a clear pattern (not random noise)."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    # Add a smooth gradient + a sharp block so the spectrum has real content.
    arr[:, :, 0] = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    arr[10:20, 10:20, 1] = 255  # sharp bright block -> high frequencies
    return arr


@pytest.fixture
def synthetic_file(tmp_path: Path, synthetic_rgb: np.ndarray) -> Path:
    """Save the synthetic image to a temp PNG the test can read back."""
    p = tmp_path / "valid.png"
    Image.fromarray(synthetic_rgb, mode="RGB").save(p)
    return p


# --------------------------------------------------------------------------- #
# Valid inputs
# --------------------------------------------------------------------------- #
def test_analyze_image_returns_expected_keys(synthetic_rgb: np.ndarray):
    result = analyze_image(synthetic_rgb)
    assert set(result.keys()) == {
        "grayscale",
        "magnitude_spectrum",
        "spectrum_display",
        "shape",
    }


def test_grayscale_is_2d_uint8(synthetic_rgb: np.ndarray):
    gray = to_grayscale(synthetic_rgb)
    assert gray.ndim == 2
    assert gray.dtype == np.uint8
    assert gray.shape == (64, 64)


def test_magnitude_spectrum_shape_dtype_and_range(synthetic_rgb: np.ndarray):
    gray = to_grayscale(synthetic_rgb)
    spec = compute_magnitude_spectrum(gray)
    assert spec.dtype == np.float32
    assert spec.shape == gray.shape
    assert np.all(np.isfinite(spec))
    # log(1 + |F|) is always >= 0.
    assert spec.min() >= 0.0


def test_spectrum_display_is_uint8_bounded(synthetic_rgb: np.ndarray):
    disp = analyze_image(synthetic_rgb)["spectrum_display"]
    assert disp.dtype == np.uint8
    assert disp.ndim == 2
    assert 0 <= disp.min() and disp.max() <= 255


def test_analyze_image_file_round_trip(synthetic_file: Path):
    result = analyze_image_file(synthetic_file)
    assert result["shape"] == (64, 64)
    assert result["grayscale"].shape == (64, 64)


def test_fft_centers_dc_component(synthetic_rgb: np.ndarray):
    """For a non-negative image the DC term (after fftshift) is the maximum.

    Math: |F[0,0]| = sum of all pixels >= |F[k]| for any k, because every
    pixel is >= 0 and only the DC term adds them all in phase.  log1p is
    monotonic, so the centre stays the largest after the log transform.
    """
    gray = to_grayscale(synthetic_rgb)
    spec = compute_magnitude_spectrum(gray)
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    assert spec[cy, cx] == spec.max(), "Centered DC should be the spectrum peak"


def test_constant_image_concentrates_energy_in_dc():
    """A solid-colour image has only a DC component: one bright centre pixel."""
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    disp = analyze_image(img)["spectrum_display"]
    assert np.count_nonzero(disp) == 1
    h, w = disp.shape
    assert disp[h // 2, w // 2] == 255


def test_normalize_flat_spectrum_returns_black():
    """A completely flat spectrum (no range) must normalise to all-black."""
    flat = np.full((8, 8), 5.0, dtype=np.float32)
    disp = normalize_for_display(flat)
    assert disp.dtype == np.uint8
    assert disp.max() == 0


# --------------------------------------------------------------------------- #
# Invalid inputs (must raise FrequencyError, never crash)
# --------------------------------------------------------------------------- #
def test_none_input_raises():
    with pytest.raises(FrequencyError):
        analyze_image(None)


def test_non_array_input_raises():
    with pytest.raises(FrequencyError):
        analyze_image("this is not an image")  # str is not an ndarray


def test_one_dimensional_shape_raises():
    with pytest.raises(FrequencyError):
        to_grayscale(np.zeros(10, dtype=np.uint8))


def test_empty_array_raises():
    with pytest.raises(FrequencyError):
        analyze_image(np.zeros((0, 0, 3), dtype=np.uint8))


def test_missing_file_raises():
    with pytest.raises(FrequencyError):
        analyze_image_file("this_file_does_not_exist_xyz.png")


def test_corrupt_file_raises():
    """assets/corrupt.png is a 20-byte text file ('this is not an image').

    cv2.imread returns None for it, so the module must raise FrequencyError.
    """
    p = ROOT / "assets" / "corrupt.png"
    if not p.exists():
        pytest.skip("assets/corrupt.png not present in this checkout")
    with pytest.raises(FrequencyError):
        analyze_image_file(p)
