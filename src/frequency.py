"""Reusable FFT frequency-analysis module for deepfake detection.

WHAT THIS MODULE DOES
- Converts an image to grayscale.
- Computes the 2D Fast Fourier Transform (FFT) of that grayscale image.
- Centres the spectrum with ``fftshift`` so low frequencies sit in the middle.
- Takes a logarithm of the magnitude so we can see faint high-frequency detail.
- Normalises the result to 0-255 so it can be shown as an image.

WHAT THIS MODULE DOES *NOT* DO
- It does NOT feed frequency features into the deepfake classifier.
- It is a *visualisation* tool only. Showing an FFT spectrum and *classifying
  with* frequency features are two different things. We are honest about this:
  until Phase 5 fusion is actually implemented and verified, the detector stays
  an RGB-only classifier and this module is purely for human inspection.

FFT IN ONE PARAGRAPH (viva-friendly)
A normal image is described by how bright each pixel is (the "spatial" view).
The Fourier transform re-describes the same image by the *frequencies* it
contains: slow, smooth changes are "low frequency" and appear in the centre of
the spectrum; sharp edges, fine textures and compression artefacts are "high
frequency" and appear towards the edges. Deepfakes often smooth or blur faces,
which removes high frequencies, so their spectrum can look "cleaner" than a real
photo's. Seeing the spectrum helps a human understand what the model might be
reacting to, but it is not the same as the model using it for classification.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# Use a non-interactive backend so figures can be built on a headless server
# (and inside Streamlit) without trying to open a GUI window.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Smallest range we treat as "flat" so we never divide by zero when normalising.
_EPS = 1e-8


class FrequencyError(Exception):
    """Raised when frequency analysis cannot be performed on an input."""


# --------------------------------------------------------------------------- #
# Core steps
# --------------------------------------------------------------------------- #
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a 2-D uint8 grayscale version of an RGB/RGBA/gray image.

    Accepts a single image as a numpy array.  ``image`` is assumed to be in RGB
    order (which is what ``src.detector.read_rgb_image`` already produces).
    """
    if image is None:
        raise FrequencyError("No image was provided to grayscale conversion.")
    if not isinstance(image, np.ndarray):
        raise FrequencyError(
            f"Expected a numpy array, got {type(image).__name__}."
        )
    if image.size == 0:
        raise FrequencyError("The image array is empty.")

    # Normalise dtype: cvtColor wants uint8. If a float array arrives in 0-255
    # range we convert it; if it is already 0-1 we scale it up first.
    arr = image
    if arr.dtype != np.uint8:
        if np.isfinite(arr).min() >= 0 and np.isfinite(arr).max() <= 1.0 + _EPS:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    raise FrequencyError(
        f"Unsupported image shape {arr.shape!r}; expected 2-D or (H, W, 3/4)."
    )


def compute_magnitude_spectrum(gray: np.ndarray) -> np.ndarray:
    """Compute the *centred log-magnitude* FFT spectrum of a grayscale image.

    Steps:
      1. ``fft2``  - 2D Fourier transform.
      2. ``fftshift`` - move zero-frequency (DC) component to the centre.
      3. ``abs``   - magnitude (drops phase information we do not visualise).
      4. ``log1p`` - log(1 + magnitude) so faint high frequencies become visible.

    Returns a float32 array of the same (H, W) shape, un-normalised.
    """
    if gray is None:
        raise FrequencyError("No grayscale image was provided.")
    if not isinstance(gray, np.ndarray):
        raise FrequencyError("Grayscale input must be a numpy array.")
    if gray.ndim != 2:
        raise FrequencyError(
            f"Grayscale image must be 2-D, got shape {gray.shape!r}."
        )
    if gray.size == 0:
        raise FrequencyError("Grayscale image is empty.")

    # float32 keeps memory reasonable; fft2 works in complex128 internally.
    f = np.fft.fft2(gray.astype(np.float32))
    f_shifted = np.fft.fftshift(f)          # centre low frequencies
    magnitude = np.abs(f_shifted)          # |F(u,v)|
    log_magnitude = np.log1p(magnitude)    # log(1 + |F|)  -> always >= 0
    return log_magnitude.astype(np.float32)


def normalize_for_display(spectrum: np.ndarray) -> np.ndarray:
    """Scale a float spectrum to an 8-bit (0-255) image for display.

    A flat (constant) input would produce a near-zero spectrum; we guard
    against dividing by zero and return a black image in that case.
    """
    if spectrum is None:
        raise FrequencyError("No spectrum was provided for normalisation.")
    spec = np.asarray(spectrum, dtype=np.float32)
    if spec.size == 0:
        raise FrequencyError("Spectrum is empty.")

    mn = float(spec.min())
    mx = float(spec.max())
    rng = mx - mn
    if rng < _EPS:
        # Flat spectrum (e.g. a solid-colour image) -> all black.
        return np.zeros_like(spec, dtype=np.uint8)

    norm = (spec - mn) / rng            # 0.0 .. 1.0
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def analyze_image(image: np.ndarray) -> Dict[str, np.ndarray | Tuple[int, int]]:
    """Full pipeline on an in-memory RGB/gray image.

    Returns a dict with:
      - ``grayscale``         : 2-D uint8
      - ``magnitude_spectrum`` : float32 log-magnitude (raw, for any later use)
      - ``spectrum_display``  : 2-D uint8 (0-255) ready to show
      - ``shape``             : (height, width) of the grayscale image
    """
    gray = to_grayscale(image)
    spectrum = compute_magnitude_spectrum(gray)
    display = normalize_for_display(spectrum)
    return {
        "grayscale": gray,
        "magnitude_spectrum": spectrum,
        "spectrum_display": display,
        "shape": (int(gray.shape[0]), int(gray.shape[1])),
    }


def analyze_image_file(image_path: str | Path) -> dict:
    """Read an image file from disk and run :func:`analyze_image` on it.

    Raises :class:`FrequencyError` for missing files, unreadable/corrupt
    images, or unsupported channel counts.
    """
    path = Path(image_path)
    if not path.exists():
        raise FrequencyError(f"Image file not found: {path}")

    image = cv2.imread(str(path))          # BGR uint8 or None
    if image is None:
        raise FrequencyError(
            "Could not read the image. Use a valid JPG or PNG file."
        )
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return analyze_image(rgb)


# --------------------------------------------------------------------------- #
# Visualisation helper (used by the Streamlit app)
# --------------------------------------------------------------------------- #
def make_spectrum_figure(
    rgb_image: np.ndarray, analysis: dict
) -> plt.Figure:
    """Build a 3-panel matplotlib figure: RGB original, grayscale, FFT spectrum.

    The figure is returned (not shown) so the caller can hand it to
    ``streamlit.pyplot``.  This keeps all FFT rendering in one module.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(analysis["grayscale"], cmap="gray")
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    # The spectrum: bright centre = low frequencies, darker edges = high
    # frequencies. cmap="gray" maps 0->black, 255->white.
    axes[2].imshow(analysis["spectrum_display"], cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("FFT magnitude spectrum (log, centred)")
    axes[2].axis("off")

    fig.tight_layout()
    return fig
