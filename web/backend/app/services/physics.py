"""
Thin FastAPI-facing wrapper around eyesim's physics core (eyesim.optics).

No optical math lives here — every computation delegates to the eyesim
package so the web layer and the offline simulation/training pipeline stay
provably consistent.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image

from eyesim.optics import blur_image, calibrated_psf_for_image
from eyesim.optimize import sharpness
from eyesim.predistort import wiener_deconvolve

_DEMO_SIZE = 160

# calibrated_psf_for_image sizes the kernel against a real geometric blur
# angle under an assumed field of view (see that function for the field-of-view assumption; formerly 25 deg, a photo viewed on
# a phone/laptop at arm's length) instead of an arbitrary pixel fraction.
# This used to be reimplemented here as a private _psf_for_image; it's now
# shared with eyesim.pipeline and the example scripts, all of which had the
# same bug independently (e.g. -2.25 D spanning ~half of a 256px image) —
# see eyesim/optics.py::calibrated_psf_for_image's docstring.


def _demo_scene(n: int = _DEMO_SIZE, spokes: int = 20) -> np.ndarray:
    """Siemens star test target: isotropic edge content at every orientation.
    Mirrors examples/full_pipeline.py::make_siemens_star."""
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x)
    phi = np.arctan2(yy, xx)
    r = np.sqrt(xx**2 + yy**2)
    star = 0.5 + 0.5 * np.sign(np.sin(spokes * phi))
    star[r > 0.95] = 1.0
    star[r < 0.05] = 0.0
    return star.astype(np.float32)


_SCENE = _demo_scene()


def png_data_url(arr: np.ndarray) -> str:
    """Encode a float [0,1] grayscale array as a base64 PNG data: URL."""
    img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype("uint8"), mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def psf_bench(S: float, C: float, axis: float, pupil_r_mm: float = 1.5) -> dict:
    """Live-slider bench: PSF heatmap + blurred demo-scene preview + sharpness,
    with the kernel scaled against the demo scene (see
    eyesim.optics.calibrated_psf_for_image)."""
    # The demo star is a small test target, not a photo of a scene: treat its
    # 160 px as 8 deg of view (3 arcmin per pixel), so blur stays visible.
    psf = calibrated_psf_for_image(S, C, axis, _SCENE.shape, pupil_radius_mm=pupil_r_mm,
                                   assumed_fov_deg=8.0)
    blurred = blur_image(_SCENE, psf)
    psf_norm = psf / psf.max()
    return {
        "psf": np.round(psf_norm, 4).tolist(),
        "preview_png": png_data_url(blurred),
        "sharpness": sharpness(blurred),
        "psf_crop": psf.shape[0],
    }


_SHADER_KERNEL_MAX = 24  # capped for GPU cost in the live fullscreen shader


def _downsample_kernel(psf: np.ndarray, max_size: int) -> np.ndarray:
    """Bin a PSF down to at most max_size x max_size for the client-side
    WebGL shader. The *persisted* blurred/predistorted images always use
    the full-fidelity kernel (scipy fftconvolve) — only the interactive
    shader approximation is size-capped."""
    n = psf.shape[0]
    if n <= max_size:
        return psf
    factor = int(np.ceil(n / max_size))
    pad = (-n) % factor
    if pad:
        psf = np.pad(psf, ((0, pad), (0, pad)))
    n2 = psf.shape[0]
    binned = psf.reshape(n2 // factor, factor, n2 // factor, factor).sum(axis=(1, 3))
    return binned


def _srgb_to_linear(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def simulate_case(image: np.ndarray, S: float, C: float, axis: float,
                  pupil_r_mm: float = 1.5) -> dict:
    """
    Simulate-mode physics for one uploaded photo: the eye's blur, the
    Wiener-predistorted "corrected" version, and a small kernel for the
    client-side screen-filter shader.

    The blur kernel is sized in image pixels via
    eyesim.optics.calibrated_psf_for_image, which converts the real
    geometric blur-circle diameter into pixels under an explicit assumed
    field-of-view — not an arbitrary fraction of the image, and not capped
    in a way that silently truncates high powers.
    """
    h, w = image.shape[:2]
    psf = calibrated_psf_for_image(S, C, axis, (h, w), pupil_radius_mm=pupil_r_mm)

    # Optics act on light intensity, so blur in linear light and re-encode for display.
    lin = _srgb_to_linear(image)
    blurred = _linear_to_srgb(blur_image(lin, psf))
    if image.ndim == 2:
        predistorted = _linear_to_srgb(wiener_deconvolve(lin, psf))
    else:
        predistorted = _linear_to_srgb(np.stack(
            [wiener_deconvolve(lin[..., ch], psf) for ch in range(image.shape[-1])],
            axis=-1,
        ))
    kernel = _downsample_kernel(psf / psf.max(), _SHADER_KERNEL_MAX)

    return {
        "blurred": blurred,
        "predistorted": predistorted,
        "shader_kernel": np.round(kernel / kernel.sum(), 5).tolist(),
        "sharpness_uncorrected": sharpness(blurred if image.ndim == 2 else blurred.mean(axis=-1)),
    }
