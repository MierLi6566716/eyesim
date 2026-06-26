"""
eyesim.predistort
=================
Pre-distortion (vision-correcting) display pipeline.

Given a prescription (S, C, axis), produce a display image that looks
garbled to a normal eye but appears sharp to an eye with that prescription.

Physics
-------
If the eye applies blur kernel h, and we display image d, the retinal image is:

    r = d * h          (* = convolution)

We want r = target (the sharp image), so d = target * h^{-1}.

Pure inversion amplifies noise at frequencies where H(f) ≈ 0.
Wiener deconvolution regularises this:

    D(f) = conj(H(f)) / (|H(f)|^2 + K) · T(f)

where K is the regularisation constant. Small K → sharper predistortion but
more noise amplification; larger K → softer, more stable output.

The output is clipped to [0, 1] (displayable range). Clipping limits
correction for severe prescriptions; the effect is largest when the PSF
is nearly uniform across the display region.

Convolution model
-----------------
Both wiener_deconvolve and _fft_blur use the same circular-convolution FFT
model (PSF placed at (0,0) via the quadrant trick). They form a consistent
forward/inverse pair. For images significantly larger than the PSF (≥8:1 ratio
in each dimension), circular convolution closely approximates the physical
linear convolution an eye performs.
"""

from __future__ import annotations
import numpy as np
from .optics import psf_from_prescription


def _embed_psf(psf: np.ndarray, h: int, w: int) -> np.ndarray:
    """Embed PSF in an h×w canvas with the PSF centre at (0,0) for FFT."""
    ph, pw = psf.shape
    ph2, pw2 = ph // 2, pw // 2
    psf_full = np.zeros((h, w))
    psf_full[:ph2, :pw2]     = psf[ph2:, pw2:]
    psf_full[:ph2, w-pw2:]   = psf[ph2:, :pw2]
    psf_full[h-ph2:, :pw2]   = psf[:ph2, pw2:]
    psf_full[h-ph2:, w-pw2:] = psf[:ph2, :pw2]
    return psf_full


def _fft_blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Circular convolution of a single-channel float image with a PSF.

    Uses the same PSF embedding as wiener_deconvolve, so the two functions
    form a consistent forward / inverse pair for testing and simulation.
    """
    h, w = image.shape
    H = np.fft.fft2(_embed_psf(psf, h, w))
    T = np.fft.fft2(image.astype(np.float64))
    return np.clip(np.real(np.fft.ifft2(H * T)), 0.0, 1.0)


def wiener_deconvolve(
    image: np.ndarray,
    psf: np.ndarray,
    noise_power: float = 1e-3,
) -> np.ndarray:
    """
    Wiener deconvolution of a single-channel float image [0, 1].

    Parameters
    ----------
    image : H × W float array in [0, 1]
    psf   : blur kernel (centre at psf[ph//2, pw//2])
    noise_power : regularisation constant K.  1e-3 is a reasonable default.
                  Lower → sharper; higher → softer.

    Returns
    -------
    predistorted : H × W float array, clipped to [0, 1]
    """
    h, w = image.shape
    H = np.fft.fft2(_embed_psf(psf, h, w))
    T = np.fft.fft2(image.astype(np.float64))
    W = np.conj(H) / (np.abs(H) ** 2 + noise_power)
    return np.clip(np.real(np.fft.ifft2(W * T)), 0.0, 1.0)


def predistort(
    image: np.ndarray,
    S: float,
    C: float,
    theta_deg: float,
    pupil_radius_mm: float = 2.0,
    wavelength_nm: float = 550.0,
    grid: int = 256,
    psf_crop: int = 64,
    noise_power: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-distort an image so that an eye with prescription (S, C, theta_deg)
    sees a sharp version of it.

    Parameters
    ----------
    image       : H × W grayscale or H × W × C colour float image in [0, 1]
    S, C        : sphere and cylinder in dioptres
    theta_deg   : cylinder axis in degrees (0–180)
    noise_power : Wiener regularisation (see wiener_deconvolve)

    Returns
    -------
    predistorted : same shape as image, values in [0, 1]
    psf          : the PSF used (useful for visualisation and verification)
    """
    psf = psf_from_prescription(
        S, C, theta_deg,
        pupil_radius_mm=pupil_radius_mm,
        wavelength_nm=wavelength_nm,
        grid=grid,
        psf_crop=psf_crop,
    )

    if image.ndim == 2:
        predistorted = wiener_deconvolve(image, psf, noise_power)
    else:
        predistorted = np.stack(
            [wiener_deconvolve(image[..., ch], psf, noise_power)
             for ch in range(image.shape[-1])],
            axis=-1,
        )

    return predistorted, psf


def simulate_display_chain(
    image: np.ndarray,
    S: float,
    C: float,
    theta_deg: float,
    **kwargs,
) -> dict[str, np.ndarray]:
    """
    Full chain: sharp → predistorted display → what the target eye sees.

    Uses the same circular-FFT blur model as wiener_deconvolve so that
    seen_corrected is the true inverse of the predistortion (up to K and
    display clipping).

    Returns a dict with keys:
        'original'      : the input image
        'predistorted'  : what gets displayed (looks garbled to a normal eye)
        'seen_corrected': what the target eye perceives (should look sharp)
        'seen_raw'      : what the target eye sees with no display correction
        'psf'           : the blur kernel
    """
    predistorted, psf = predistort(image, S, C, theta_deg, **kwargs)

    if image.ndim == 2:
        seen_corrected = _fft_blur(predistorted, psf)
        seen_raw = _fft_blur(image, psf)
    else:
        seen_corrected = np.stack(
            [_fft_blur(predistorted[..., ch], psf) for ch in range(image.shape[-1])],
            axis=-1,
        )
        seen_raw = np.stack(
            [_fft_blur(image[..., ch], psf) for ch in range(image.shape[-1])],
            axis=-1,
        )

    return {
        "original": image,
        "predistorted": predistorted,
        "seen_corrected": seen_corrected,
        "seen_raw": seen_raw,
        "psf": psf,
    }
