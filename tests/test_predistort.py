"""
Tests for the pre-distortion pipeline.

Key property under test: predistort(image, Rx) * blur(Rx) ≈ image.
Both predistort and the simulation blur use the same circular-FFT model,
so they form a mathematically consistent forward/inverse pair.
PSNR tests use a low-frequency structured image: most energy is at
spatial frequencies the PSF preserves, so predistortion has recoverable
signal to work with.
"""
import numpy as np
import pytest
from eyesim.predistort import wiener_deconvolve, predistort, simulate_display_chain, _fft_blur
from eyesim.optics import psf_from_prescription


def _psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


@pytest.fixture
def img_smooth():
    """256×256 sinusoidal image at 4 cycles per dimension.
    Frequency f = 4/256 ≈ 0.016 cy/px — well inside the passband of
    prescriptions up to ~-2D, so blur attenuates but does not destroy signal.
    """
    n = 256
    x = np.arange(n)
    xx, yy = np.meshgrid(x, x)
    img = 0.5 + 0.35 * np.sin(2 * np.pi * 4 * xx / n) * np.cos(2 * np.pi * 4 * yy / n)
    return img.astype(np.float32)


@pytest.fixture
def mild_rx():
    return dict(S=-1.5, C=-0.5, theta_deg=30.0)


@pytest.fixture
def strong_rx():
    return dict(S=-4.0, C=-2.0, theta_deg=90.0)


# --------------------------------------------------------------------------
# Output contract (shape / range)
# --------------------------------------------------------------------------
def test_predistort_shape_and_range():
    rng = np.random.default_rng(1)
    img = rng.random((128, 128)).astype(np.float32)
    out, _ = predistort(img, S=-1.5, C=-0.5, theta_deg=30.0, grid=128, psf_crop=32)
    assert out.shape == img.shape
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_predistort_colour_shape():
    rng = np.random.default_rng(7)
    img = rng.random((64, 64, 3)).astype(np.float32)
    out, _ = predistort(img, -2.0, -1.0, 45.0, grid=128, psf_crop=32)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_display_chain_keys(img_smooth, mild_rx):
    chain = simulate_display_chain(img_smooth, **mild_rx, grid=128, psf_crop=32)
    for key in ("original", "predistorted", "seen_corrected", "seen_raw", "psf"):
        assert key in chain


# --------------------------------------------------------------------------
# Physics: predistorted ⊛ PSF ≈ original
# --------------------------------------------------------------------------
def test_display_chain_improves_psnr(img_smooth, mild_rx):
    """
    For a structured image, predistortion + blur should recover the original
    significantly better than leaving the image uncorrected.
    """
    chain = simulate_display_chain(img_smooth, grid=128, psf_crop=32, **mild_rx)
    psnr_corrected = _psnr(img_smooth, chain["seen_corrected"])
    psnr_raw = _psnr(img_smooth, chain["seen_raw"])
    assert psnr_corrected > psnr_raw + 5.0, (
        f"Correction gave insufficient benefit: corrected={psnr_corrected:.1f} dB "
        f"vs raw={psnr_raw:.1f} dB"
    )


def test_lower_noise_power_sharper(img_smooth, mild_rx):
    """Smaller K → more aggressive deconvolution → higher PSNR."""
    def psnr_at(k):
        chain = simulate_display_chain(img_smooth, **mild_rx, noise_power=k,
                                       grid=128, psf_crop=32)
        return _psnr(img_smooth, chain["seen_corrected"])

    assert psnr_at(1e-4) >= psnr_at(1e-2) - 1.0


# --------------------------------------------------------------------------
# Wiener deconvolve is the inverse of _fft_blur (consistent model)
# --------------------------------------------------------------------------
def test_wiener_round_trip(img_smooth):
    """
    _fft_blur then wiener_deconvolve should nearly recover the original.
    Both use the same circular-FFT model, so this is a tight mathematical check.
    """
    psf = psf_from_prescription(-1.5, -0.5, 30.0, grid=128, psf_crop=32)
    blurred = _fft_blur(img_smooth, psf)
    recovered = wiener_deconvolve(blurred, psf, noise_power=1e-3)
    psnr_recovered = _psnr(img_smooth, recovered)
    psnr_blurred = _psnr(img_smooth, blurred)
    assert psnr_recovered > psnr_blurred + 20.0, (
        f"Round-trip failed: recovered={psnr_recovered:.1f} dB "
        f"vs blurred={psnr_blurred:.1f} dB"
    )
