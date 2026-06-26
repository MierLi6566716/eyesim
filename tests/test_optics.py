"""Tests for the eye simulator."""
import numpy as np
from eyesim.optics import (
    prescription_to_power_vector,
    psf_from_prescription,
    simulate_vision,
)


def _spread(psf):
    n = psf.shape[0]
    y, x = np.mgrid[0:n, 0:n]
    cx = (psf * x).sum()
    cy = (psf * y).sum()
    sx = np.sqrt((psf * (x - cx) ** 2).sum())
    sy = np.sqrt((psf * (y - cy) ** 2).sum())
    return sx, sy


def test_power_vector_formulas():
    M, J0, J45 = prescription_to_power_vector(-2.0, -1.0, 90.0)
    assert np.isclose(M, -2.5)
    assert np.isclose(J0, -0.5, atol=1e-6)   # cos(180deg) = -1
    assert np.isclose(J45, 0.0, atol=1e-6)


def test_psf_normalised():
    psf = psf_from_prescription(-2.0, 0.0, 0.0)
    assert np.isclose(psf.sum(), 1.0, atol=1e-6)


def test_zero_prescription_is_sharpest():
    p0 = psf_from_prescription(0, 0, 0, grid=256, psf_crop=128)
    p2 = psf_from_prescription(-2, 0, 0, grid=256, psf_crop=128)
    assert max(_spread(p0)) < max(_spread(p2))


def test_sphere_is_symmetric():
    p = psf_from_prescription(-2.0, 0.0, 0.0, grid=256, psf_crop=128)
    sx, sy = _spread(p)
    assert np.isclose(sx, sy, rtol=0.02)


def test_blur_grows_with_power():
    spreads = [max(_spread(psf_from_prescription(S, 0, 0, grid=256, psf_crop=128)))
               for S in [0.0, -1.0, -2.0, -3.0]]
    assert all(b < a for b, a in zip(spreads, spreads[1:]))


def test_cylinder_is_directional_and_rotates():
    p0 = psf_from_prescription(0.0, -3.0, 0.0, grid=256, psf_crop=128)
    p90 = psf_from_prescription(0.0, -3.0, 90.0, grid=256, psf_crop=128)
    sx0, sy0 = _spread(p0)
    sx90, sy90 = _spread(p90)
    # axis 0 and axis 90 should be mirror images: ratio inverts
    assert (sx0 / sy0) < 1.0 < (sx90 / sy90)
    assert np.isclose(sx0 / sy0, sy90 / sx90, rtol=0.05)


def test_simulate_vision_shape_preserved():
    img = np.random.rand(128, 128)
    out, psf = simulate_vision(img, -1.5, -0.5, 30.0, grid=128, psf_crop=48)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
