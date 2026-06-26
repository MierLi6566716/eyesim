"""
Tests for the closed-loop sharpness optimizer.

Physics invariants under test:
- power_vector <-> prescription round-trip is exact
- sharpness metric orders blurry < sharp images correctly
- optimizer finds a correction that improves sharpness substantially
- for a pure-sphere prescription the optimised M_correction is within 0.25 D
- for a sphere+cylinder prescription all three power-vector components converge

Image: isotropic sum of sinusoids at 6 orientations. This gives a unimodal
sharpness landscape — unlike a directional sinusoid, any residual aberration
reduces sharpness monotonically from the diffraction limit in all directions.
Tests use grid=64, psf_crop=32 (minimum psf_crop to avoid ring-artefact at
high myopia from PSF truncation).
"""

import numpy as np
import pytest
from eyesim.optimize import (
    sharpness,
    power_vector_to_prescription,
    run_closed_loop,
)
from eyesim.optics import prescription_to_power_vector, psf_from_prescription, blur_image


@pytest.fixture(scope="module")
def img_iso():
    n = 128
    x = np.arange(n)
    xx, yy = np.meshgrid(x, x)
    img = np.zeros((n, n))
    for angle in range(0, 180, 30):
        rad = np.radians(angle)
        img += np.sin(2 * np.pi * (4 * np.cos(rad) * xx / n + 4 * np.sin(rad) * yy / n))
    return np.clip(img / np.abs(img).max() * 0.35 + 0.5, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Unit tests: power-vector math
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("S,C,axis", [
    (-1.5, 0.0,  0.0),
    (-1.5, -0.5, 30.0),
    (2.0,  -1.0, 90.0),
    (0.0,  -0.75, 45.0),
])
def test_power_vector_round_trip(S, C, axis):
    """prescription -> power_vector -> prescription should be lossless."""
    M, J0, J45 = prescription_to_power_vector(S, C, axis)
    S2, C2, ax2 = power_vector_to_prescription(M, J0, J45)
    assert abs(S2 - S) < 1e-6
    assert abs(C2 - C) < 1e-6
    # axis is only defined when C != 0
    if abs(C) > 1e-3:
        assert abs(ax2 - axis) < 1e-4


def test_power_vector_zero_astig():
    """Pure sphere: J0=J45=0, C=0, S=M, axis undefined (returns 0)."""
    S, C, axis = power_vector_to_prescription(-1.5, 0.0, 0.0)
    assert abs(S - (-1.5)) < 1e-9
    assert abs(C) < 1e-9
    assert axis == 0.0


# ---------------------------------------------------------------------------
# Unit tests: sharpness metric
# ---------------------------------------------------------------------------

def test_sharpness_orders_blur(img_iso):
    """
    Sharpness must fall as defocus increases.
    Comparison is PSF-mediated: diffraction-limited vs mild vs heavy defocus.
    Comparing raw image vs blurred is not valid because the defocus PSF's
    ring structure can create new edges, raising the Laplacian spuriously.
    """
    psf_none  = psf_from_prescription(0.0,  0.0, 0.0, grid=64, psf_crop=32)
    psf_mild  = psf_from_prescription(-0.5, 0.0, 0.0, grid=64, psf_crop=32)
    psf_heavy = psf_from_prescription(-2.0, 0.0, 0.0, grid=64, psf_crop=32)
    s_sharp = sharpness(blur_image(img_iso, psf_none))
    s_mild  = sharpness(blur_image(img_iso, psf_mild))
    s_heavy = sharpness(blur_image(img_iso, psf_heavy))
    assert s_sharp > s_mild > s_heavy, (
        f"Sharpness not monotone: {s_sharp:.6f} > {s_mild:.6f} > {s_heavy:.6f}"
    )


# ---------------------------------------------------------------------------
# Integration tests: run_closed_loop
# ---------------------------------------------------------------------------

def test_optimizer_sphere_improves_sharpness(img_iso):
    """Correction should raise sharpness by at least 10 dB."""
    res = run_closed_loop(-1.5, 0.0, 0.0, img_iso, grid=64, psf_crop=32, max_iter=400)
    assert res["sharpness_improvement_dB"] > 10.0, (
        f"Insufficient improvement: {res['sharpness_improvement_dB']:.1f} dB"
    )


def test_optimizer_sphere_converges(img_iso):
    """
    For a -1.5 D sphere, M_correction should converge to +1.5 D within 0.25 D.
    J0 and J45 corrections should be near zero.
    """
    res = run_closed_loop(-1.5, 0.0, 0.0, img_iso, grid=64, psf_crop=32, max_iter=400)
    assert abs(res["M_correction"] - 1.5) < 0.25, (
        f"M_correction {res['M_correction']:.3f} too far from 1.5"
    )
    assert abs(res["J0_correction"]) < 0.25
    assert abs(res["J45_correction"]) < 0.25


def test_optimizer_astigmat_converges(img_iso):
    """
    For S=-1.5, C=-0.5, axis=30, all three power-vector components must
    converge within 0.25 D of the true correction.
    True correction: M=+1.75, J0=-0.125, J45=-0.2165.
    """
    res = run_closed_loop(-1.5, -0.5, 30.0, img_iso, grid=64, psf_crop=32, max_iter=400)
    assert abs(res["M_correction"]   - 1.75)   < 0.25, (
        f"M_correction {res['M_correction']:.3f} not within 0.25 of 1.75"
    )
    assert abs(res["J0_correction"]  - (-0.125))  < 0.25
    assert abs(res["J45_correction"] - (-0.2165)) < 0.25


def test_optimizer_history_populated(img_iso):
    """history should contain at least one entry per iteration, all valid."""
    res = run_closed_loop(-1.5, 0.0, 0.0, img_iso, grid=64, psf_crop=32, max_iter=400)
    assert len(res["history"]) > 0
    for entry in res["history"]:
        M_c, J0_c, J45_c, s = entry
        assert np.isfinite(s) and s >= 0.0
