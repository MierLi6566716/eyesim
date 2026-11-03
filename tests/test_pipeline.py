"""Tests for the whole simulated loop, with a fake estimator so no network is needed."""

import numpy as np
import pytest

from eyesim.optics import prescription_to_power_vector, min_psf_crop, psf_from_prescription
from eyesim.pipeline import run_full_pipeline


def _star(n=256, spokes=16):
    # 256 px: smaller images can't show the difference between mild and strong blur.
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x)
    phi = np.arctan2(yy, xx)
    r = np.sqrt(xx ** 2 + yy ** 2)
    img = 0.5 + 0.5 * np.sign(np.sin(spokes * phi))
    img[r > 0.95] = 1.0
    return img.astype(np.float32)


def _perfect_estimator(S, C, axis):
    truth = prescription_to_power_vector(S, C, axis)
    return lambda frames: truth


def _biased_estimator(S, C, axis, dM=0.3):
    M, J0, J45 = prescription_to_power_vector(S, C, axis)
    return lambda frames: (M + dM, J0, J45)


def test_pipeline_stages_present():
    """By default the loop stops after CORRECT; REFINE needs a bench camera."""
    S, C, axis = -1.5, -0.5, 30.0
    res = run_full_pipeline(S, C, axis, _perfect_estimator(S, C, axis), _star(),
                            n_frames=2, seed=0, max_iter=50)
    for key in ["seen_uncorrected", "photoref", "M_cnn", "seen_cnn", "summary"]:
        assert key in res
    assert "seen_final" not in res
    assert "loop" not in res
    assert res["photoref_frames"].shape[0] == 2


def test_pipeline_refine_stage_present_when_requested():
    S, C, axis = -1.5, -0.5, 30.0
    res = run_full_pipeline(S, C, axis, _perfect_estimator(S, C, axis), _star(),
                            n_frames=2, seed=0, max_iter=50, refine=True)
    for key in ["seen_final", "loop"]:
        assert key in res


def test_perfect_estimate_gives_zero_residual():
    S, C, axis = -2.0, -0.5, 60.0
    res = run_full_pipeline(S, C, axis, _perfect_estimator(S, C, axis), _star(),
                            refine=False, seed=0)
    assert res["summary"]["residual_D_after_cnn"] < 1e-9
    # corrected view must be sharper than uncorrected
    assert res["sharpness_cnn"] > res["sharpness_uncorrected"]


def test_refinement_reduces_biased_estimate():
    """REFINE fixes a 0.3 D error in the estimate."""
    S, C, axis = -1.5, 0.0, 0.0
    res = run_full_pipeline(S, C, axis, _biased_estimator(S, C, axis, dM=0.3),
                            _star(), seed=0, max_iter=150, refine=True)
    assert abs(res["summary"]["residual_D_after_cnn"] - 0.3) < 1e-6
    assert res["summary"]["residual_D_after_loop"] < 0.1
    assert res["sharpness_final"] >= res["sharpness_cnn"]


def test_min_psf_crop_contains_energy():
    """The crop recommended by min_psf_crop must keep ~all PSF energy.

    98% threshold: the geometric blur disc is fully contained, the remaining
    <2% lives in diffraction (Airy) tails that decay too slowly to bound
    tightly at low powers.
    """
    for M in [-1.0, -2.5, -4.0]:
        crop = min_psf_crop(abs(M))
        grid = max(2 * crop, 256)
        full = psf_from_prescription(M, 0, 0, grid=grid, psf_crop=None)
        c, h = grid // 2, crop // 2
        kept = full[c - h:c + h, c - h:c + h].sum()
        assert kept > 0.98, f"M={M}: crop {crop} keeps only {kept:.2%}"
