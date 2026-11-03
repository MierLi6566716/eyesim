"""
The whole device in simulation, for one eye:

  1. SEE      what the uncorrected eye sees
  2. SENSE    a few infrared pupil photos of that eye (photorefraction.py)
  3. ESTIMATE the network reads them -> (M, J0, J45)
  4. CORRECT  set the lens to minus that estimate; what the eye sees now
  5. REFINE   optional: a sharpness search starting from the estimate. This
              needs a camera that sees what the eye sees, so it only works on
              a bench with a test chart, not in glasses.

The true prescription is only used to make the photos and to score the
result; the network never sees it. Any callable frames -> (M, J0, J45) can be
the estimator, so tests can use a fake one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .optics import (
    prescription_to_power_vector,
    calibrated_psf_for_image,
    blur_image,
)
from .optimize import (
    sharpness,
    power_vector_to_prescription,
    run_closed_loop,
)
from .photorefraction import Eye, Geometry, Scene, aux_vector, render_pair

# The simulated rig: camera 1 m away, LEDs 10 mm off the lens, 2.6 mm lens opening.
RIG = Geometry(d_mm=1000.0, ecc_mm=10.0, aperture_r_mm=1.3)
PUPIL_R_MM = 3.0


# (N, H, W, 2) or (H, W, 2) photos -> (M, J0, J45) in dioptres
Estimator = Callable[[np.ndarray], tuple[float, float, float]]


def load_estimator(checkpoint: str | Path, device: str | None = None) -> Estimator:
    """
    Wrap a trained checkpoint as an Estimator for photos taken with RIG.
    A stack of photos of the same eye gets one averaged answer.
    """
    import torch
    from .cnn import PupilNet

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PupilNet(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    aux = aux_vector(RIG.d_mm, RIG.ecc_mm, PUPIL_R_MM, 2 * RIG.aperture_r_mm)[:model.aux_dim]

    def estimate(image: np.ndarray) -> tuple[float, float, float]:
        imgs = image[None] if image.ndim == 3 else image        # (N, H, W, 2)
        x = torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2), dtype=np.float32))
        a = torch.from_numpy(np.repeat(aux[None], len(imgs), 0))
        with torch.no_grad():
            pred = model(x.to(device), a.to(device)).mean(dim=0).cpu().numpy()
        return float(pred[0]), float(pred[1]), float(pred[2])

    return estimate


def _residual_view(test_image, M_res, J0_res, J45_res, *,
                   pupil_radius_mm, wavelength_nm):
    """What the eye sees with a given leftover error."""
    S, C, ax = power_vector_to_prescription(M_res, J0_res, J45_res)
    psf = calibrated_psf_for_image(S, C, ax, test_image.shape,
                                   pupil_radius_mm=pupil_radius_mm,
                                   wavelength_nm=wavelength_nm)
    return blur_image(test_image, psf), psf


def run_full_pipeline(
    true_S: float,
    true_C: float,
    true_axis: float,
    estimator: Estimator,
    test_image: np.ndarray,
    n_frames: int = 4,
    sensor_noise_std: float = 0.02,
    refine: bool = False,
    seed: int | None = None,
    pupil_radius_mm: float = 2.0,
    wavelength_nm: float = 550.0,
    max_iter: int = 200,
) -> dict:
    """
    One sense -> estimate -> correct cycle (plus refine if asked). Returns a
    dict with every stage: seen_uncorrected, photoref_frames, M/J0/J45_cnn,
    seen_cnn, residual_cnn, and with refine: loop, seen_final, residual_final.
    "summary" has the leftover error (D) and sharpness after each stage.
    """
    rng = np.random.default_rng(seed)
    M_eye, J0_eye, J45_eye = prescription_to_power_vector(true_S, true_C, true_axis)
    out: dict = {
        "M_true": M_eye, "J0_true": J0_eye, "J45_true": J45_eye,
        "S_true": float(true_S), "C_true": float(true_C), "axis_true": float(true_axis),
    }

    # Stage 1 — the uncorrected eye looks at the scene
    seen_raw, psf_eye = _residual_view(
        test_image, M_eye, J0_eye, J45_eye,
        pupil_radius_mm=pupil_radius_mm, wavelength_nm=wavelength_nm)
    out["psf_eye"] = psf_eye
    out["seen_uncorrected"] = seen_raw
    out["sharpness_uncorrected"] = sharpness(seen_raw)

    # Stage 2 — sensing: n photorefraction frames of the same eye
    eye = Eye(M=M_eye, J0=J0_eye, J45=J45_eye, pupil_r_mm=PUPIL_R_MM, scatter_mrad=2.0)
    scene = Scene(glint_amp=1.5, red_reflex=0.03, surround=0.1, read_noise=sensor_noise_std)
    frames = np.stack([render_pair(eye, RIG, scene, rng) for _ in range(n_frames)])
    out["photoref_frames"] = frames
    out["photoref"] = frames.mean(axis=0)

    # Stage 3 — CNN estimate (frame-averaged)
    M_hat, J0_hat, J45_hat = estimator(frames)
    out["M_cnn"], out["J0_cnn"], out["J45_cnn"] = M_hat, J0_hat, J45_hat
    S_hat, C_hat, ax_hat = power_vector_to_prescription(M_hat, J0_hat, J45_hat)
    out["S_cnn"], out["C_cnn"], out["axis_cnn"] = S_hat, C_hat, ax_hat

    # Stage 4 — open-loop correction: lens set to the negated CNN estimate
    corr_cnn = (-M_hat, -J0_hat, -J45_hat)
    res_cnn = (M_eye + corr_cnn[0], J0_eye + corr_cnn[1], J45_eye + corr_cnn[2])
    out["residual_cnn"] = res_cnn
    seen_cnn, psf_cnn = _residual_view(
        test_image, *res_cnn,
        pupil_radius_mm=pupil_radius_mm, wavelength_nm=wavelength_nm)
    out["seen_cnn"] = seen_cnn
    out["psf_cnn"] = psf_cnn
    out["sharpness_cnn"] = sharpness(seen_cnn)

    # Stage 5 — closed-loop refinement warm-started from the CNN correction
    if refine:
        loop = run_closed_loop(
            true_S, true_C, true_axis, test_image,
            pupil_radius_mm=pupil_radius_mm, wavelength_nm=wavelength_nm,
            x0=corr_cnn, coarse_scan=False, simplex_size=0.25,
            max_iter=max_iter,
        )
        out["loop"] = loop
        res_final = (M_eye + loop["M_correction"],
                     J0_eye + loop["J0_correction"],
                     J45_eye + loop["J45_correction"])
        out["residual_final"] = res_final
        seen_final, psf_final = _residual_view(
            test_image, *res_final,
            pupil_radius_mm=pupil_radius_mm, wavelength_nm=wavelength_nm)
        out["seen_final"] = seen_final
        out["psf_final"] = psf_final
        out["sharpness_final"] = sharpness(seen_final)

    def _pv_norm(m, j0, j45):
        return float(np.sqrt(m ** 2 + j0 ** 2 + j45 ** 2))

    summary = {
        "residual_D_uncorrected": _pv_norm(M_eye, J0_eye, J45_eye),
        "residual_D_after_cnn": _pv_norm(*res_cnn),
        "sharpness_uncorrected": out["sharpness_uncorrected"],
        "sharpness_after_cnn": out["sharpness_cnn"],
    }
    if refine:
        summary["residual_D_after_loop"] = _pv_norm(*res_final)
        summary["sharpness_after_loop"] = out["sharpness_final"]
        summary["loop_evals"] = len(out["loop"]["history"])
    out["summary"] = summary
    return out
