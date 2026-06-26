"""
Closed-loop optimizer.

Simulates the core control loop of the self-refracting corrector: given a
test image displayed through an eye with an unknown prescription, find the
tunable-lens correction that maximises image sharpness.

Three functions:
1. Sharpness metric: This is what we are trying to maximize. 
2. Optimizer: We need this to be gradient-free for hardware aplications. 
3. Search space: We optimize over (M, J0, J45), Cartesian. 
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import laplace
from .optics import prescription_to_power_vector, psf_from_prescription, blur_image


def sharpness(image: np.ndarray) -> float:
    """
    Blurry images have low Laplacian energy, sharp images have high energy.
    """
    return float(np.mean(laplace(image.astype(np.float64)) ** 2))


def power_vector_to_prescription(M: float, J0: float, J45: float) -> tuple[float, float, float]:
    """
    Convert (M, J0, J45) back to (S, C, axis_deg).
    """
    astig = np.sqrt(J0 ** 2 + J45 ** 2)   # = |C| / 2
    C = -2.0 * astig
    axis = float((0.5 * np.degrees(np.arctan2(J45, J0))) % 180) if astig > 1e-6 else 0.0
    S = M + astig                           # M − C/2 = M + |C|/2
    return float(S), float(C), axis


def run_closed_loop(
    true_S: float,
    true_C: float,
    true_axis: float,
    test_image: np.ndarray,
    pupil_radius_mm: float = 2.0,
    wavelength_nm: float = 550.0,
    grid: int = 128,
    psf_crop: int = 32,
    x0: tuple[float, float, float] = (0.0, 0.0, 0.0),
    max_iter: int = 400,
    tol: float = 1e-4,
) -> dict:
    """
    Simulate the closed-loop correction for one eye prescription.

    Runs a gradient-free Nelder-Mead search over tunable-lens correction
    (M_c, J0_c, J45_c) to maximise the sharpness of: blur(test_image,  PSF(eye_prescription + correction))

    Returns: 
    dict with keys:
        S_correction, C_correction, axis_correction
            Optimal correction in clinical form.
        M_correction, J0_correction, J45_correction
            Same in power-vector form.
        S_estimate, C_estimate, axis_estimate
            Estimated eye prescription (≈ −correction).
        final_sharpness, sharpness_uncorrected, sharpness_improvement_dB
            Sharpness before and after optimisation.
        iterations, converged
            Nelder-Mead diagnostics.
        history
            List of (M_c, J0_c, J45_c, sharpness) for every evaluation.
    """
    M_eye, J0_eye, J45_eye = prescription_to_power_vector(true_S, true_C, true_axis)
    history: list[tuple] = []
    def objective(params):
        M_c, J0_c, J45_c = params
        S_res, C_res, ax_res = power_vector_to_prescription(
            M_eye + M_c, J0_eye + J0_c, J45_eye + J45_c
        )
        psf = psf_from_prescription(
            S_res, C_res, ax_res,
            pupil_radius_mm=pupil_radius_mm,
            wavelength_nm=wavelength_nm,
            grid=grid,
            psf_crop=psf_crop,
        )
        s = sharpness(blur_image(test_image, psf))
        history.append((float(M_c), float(J0_c), float(J45_c), s))
        return -s  # minimise negative sharpness
    
    # Baseline: sharpness with zero correction applied
    S0, C0, ax0 = power_vector_to_prescription(M_eye, J0_eye, J45_eye)
    psf0 = psf_from_prescription(S0, C0, ax0, pupil_radius_mm=pupil_radius_mm, wavelength_nm=wavelength_nm, grid=grid, psf_crop=psf_crop)
    sharp_uncorr = sharpness(blur_image(test_image, psf0))
    J00, J450 = float(x0[1]), float(x0[2])

    # Stage 1: coarse 1D scan in M with J0/J45 fixed.
    # The sharpness landscape is nearly flat over several dioptres near zero
    # correction, then rises sharply near the true prescription. Scanning M
    # alone (no astigmatism) avoids the spurious local maxima that arise when
    # directional PSFs interact with non-isotropic images.
    best_M = float(x0[0])
    best_s_scan = -np.inf
    for M_scan in np.arange(-5.0, 5.5, 0.5):
        S_s, C_s, ax_s = power_vector_to_prescription(
            M_eye + M_scan, J0_eye + J00, J45_eye + J450
        )
        psf_s = psf_from_prescription(S_s, C_s, ax_s,
                                      pupil_radius_mm=pupil_radius_mm,
                                      wavelength_nm=wavelength_nm,
                                      grid=grid, psf_crop=psf_crop)
        s_scan = sharpness(blur_image(test_image, psf_s))
        history.append((float(M_scan), J00, J450, s_scan))
        if s_scan > best_s_scan:
            best_s_scan = s_scan
            best_M = float(M_scan)

    # Stage 2: Nelder-Mead from the best M, ±0.5 D simplex in all three axes.
    x_start = [best_M, J00, J450]
    init_simplex = np.array([
        [x_start[0],       x_start[1],       x_start[2]      ],
        [x_start[0] + 0.5, x_start[1],       x_start[2]      ],
        [x_start[0],       x_start[1] + 0.5, x_start[2]      ],
        [x_start[0],       x_start[1],       x_start[2] + 0.5],
    ])
    result = minimize(
        objective,
        x_start,
        method="Nelder-Mead",
        options={
            "maxiter": max_iter,
            "xatol": tol,
            "fatol": tol * 1e-3,
            "initial_simplex": init_simplex,
        },
    )

    M_c, J0_c, J45_c = result.x
    S_c, C_c, ax_c = power_vector_to_prescription(M_c, J0_c, J45_c)
    # Estimated eye prescription is the negative of the optimal correction
    S_est, C_est, ax_est = power_vector_to_prescription(-M_c, -J0_c, -J45_c)
    sharp_final = -result.fun
    with np.errstate(divide="ignore"):
        improvement_dB = float(
            10.0 * np.log10(max(sharp_final, 1e-30) / max(sharp_uncorr, 1e-30))
        )

    return {
        "S_correction": S_c,
        "C_correction": C_c,
        "axis_correction": ax_c,
        "M_correction": float(M_c),
        "J0_correction": float(J0_c),
        "J45_correction": float(J45_c),
        "S_estimate": S_est,
        "C_estimate": C_est,
        "axis_estimate": ax_est,
        "final_sharpness": sharp_final,
        "sharpness_uncorrected": sharp_uncorr,
        "sharpness_improvement_dB": improvement_dB,
        "iterations": result.nit,
        "converged": result.success,
        "history": history,
    }
