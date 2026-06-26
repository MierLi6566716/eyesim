"""
Eccentric photorefraction image synthesizer.

In eccentric photorefraction (Kaakinen 1979), an IR point source is placed
off-axis from the camera. Light enters the eye through the pupil, reflects
off the retina, and exits. Because the exit wavefront is aberrated by the
eye's refractive error, different parts of the pupil send their reflected
rays in different directions. The camera captures only the rays that reach
its aperture — creating a crescent or half-moon pattern whose position,
width, and orientation encode (S, C, axis).

Double-pass wavefront model
---------------------------
For pupil point (x, y) [mm], source at eccentricity (ex, ey) [mm], camera
at distance D [mm]:

    slope_x = 2 * ( (M + J0)*x + J45*y ) * 1e-3   [rad, double pass]
    slope_y = 2 * ( (M - J0)*y + J45*x ) * 1e-3

    exit_x  = slope_x * D - ex                      [mm at camera plane]
    exit_y  = slope_y * D - ey

    L(x,y) = exp( -(exit_x^2 + exit_y^2) / (2*sigma^2) ) * pupil_mask

The camera "sees" a ray at (x,y) if (exit_x, exit_y) lies within its
aperture (sigma). A source at horizontal eccentricity encodes M and J0;
a vertical source encodes M and J45. Using both gives full (M, J0, J45)
recovery from a single frame with two LEDs.

Production notes
----------------
- Real hardware uses 850 nm IR (avoids visible glare, matches retinal
  reflectance peak). This does not change the wavefront gradient model
  but affects diffraction disc size and chromatic dispersion.
- Pupil radius should track the patient's actual pupil (3–4 mm typical
  in IR illumination). Wider pupils expose more higher-order aberrations
  and improve sensitivity for mild prescriptions.
- The current model uses only M, J0, J45 (Zernike defocus + astigmatism).
  For production, add at least c_coma (Z3^±1), c_trefoil (Z3^±3), and
  c_spherical (Z4^0) to the wavefront gradient.
- Domain randomisation: vary D by ±20 mm, eccentricity by ±1 mm, pupil
  radius by ±0.5 mm across samples to reduce the sim-to-real gap.
"""

from __future__ import annotations
import numpy as np
from .optics import prescription_to_power_vector


# Default hardware geometry (matches a practical wearable prototype)
_D_MM        = 1000.0    # camera-to-eye distance [mm] — 1 metre
_ECCENTRICITY = 10.0     # source offset from camera axis [mm]
_PUPIL_R_MM  = 3.0       # assumed IR pupil radius [mm]
_SIGMA_MM    = 3.0       # effective camera aperture at pupil plane [mm]


def render_single(
    S: float,
    C: float,
    axis: float,
    ex: float,
    ey: float,
    D_mm: float = _D_MM,
    pupil_r_mm: float = _PUPIL_R_MM,
    sigma_mm: float = _SIGMA_MM,
    image_size: int = 64,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Render a single-channel photorefraction image for one source direction.

    Parameters
    ----------
    S, C, axis : prescription (dioptres / degrees)
    ex, ey     : source eccentricity in x and y [mm]
    D_mm       : camera-to-eye distance [mm]
    pupil_r_mm : pupil radius [mm]
    sigma_mm   : camera aperture (Gaussian sigma at pupil plane) [mm]
    image_size : output pixel size (square)
    noise_std  : additive Gaussian noise sigma (0 = clean)
    rng        : numpy Generator for reproducibility

    Returns
    -------
    float32 array of shape (image_size, image_size) in [0, 1]
    """
    M, J0, J45 = prescription_to_power_vector(S, C, axis)

    coords = np.linspace(-pupil_r_mm, pupil_r_mm, image_size)
    x, y = np.meshgrid(coords, coords)
    pupil_mask = (x ** 2 + y ** 2) <= pupil_r_mm ** 2

    # Double-pass wavefront gradient [rad]
    slope_x = 2.0 * ((M + J0) * x + J45 * y) * 1e-3
    slope_y = 2.0 * ((M - J0) * y + J45 * x) * 1e-3

    # Ray exit position at camera plane relative to camera centre [mm]
    exit_x = slope_x * D_mm - ex
    exit_y = slope_y * D_mm - ey

    # Luminance: Gaussian camera aperture response
    L = np.exp(-(exit_x ** 2 + exit_y ** 2) / (2.0 * sigma_mm ** 2))

    # Low-level retinal background visible through any pupil
    L = L * 0.90 + 0.05
    L *= pupil_mask.astype(np.float32)

    if noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        L = L + noise_std * rng.standard_normal(L.shape).astype(np.float32)

    return np.clip(L, 0.0, 1.0).astype(np.float32)


def render_multichannel(
    S: float,
    C: float,
    axis: float,
    eccentricity_mm: float = _ECCENTRICITY,
    D_mm: float = _D_MM,
    pupil_r_mm: float = _PUPIL_R_MM,
    sigma_mm: float = _SIGMA_MM,
    image_size: int = 64,
    noise_std: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Render a 2-channel photorefraction image: horizontal + vertical source.

    Channel 0: source at (+eccentricity, 0) — encodes M and J0
    Channel 1: source at (0, +eccentricity) — encodes M and J45

    Together the two channels allow full (M, J0, J45) recovery by the CNN.

    Returns
    -------
    float32 array of shape (image_size, image_size, 2) in [0, 1]
    """
    if rng is None:
        rng = np.random.default_rng()

    ch0 = render_single(S, C, axis,
                        ex=eccentricity_mm, ey=0.0,
                        D_mm=D_mm, pupil_r_mm=pupil_r_mm, sigma_mm=sigma_mm,
                        image_size=image_size, noise_std=noise_std, rng=rng)
    ch1 = render_single(S, C, axis,
                        ex=0.0, ey=eccentricity_mm,
                        D_mm=D_mm, pupil_r_mm=pupil_r_mm, sigma_mm=sigma_mm,
                        image_size=image_size, noise_std=noise_std, rng=rng)
    return np.stack([ch0, ch1], axis=-1)


def sample_prescription(rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Draw one (S, C, axis) sample from a population-realistic distribution.

    Based on epidemiological data (Vitale 2009, Kempen 2004):
      - Spherical equivalent M ~ Normal(-0.5, 2.0), clipped to [-6, 4] D
      - J0 ~ Normal(0, 0.4), clipped to [-2, 2] D
      - J45 ~ Normal(0, 0.4), clipped to [-2, 2] D
      - Convert back to (S, C, axis) in minus-cylinder form

    Using power-vector sampling avoids the axis-wrapping problem.
    """
    M   = float(np.clip(rng.normal(-0.5, 2.0),  -6.0, 4.0))
    J0  = float(np.clip(rng.normal(0.0,  0.4),  -2.0, 2.0))
    J45 = float(np.clip(rng.normal(0.0,  0.4),  -2.0, 2.0))

    # Convert to (S, C, axis) — identical to power_vector_to_prescription in optimize.py
    astig = np.sqrt(J0 ** 2 + J45 ** 2)
    C = -2.0 * astig
    axis = float((0.5 * np.degrees(np.arctan2(J45, J0))) % 180) if astig > 1e-6 else 0.0
    S = M + astig

    return float(S), float(C), axis
