"""
eyesim.optics
=============
Turn a spectacle prescription (Sphere, Cylinder, Axis) into the blur a human
eye applies to what it sees, using Fourier optics.

Pipeline:
    (S, C, theta)  ->  wavefront aberration over the pupil  (Zernike defocus + astigmatism)
                   ->  pupil function  P = aperture * exp(i * k * W)
                   ->  PSF = |FFT(P)|^2          (incoherent point-spread function)
                   ->  blurred image = sharp image (conv) PSF

The "medical" content is only this: a prescription is three numbers, and defocus
+ astigmatism are three low-order Zernike modes. Everything else is signal
processing.

Conventions
-----------
- Prescription in dioptres (D) for S and C, degrees for axis theta (0..180).
- Negative S = myopia (near-sighted), positive S = hyperopia.
- We use the *minus-cylinder* convention (common in optometry); the simulator is
  sign-agnostic for blur magnitude since blur depends on |defocus|, but we keep
  signs so the same code can later drive a *corrector* (which needs the sign).
"""

from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# 1. Prescription  ->  power-vector  ->  Zernike coefficients
# ---------------------------------------------------------------------------
def prescription_to_power_vector(S: float, C: float, theta_deg: float):
    """
    Convert (Sphere, Cylinder, Axis) to the clinical power-vector (M, J0, J45).

    This is the standard Thibos power-vector decomposition used throughout the
    refraction literature. It removes the axis ambiguity by turning astigmatism
    into two orthogonal Cartesian components.

        M   = S + C/2                 (spherical equivalent / mean defocus)
        J0  = -(C/2) cos(2*theta)     (with-/against-the-rule astigmatism)
        J45 = -(C/2) sin(2*theta)     (oblique astigmatism)
    """
    theta = np.deg2rad(theta_deg)
    M = S + C / 2.0
    J0 = -(C / 2.0) * np.cos(2 * theta)
    J45 = -(C / 2.0) * np.sin(2 * theta)
    return M, J0, J45


def power_vector_to_zernike(M, J0, J45, pupil_radius_mm, wavelength_nm=550.0):
    """
    Convert power-vector dioptres into Zernike wavefront coefficients (in metres
    of optical path difference), for defocus (Z2^0) and the two astigmatism modes
    (Z2^-2, Z2^2).

    A wavefront curvature of P dioptres over a pupil of radius r produces a
    peak optical path difference. For defocus the standard relation between
    dioptric power P (1/m) and the RMS-normalised Zernike defocus coefficient
    c_2^0 is:

        c_2^0 = - P * r^2 / (4 * sqrt(3))

    and for the astigmatism terms (the J0/J45 components), using the same r^2
    scaling with the sqrt(6) normalisation of the 2nd-order astigmatism Zernikes:

        c_2^2  =  J0  * r^2 / (2 * sqrt(6)) * (-2)   [see note below]

    We fold the constants into clear factors below. Coefficients are returned in
    metres so they can be multiplied by the wavenumber k = 2*pi/lambda directly.
    """
    r = pupil_radius_mm * 1e-3  # mm -> m
    # Defocus: convert dioptres of mean power to RMS Zernike defocus (metres)
    c_defocus = -M * r**2 / (4.0 * np.sqrt(3.0))
    # Astigmatism: J0 and J45 map to the two 2nd-order astigmatism Zernikes.
    c_astig_0 = -J0 * r**2 / (2.0 * np.sqrt(6.0)) * 2.0   # Z2^2  (0/90 oriented)
    c_astig_45 = -J45 * r**2 / (2.0 * np.sqrt(6.0)) * 2.0  # Z2^-2 (45 oriented)
    return c_defocus, c_astig_0, c_astig_45


# ---------------------------------------------------------------------------
# 2. Zernike wavefront over a circular pupil
# ---------------------------------------------------------------------------
def _unit_grid(n):
    """Return rho, phi polar coords on an n x n grid spanning the unit disk,
    plus the circular aperture mask."""
    x = np.linspace(-1, 1, n)
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx**2 + yy**2)
    phi = np.arctan2(yy, xx)
    aperture = (rho <= 1.0).astype(float)
    return rho, phi, aperture


def wavefront(n, c_defocus, c_astig_0, c_astig_45):
    """
    Build the wavefront aberration map W (in metres) on an n x n grid over the
    pupil, from the three Zernike coefficients.

    Zernike modes used (Noll/OSA normalised, unit disk):
        Z2^0  (defocus)        =  sqrt(3) * (2 rho^2 - 1)
        Z2^2  (astig 0/90)     =  sqrt(6) * rho^2 * cos(2 phi)
        Z2^-2 (astig 45)       =  sqrt(6) * rho^2 * sin(2 phi)
    """
    rho, phi, aperture = _unit_grid(n)
    Z_defocus = np.sqrt(3.0) * (2.0 * rho**2 - 1.0)
    Z_astig_0 = np.sqrt(6.0) * rho**2 * np.cos(2 * phi)
    Z_astig_45 = np.sqrt(6.0) * rho**2 * np.sin(2 * phi)
    W = c_defocus * Z_defocus + c_astig_0 * Z_astig_0 + c_astig_45 * Z_astig_45
    return W * aperture, aperture


# ---------------------------------------------------------------------------
# 3. Pupil function  ->  PSF
# ---------------------------------------------------------------------------
def psf_from_prescription(
    S, C, theta_deg,
    pupil_radius_mm=2.0,
    wavelength_nm=550.0,
    grid=256,
    psf_crop=None,
):
    """
    Compute the (incoherent) point-spread function for a given prescription.

    Returns a normalised 2-D PSF (sums to 1). The PSF is the blur kernel the eye
    applies to the retinal image.
    """
    M, J0, J45 = prescription_to_power_vector(S, C, theta_deg)
    cd, ca0, ca45 = power_vector_to_zernike(M, J0, J45, pupil_radius_mm, wavelength_nm)
    W, aperture = wavefront(grid, cd, ca0, ca45)

    wavelength_m = wavelength_nm * 1e-9
    k = 2.0 * np.pi / wavelength_m
    pupil_function = aperture * np.exp(1j * k * W)

    # PSF = |FT(pupil)|^2  (Fraunhofer / Fourier optics)
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_function)))
    psf = np.abs(field) ** 2
    psf /= psf.sum()

    if psf_crop is not None:
        c = grid // 2
        h = psf_crop // 2
        psf = psf[c - h:c + h, c - h:c + h]
        psf /= psf.sum()
    return psf


# ---------------------------------------------------------------------------
# 4. Apply the blur to an image
# ---------------------------------------------------------------------------
def blur_image(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Convolve an image (HxW grayscale or HxWxC) with a PSF via FFT.
    Returns a same-shape float image.
    """
    from scipy.signal import fftconvolve

    if image.ndim == 2:
        out = fftconvolve(image, psf, mode="same")
    else:
        out = np.stack(
            [fftconvolve(image[..., ch], psf, mode="same")
             for ch in range(image.shape[-1])],
            axis=-1,
        )
    return np.clip(out, 0.0, 1.0)


def simulate_vision(image, S, C, theta_deg, pupil_radius_mm=2.0,
                    wavelength_nm=550.0, grid=256, psf_crop=64):
    """Convenience: image + prescription -> what that eye sees."""
    psf = psf_from_prescription(
        S, C, theta_deg,
        pupil_radius_mm=pupil_radius_mm,
        wavelength_nm=wavelength_nm,
        grid=grid, psf_crop=psf_crop,
    )
    return blur_image(image, psf), psf
