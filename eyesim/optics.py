"""
Turn a prescription (Sphere, Cylinder, Axis) into the blur a human
eye sees using Fourier optics.

Pipeline:
    (S, C, theta)  ->  wavefront aberration over the pupil  (Zernike defocus + astigmatism)
                   ->  pupil function  P = aperture * exp(i * k * W)
                   ->  PSF = |FFT(P)|^2 
                   ->  blurred image = sharp image (conv) PSF

Conventions
-----------
- Prescription in dioptres (D) for S and C, degrees for axis theta (0..180).
- Negative S = myopia (near-sighted), positive S = hyperopia.
"""

from __future__ import annotations
import numpy as np

# Part 1: We convert (s, c, theta) to (M, J0, J45)
def prescription_to_power_vector(S: float, C: float, theta_deg: float):
    """
    Convert (Sphere, Cylinder, Axis) to the power-vector (M, J0, J45).
        M = S + C/2                 
        J0 = -(C/2) cos(2*theta)     
        J45 = -(C/2) sin(2*theta)     
    """
    theta = np.deg2rad(theta_deg)
    M = S + C / 2.0
    J0 = -(C / 2.0) * np.cos(2 * theta)
    J45 = -(C / 2.0) * np.sin(2 * theta)
    return M, J0, J45

# Part 2: We get zernike coeeficients by converting D to meters of optial path difference using pupil radius
def power_vector_to_zernike(M, J0, J45, pupil_radius_mm, wavelength_nm=550.0):
    """
    Convert power-vector dioptres into Zernike wavefront coefficients, for defocus (Z2^0) and the two astigmatism modes
    (Z2^-2, Z2^2).
    """
    r = pupil_radius_mm * 1e-3  # mm -> m
    # Defocus: convert dioptres of mean power to RMS Zernike defocus (metres)
    c_defocus = -M * r**2 / (4.0 * np.sqrt(3.0))
    # Astigmatism: J0 and J45 map to the two 2nd-order astigmatism Zernikes.
    c_astig_0 = -J0 * r**2 / (2.0 * np.sqrt(6.0)) * 2.0   # Z2^2  (0/90 oriented)
    c_astig_45 = -J45 * r**2 / (2.0 * np.sqrt(6.0)) * 2.0  # Z2^-2 (45 oriented)
    return c_defocus, c_astig_0, c_astig_45


def _unit_grid(n):
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

    Zernike modes used:
        Z2^0  (defocus) =  sqrt(3) * (2 rho^2 - 1)
        Z2^2  (astig 0/90) =  sqrt(6) * rho^2 * cos(2 phi)
        Z2^-2 (astig 45)=  sqrt(6) * rho^2 * sin(2 phi)
    """
    rho, phi, aperture = _unit_grid(n)
    Z_defocus = np.sqrt(3.0) * (2.0 * rho**2 - 1.0)
    Z_astig_0 = np.sqrt(6.0) * rho**2 * np.cos(2 * phi)
    Z_astig_45 = np.sqrt(6.0) * rho**2 * np.sin(2 * phi)
    W = c_defocus * Z_defocus + c_astig_0 * Z_astig_0 + c_astig_45 * Z_astig_45
    return W * aperture, aperture

# Fourier optics
def psf_from_prescription(
    S, C, theta_deg,
    pupil_radius_mm=2.0,
    wavelength_nm=550.0,
    grid=256,
    psf_crop=None,
):
    """
    Compute the point-spread function for a given prescription.

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

# Convolve with image to get retinal image.
def blur_image(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Convolve an image (HxW grayscale or HxWxC) with a PSF via FFT, Basically apply blur
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
    """image + prescription -> what that eye sees."""
    psf = psf_from_prescription(
        S, C, theta_deg,
        pupil_radius_mm=pupil_radius_mm,
        wavelength_nm=wavelength_nm,
        grid=grid, psf_crop=psf_crop,
    )
    return blur_image(image, psf), psf
