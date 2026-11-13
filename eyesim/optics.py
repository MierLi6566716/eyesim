"""
Shows what an eye with a given prescription sees: prescription -> blur.

Steps:
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

# Part 1: (S, C, axis) -> (M, J0, J45)
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
    return float(M), float(J0), float(J45)

# Part 2: dioptres -> Zernike coefficients (metres of wavefront error over the pupil)
def power_vector_to_zernike(M, J0, J45, pupil_radius_mm, wavelength_nm=550.0):
    """
    Convert power-vector dioptres into Zernike wavefront coefficients, for defocus (Z2^0) and the two astigmatism modes
    (Z2^-2, Z2^2).
    """
    r = pupil_radius_mm * 1e-3  # mm -> m
    # Defocus: convert dioptres of mean power to RMS Zernike defocus (metres).
    # Standard relation (Thibos et al. 2002, "Power vector analysis..."):
    #   M = -4*sqrt(3) * c_2^0 / r^2   =>   c_2^0 = -M * r^2 / (4*sqrt(3))
    c_defocus = -M * r**2 / (4.0 * np.sqrt(3.0))
    # Astigmatism: J0 and J45 map to the two 2nd-order astigmatism Zernikes
    # by the same-family relation: J = -2*sqrt(6) * c_2^{+-2} / r^2
    #   =>   c_2^{+-2} = -J * r^2 / (2*sqrt(6))
    c_astig_0 = -J0 * r**2 / (2.0 * np.sqrt(6.0))    # Z2^2  (0/90 oriented)
    c_astig_45 = -J45 * r**2 / (2.0 * np.sqrt(6.0))  # Z2^-2 (45 oriented)
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

def blur_diameter_arcmin(M_dioptres: float, pupil_radius_mm: float = 2.0) -> float:
    """
    Blur circle diameter in arcminutes for a focus error of |M| D through a
    pupil of this radius (angle = M * pupil diameter). Example: 1 D through a
    3 mm pupil gives about 10 arcmin, roughly 20/100 to 20/200 vision.
    """
    pupil_diameter_m = 2.0 * pupil_radius_mm * 1e-3
    theta_rad = abs(M_dioptres) * pupil_diameter_m
    return theta_rad * (180.0 / np.pi) * 60.0


def min_psf_crop(M_max: float, pupil_radius_mm: float = 2.0,
                 wavelength_nm: float = 550.0, margin: float = 1.5) -> int:
    """
    Smallest even PSF crop that still holds the whole blur disc for |M| <= M_max.
    The disc spans about |M| * pupil_diameter^2 / wavelength bins; cropping
    tighter cuts off part of the blur.
    """
    d = 2.0 * pupil_radius_mm * 1e-3
    n_bins = abs(M_max) * d ** 2 / (wavelength_nm * 1e-9)
    crop = int(np.ceil(n_bins * margin / 2.0)) * 2
    return max(crop, 16)


def calibrated_psf_for_image(
    S: float, C: float, theta_deg: float, image_shape: tuple[int, int],
    pupil_radius_mm: float = 1.5, wavelength_nm: float = 550.0,
    assumed_fov_deg: float = 50.0, native_crop_cap: int = 512,
) -> np.ndarray:
    """
    Blur kernel for a photograph of image_shape. The photo's shorter side is
    assumed to span assumed_fov_deg of the real scene (a phone's main camera:
    about 50 deg on the short side of a 4:3 photo), so that the blur matches
    what this eye would see looking at the scene itself.

    The native PSF has one bin per lambda/d radians (the pupil fills the FFT
    grid); it is resampled by the exact ratio of that bin angle to the angle
    of one image pixel. (Until Nov 2026 the whole crop window, which includes
    a 1.5x margin, was stretched to the blur diameter, so blur came out
    0.5-0.8x too small; together with a 25 deg field assumption this gave
    prescription-dependent errors.)
    """
    from scipy.ndimage import gaussian_filter as _gauss, zoom as _zoom

    h, w = image_shape[:2]
    worst = max(abs(S), abs(S + C))  # the two principal meridian powers
    arcmin_per_image_px = (assumed_fov_deg * 60.0) / min(h, w)

    native_crop = int(np.clip(min_psf_crop(worst + 0.5, pupil_radius_mm), 16, native_crop_cap))
    native_grid = max(2 * native_crop, 256)
    psf_native = psf_from_prescription(
        S, C, theta_deg, pupil_radius_mm=pupil_radius_mm, wavelength_nm=wavelength_nm,
        grid=native_grid, psf_crop=native_crop,
    )
    d = 2.0 * pupil_radius_mm * 1e-3
    bin_arcmin = (wavelength_nm * 1e-9 / d) * (native_grid - 1) / native_grid * (180.0 / np.pi) * 60.0
    zoom = bin_arcmin / arcmin_per_image_px
    if zoom < 1.0:                                   # anti-alias before shrinking
        psf_native = _gauss(psf_native, 0.5 / zoom)
    psf_resampled = _zoom(psf_native, zoom, order=1)
    psf_resampled = np.clip(psf_resampled, 0.0, None)
    total = psf_resampled.sum()
    if psf_resampled.size < 1 or total <= 0:
        psf_resampled = np.ones((1, 1))
        total = 1.0
    return psf_resampled / total


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

def blur_image(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """
    Blur an image (H x W or H x W x C) with a PSF.
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
