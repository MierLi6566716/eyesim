"""
Makes fake eye photos for training the network.

The camera sits a distance d in front of the eye with an infrared LED a few mm
beside its lens. LED light goes into the eye, lights a spot on the retina, and
comes back out through the pupil. If the eye is out of focus for the camera,
only part of that returning light gets into the camera lens, so the pupil
shows a bright crescent. Which side the crescent is on and how big it is tells
you the eye's prescription. This file traces those rays.

How it works (lengths in mm, powers in dioptres D):

  * A ray from the LED enters the pupil at point p1, lands on the retina, and
    leaves again through pupil point p2. Its spot in the camera plane is
        x_c = x_led + g(p2) - g(p1),   g(p) = p + d * (slope of the eye's wavefront at p)
  * Pixel p2 is as bright as the share of p1 points whose return ray lands
    inside the camera lens opening.
  * For pure defocus g(p) = (1 + d * Rx) p, so at Rx = -1/d every ray goes
    straight back to the LED, nothing reaches the camera, and the pupil is
    dark. That is the "blind band" (dark zone) centred on -1000/d D.

On top of that it adds what real photos have: spherical aberration, coma,
retinal scatter, the uniform red-reflex glow, the corneal glint of each LED, a
faint second glint, iris texture, uneven lighting, eyelid and lashes, tear
film, lens blur, sensor noise.

Labels are the refraction at 850 nm. A human eye is about LCA_850_D more
hyperopic in infrared than its glasses prescription; ``ir_to_visible``
converts.

Image convention: x to the right, y UP (row 0 is the top). Channel 0 is the
LED at led_angles_deg[0] (0 deg, right of the lens as seen in the image),
channel 1 the LED at 90 deg (above the lens). An eye more myopic than the
null shows its crescent on the LED's side, a more hyperopic one on the far side.
Based on Chen, Tan & Lewis, Optics Express 11(14), 2003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import ncx2

# Longitudinal chromatic aberration of the human eye, 850 nm relative to 589 nm,
# from the chromatic reduced eye (Thibos 1992): 1.68524 - 633.46 / (850 - 214.102).
LCA_850_D = 0.69

# Longitudinal spherical aberration of the schematic eyes in the paper's Fig. 1,
# read at 7 mm pupil diameter and expressed as s = LSA / r^2  [D / mm^2].
SA_NAVARRO = 2.00 / 3.5 ** 2
SA_ARIZONA = 2.25 / 3.5 ** 2
SA_LIOU = 0.75 / 3.5 ** 2


def ir_to_visible(M_ir: float) -> float:
    """Visible-light spherical equivalent of a human eye measured at 850 nm."""
    return M_ir - LCA_850_D


@dataclass(frozen=True)
class Geometry:
    """Where the camera and LEDs are. Defaults: the IMX708 rig at 37 cm."""

    d_mm: float = 370.0              # eye reference plane -> camera/LED plane
    ecc_mm: float = 10.0             # LED centre -> camera aperture centre
    aperture_r_mm: float = 1.3       # camera entrance pupil radius
    source_r_mm: float = 0.8         # LED emitter radius
    led_angles_deg: tuple = (0.0, 90.0)
    # Optional straight shield between aperture and LED (paper's instrument):
    # aperture points farther than this along the LED direction are blocked.
    knife_edge_mm: float | None = None


@dataclass(frozen=True)
class Eye:
    """One eye. M, J0, J45 are the label (dioptres, instrument wavelength)."""

    M: float = 0.0
    J0: float = 0.0
    J45: float = 0.0
    pupil_r_mm: float = 3.0
    sa_d_per_mm2: float = 0.0        # longitudinal spherical aberration / r^2
    coma_d_per_mm: tuple = (0.0, 0.0)
    apodization_per_mm2: float = 0.0  # Stiles-Crawford-like weight exp(-rho r^2)
    scatter_mrad: float = 1.0        # lateral scatter in the retina, as an angle
    pupil_ellipse: tuple = (1.0, 0.0)  # (minor/major ratio, major-axis angle deg)


@dataclass(frozen=True)
class Scene:
    """Everything that is not optics: framing, extra light, and the sensor."""

    image_size: int = 64
    fill: float = 0.8                # pupil diameter / frame width
    centre_frac: tuple = (0.0, 0.0)  # pupil centre offset, fraction of pupil radius
    # Camera gain. The frame is exposure * I / I_REF, so a typical bright
    # crescent reaches full scale at exposure = 1 and clips above that. Shutter
    # and gain are locked on the rig, so dark-zone frames really are dark.
    exposure: float = 0.8
    red_reflex: float = 0.0          # uniform in-pupil background, fraction of full scale
    surround: float = 0.05           # level of the iris around the pupil
    iris_amp: float = 0.0            # contrast of the radial iris texture
    iris_r_mm: float = 5.9           # beyond this is sclera
    light_grad: tuple = (0.0, 0.0)   # (strength, direction deg) of uneven light on the face
    ring: tuple = (0.0, 0.0)         # (extra level, width as fraction of radius) just outside
    glint_amp: float = 0.0           # corneal / lens-surface reflection of the LED
    glint_sigma_px: float = 1.0
    glint_pos_frac: tuple = (0.0, 0.0)   # glint centre for an on-axis source
    glint_led_shift_frac: float = 0.15   # extra shift toward each LED
    glint2_amp: float = 0.0          # faint reflection from the back of the eye's lens
    lashes: int = 0                  # number of eyelashes crossing the top of the frame
    lid_frac: float = 0.0            # fraction of pupil diameter hidden by the upper lid
    tear_amp: float = 0.0            # multiplicative tear-film ripple amplitude
    blur_px: float = 0.6             # camera defocus / lens blur
    read_noise: float = 0.01
    shot_noise: float = 0.01
    bits: int = 8
    ambient: float = 0.0             # stray light added to every pixel
    # Reflections off a trial lens or the eye's own front surfaces: bright,
    # often clipped blobs at fixed places, the same in both LED frames
    # (x, y in pupil radii, sigma in px, amplitude).
    reflections: tuple = ()


# Fraction of the entrance pupil collected at the peak of a bright crescent
# on the default rig: this maps to full scale at exposure 1.
I_REF = 0.03

_SUNFLOWER_CACHE: dict = {}


def _sunflower(n: int) -> np.ndarray:
    """n points spread evenly over the unit disc, shape (n, 2)."""
    if n not in _SUNFLOWER_CACHE:
        k = np.arange(n) + 0.5
        r = np.sqrt(k / n)
        th = k * np.pi * (3.0 - np.sqrt(5.0))
        _SUNFLOWER_CACHE[n] = np.stack([r * np.cos(th), r * np.sin(th)], 1).astype(np.float32)
    return _SUNFLOWER_CACHE[n]


def _ellipse_matrix(eye: Eye) -> np.ndarray:
    """Maps the unit disc onto the (possibly elliptical) pupil."""
    ratio, ang = eye.pupil_ellipse
    c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    rot = np.array([[c, -s], [s, c]])
    return (rot @ np.diag([eye.pupil_r_mm, eye.pupil_r_mm * ratio]) @ rot.T).astype(np.float32)


def _g(p: np.ndarray, eye: Eye, d_mm: float) -> np.ndarray:
    """g(p) = p + d * grad W(p), for points p of shape (..., 2) in mm."""
    x, y = p[..., 0], p[..., 1]
    r2 = x * x + y * y
    gx = (eye.M + eye.J0) * x + eye.J45 * y
    gy = (eye.M - eye.J0) * y + eye.J45 * x
    gx = gx - eye.sa_d_per_mm2 * r2 * x
    gy = gy - eye.sa_d_per_mm2 * r2 * y
    cx, cy = eye.coma_d_per_mm
    if cx or cy:
        dot = cx * x + cy * y
        gx = gx + 2.0 * x * dot + r2 * cx
        gy = gy + 2.0 * y * dot + r2 * cy
    k = d_mm * 1e-3
    return np.stack([x + k * gx, y + k * gy], -1).astype(np.float32)


class _Aperture:
    """Camera aperture response A(q): probability that a return ray aimed at
    camera-plane point q (relative to the aperture centre) is collected, after
    blurring by the source size and retinal scatter."""

    def __init__(self, geom: Geometry, sigma_mm: float, led_dir: np.ndarray):
        self.sigma = max(float(sigma_mm), 0.05)
        self.a = geom.aperture_r_mm
        self.knife = geom.knife_edge_mm
        if self.knife is None:
            # Disc blurred by an isotropic Gaussian = non-central chi-square CDF.
            rmax = self.a + 6.0 * self.sigma
            self.r = np.linspace(0.0, rmax, 400)
            self.table = ncx2.cdf((self.a / self.sigma) ** 2, 2, (self.r / self.sigma) ** 2)
            self.table = self.table.astype(np.float32)
        else:
            half = self.a + 5.0 * self.sigma
            n = 161
            ax = np.linspace(-half, half, n)
            X, Y = np.meshgrid(ax, ax)
            mask = (X ** 2 + Y ** 2 <= self.a ** 2) & (X * led_dir[0] + Y * led_dir[1] <= self.knife)
            self.step = ax[1] - ax[0]
            self.half = half
            self.grid = gaussian_filter(mask.astype(np.float32), self.sigma / self.step, mode="constant")

    def __call__(self, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
        if self.knife is None:
            return np.interp(np.sqrt(qx * qx + qy * qy), self.r, self.table, right=0.0)
        cols = (qx + self.half) / self.step
        rows = (qy + self.half) / self.step
        return map_coordinates(self.grid, [rows.ravel(), cols.ravel()], order=1,
                               mode="constant").reshape(qx.shape)


def reflex(eye: Eye, geom: Geometry = Geometry(), image_size: int = 64, fill: float = 0.8,
           centre_frac: tuple = (0.0, 0.0), n_entry: int = 300):
    """
    Noise-free retinal reflex for each LED.

    Returns
    -------
    I    : float32 (n_led, H, W)  fraction of the entrance pupil collected (0..1)
    mask : bool (H, W)            pixels inside the pupil
    xy   : float32 (H, W, 2)      pupil-plane coordinates of each pixel, mm,
                                  relative to the pupil centre
    """
    R = eye.pupil_r_mm
    half = R / fill
    ax = (np.arange(image_size) + 0.5) / image_size * 2.0 * half - half
    X, Y = np.meshgrid(ax, -ax)                       # row 0 is +y
    X = X - centre_frac[0] * R
    Y = Y - centre_frac[1] * R
    xy = np.stack([X, Y], -1).astype(np.float32)

    E = _ellipse_matrix(eye)
    unit = xy @ np.linalg.inv(E).T
    mask = (unit ** 2).sum(-1) <= 1.0

    p1 = _sunflower(n_entry) @ E.T
    p2 = xy[mask]
    w1 = np.exp(-eye.apodization_per_mm2 * (p1 ** 2).sum(-1)).astype(np.float32)
    w2 = np.exp(-eye.apodization_per_mm2 * (p2 ** 2).sum(-1)).astype(np.float32)

    diff = _g(p2, eye, geom.d_mm)[:, None, :] - _g(p1, eye, geom.d_mm)[None, :, :]
    sigma = np.hypot(geom.d_mm * eye.scatter_mrad * 1e-3, geom.source_r_mm / 2.0)

    out = np.zeros((len(geom.led_angles_deg), image_size, image_size), np.float32)
    ap = None
    for c, ang in enumerate(geom.led_angles_deg):
        led_dir = np.array([np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))])
        if ap is None or geom.knife_edge_mm is not None:   # the plain disc is the same for every LED
            ap = _Aperture(geom, sigma, led_dir)
        e = geom.ecc_mm * led_dir
        A = ap(diff[..., 0] + e[0], diff[..., 1] + e[1])
        out[c][mask] = (A * w1[None, :]).sum(1) / w1.sum() * w2
    return out, mask, xy


def render_pair(eye: Eye, geom: Geometry = Geometry(), scene: Scene = Scene(),
                rng: np.random.Generator | None = None, n_entry: int = 300,
                return_peak: bool = False):
    """
    Camera frames for both LEDs, with background, glint, blur, noise and
    quantisation. Returns float32 (H, W, n_led) in [0, 1].

    With ``return_peak`` also returns the peak reflex level before clipping
    (1 = full scale), which says whether the frame carries a usable crescent
    or sits in the instrument's dark zone.
    """
    rng = rng or np.random.default_rng()
    n = scene.image_size
    I, mask, xy = reflex(eye, geom, n, scene.fill, scene.centre_frac, n_entry)
    R = eye.pupil_r_mm
    r_unit = np.sqrt((xy ** 2).sum(-1)) / R
    px_per_r = n * scene.fill / 2.0

    I = I * (scene.exposure / I_REF)
    peak = float(I.max())

    if scene.tear_amp > 0.0:                      # tear film: a slow ripple over the eye
        ripple = gaussian_filter(rng.standard_normal((n, n)), n / 14.0)
        ripple = 1.0 + scene.tear_amp * ripple / (np.abs(ripple).max() + 1e-9)

    out = np.empty((n, n, I.shape[0]), np.float32)
    lid = (xy[..., 1] / R) > (1.0 - 2.0 * scene.lid_frac) if scene.lid_frac > 0 else None
    ring_level, ring_w = scene.ring

    # Everything outside the pupil is the same in both frames: iris, sclera,
    # uneven light, then lashes darken the whole frame.
    back = np.full((n, n), scene.surround, np.float32)
    if scene.iris_amp > 0.0:
        back *= 1.0 + scene.iris_amp * _iris_texture(rng, xy, R, scene.iris_r_mm)
    back[r_unit * R > scene.iris_r_mm] = scene.surround * 1.6 + 0.05
    if ring_level > 0.0:
        back[(r_unit > 1.0) & (r_unit < 1.0 + ring_w)] += ring_level
    g, gang = scene.light_grad
    if g:
        a = np.deg2rad(gang)
        back *= np.clip(1.0 + g * (xy[..., 0] * np.cos(a) + xy[..., 1] * np.sin(a)) / (R / scene.fill), 0.2, None)
    lash = _lashes(rng, n, scene.lashes) if scene.lashes else None

    for c, ang in enumerate(geom.led_angles_deg):
        img = back.copy()
        img[mask] = I[c][mask] + scene.red_reflex
        if lid is not None:
            img[lid] = scene.surround * 1.5 + 0.02

        if scene.glint_amp > 0.0:
            a = np.deg2rad(ang)
            gx = (scene.glint_pos_frac[0] + scene.glint_led_shift_frac * np.cos(a)) * R
            gy = (scene.glint_pos_frac[1] + scene.glint_led_shift_frac * np.sin(a)) * R
            for amp, (px, py), sig in ((scene.glint_amp, (gx, gy), scene.glint_sigma_px),
                                       (scene.glint_amp * scene.glint2_amp, (-0.6 * gx, -0.6 * gy), 1.5)):
                if amp > 0.0:
                    d2 = ((xy[..., 0] - px) ** 2 + (xy[..., 1] - py) ** 2) * (px_per_r / R) ** 2
                    img = img + amp * np.exp(-d2 / (2.0 * sig ** 2))

        for rx, ry, rs, ra in scene.reflections:
            d2 = ((xy[..., 0] - rx * R) ** 2 + (xy[..., 1] - ry * R) ** 2) * (px_per_r / R) ** 2
            img = img + ra * np.exp(-d2 / (2.0 * rs ** 2))

        if scene.tear_amp > 0.0:
            img = img * ripple
        if lash is not None:
            img = img * lash
        img = img + scene.ambient
        if scene.blur_px > 0.0:
            img = gaussian_filter(img, scene.blur_px)
        noise = np.sqrt(scene.read_noise ** 2 + scene.shot_noise ** 2 * np.clip(img, 0, None))
        img = img + noise * rng.standard_normal((n, n)).astype(np.float32)
        levels = 2 ** scene.bits - 1
        out[..., c] = np.round(np.clip(img, 0.0, 1.0) * levels) / levels
    return (out, peak) if return_peak else out


def _iris_texture(rng, xy, R, iris_r_mm):
    """Radial streaks like iris fibres, zero mean, about -1..1."""
    nt, nr = 160, 24
    tex = gaussian_filter(rng.standard_normal((nt, nr)), (0.8, 3.0), mode=("wrap", "nearest"))
    tex /= np.abs(tex).max() + 1e-9
    th = (np.arctan2(xy[..., 1], xy[..., 0]) + np.pi) / (2 * np.pi) * nt
    r = (np.hypot(xy[..., 0], xy[..., 1]) - R) / max(iris_r_mm - R, 0.5) * (nr - 1)
    return map_coordinates(tex, [th.ravel(), np.clip(r, 0, nr - 1).ravel()], order=1,
                           mode="grid-wrap").reshape(th.shape).astype(np.float32)


def _lashes(rng, n, count):
    """Multiplicative shadow of a few eyelashes hanging into the top of the frame."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    shade = np.ones((n, n), np.float32)
    for _ in range(count):
        x0, ang = rng.uniform(0, n), np.deg2rad(rng.uniform(-35, 35))
        length, width, depth = rng.uniform(0.1, 0.45) * n, rng.uniform(0.4, 1.2), rng.uniform(0.3, 0.8)
        dx, dy = np.sin(ang), np.cos(ang)                  # pointing down into the frame
        t = np.clip((xx - x0) * dx + yy * dy, 0, length)
        d2 = (xx - x0 - t * dx) ** 2 + (yy - t * dy) ** 2
        shade *= 1.0 - depth * np.exp(-d2 / (2 * width ** 2))
    return shade


# ------------------------------------------------------------------ sampling

M_RANGE_D = (-9.0, 5.0)              # refraction (850 nm, eye + any trial lens) the network learns


def sample_eye(rng: np.random.Generator, m_range=M_RANGE_D) -> Eye:
    """Draw one human eye: refraction plus the optics that differ between people."""
    # Refraction: half spread evenly, half like a real population.
    if rng.random() < 0.5:
        M = rng.uniform(*m_range)
    else:
        M = float(np.clip(rng.normal(-1.0, 2.5), *m_range))
    if rng.random() < 0.25:
        J0 = J45 = 0.0
    else:
        cyl = min(abs(rng.normal(0.0, 1.0)), 3.0)
        th = rng.uniform(0.0, np.pi)
        J0, J45 = 0.5 * cyl * np.cos(2 * th), 0.5 * cyl * np.sin(2 * th)
    # Spherical aberration at 7 mm from +0.75 D (Liou) to +2.25 D (Arizona eye),
    # with a lot of variation between people (Chen 2003, Fig. 1 and 5).
    sa = float(np.clip(rng.normal(0.12, 0.07), -0.05, 0.35))
    coma = rng.normal(0.0, 0.02, 2)
    return Eye(M=float(M), J0=float(J0), J45=float(J45),
               pupil_r_mm=float(rng.uniform(1.5, 4.0)),                 # 3 to 8 mm pupil
               sa_d_per_mm2=sa, coma_d_per_mm=(float(coma[0]), float(coma[1])),
               apodization_per_mm2=float(rng.uniform(0.0, 0.12)),       # Stiles-Crawford
               scatter_mrad=float(np.exp(rng.uniform(np.log(0.7), np.log(8.0)))),
               pupil_ellipse=(float(rng.uniform(0.9, 1.0)), float(rng.uniform(0, 180))))  # gaze angle


# Rig ranges the network is trained on. Distance is drawn evenly in 1/d,
# because the optics scale with 1/d (the blind band sits at -1000/d D).
D_RANGE_MM = (200.0, 1000.0)
ECC_RANGE_MM = (6.0, 20.0)
APERTURE_RANGE_MM = (1.5, 4.5)       # lens opening diameter = focal length / f-number


def sample_geometry(rng: np.random.Generator, d_range=D_RANGE_MM, ecc_range=ECC_RANGE_MM,
                    aperture_range=APERTURE_RANGE_MM) -> Geometry:
    """A random rig. d, ecc and aperture are told to the network; LED size is not."""
    return Geometry(
        d_mm=float(1000.0 / rng.uniform(1000.0 / d_range[1], 1000.0 / d_range[0])),
        ecc_mm=float(rng.uniform(*ecc_range)),
        aperture_r_mm=float(rng.uniform(*aperture_range) / 2.0),
        source_r_mm=float(rng.uniform(0.3, 1.5)),
    )


def sample_scene(rng: np.random.Generator, image_size: int = 64) -> Scene:
    """Framing, everything around the pupil, and the sensor."""
    fill = float(rng.uniform(0.5, 0.95))
    return Scene(
        image_size=image_size,
        fill=fill,
        centre_frac=(float(rng.normal(0, 0.08)), float(rng.normal(0, 0.08))),
        exposure=float(np.exp(rng.uniform(np.log(0.25), np.log(4.0)))),   # > 1 clips bright crescents
        # Light scattered many times in the eye: a uniform glow with no
        # refraction information, stronger in infrared (Chen 2003, section 4).
        red_reflex=float(rng.exponential(0.08)),
        surround=float(rng.uniform(0.0, 0.35)),
        iris_amp=float(rng.uniform(0.0, 0.6)),
        light_grad=(float(rng.uniform(0.0, 0.4)), float(rng.uniform(0, 360))),
        ring=(float(rng.uniform(0.0, 0.3)) if rng.random() < 0.5 else 0.0, float(rng.uniform(0.05, 0.4))),
        glint_amp=float(rng.uniform(0.5, 3.0)) if rng.random() < 0.9 else 0.0,
        glint_sigma_px=float(rng.uniform(0.5, 2.5)),
        glint_pos_frac=(float(rng.normal(0, 0.15)), float(rng.normal(0, 0.15))),
        glint_led_shift_frac=float(rng.uniform(0.02, 0.35)),
        glint2_amp=float(rng.uniform(0.0, 0.1)),
        lid_frac=float(rng.uniform(0.05, 0.3)) if rng.random() < 0.25 else 0.0,
        lashes=int(rng.integers(2, 10)) if rng.random() < 0.3 else 0,
        tear_amp=float(rng.uniform(0.1, 0.5)) if rng.random() < 0.15 else 0.0,
        # Tried 12 Nov 2026: blur up to 0.22 of the pupil radius (the model-eye photos fit ~0.15).
        # Networks trained that way read the real photos 1.2-1.4 D too weak, so reverted.
        blur_px=float(rng.uniform(0.3, 1.8)),
        read_noise=float(rng.uniform(0.004, 0.03)),
        shot_noise=float(rng.uniform(0.005, 0.04)),
        ambient=float(rng.exponential(0.02)),
        # Trial-lens / front-surface reflections (model-eye photos, Nov 2026).
        reflections=tuple((float(rng.uniform(-1.3, 1.3)), float(rng.uniform(-1.3, 1.3)),
                           float(rng.uniform(1.0, 4.0)), float(rng.uniform(0.5, 4.0)))
                          for _ in range(int(rng.integers(1, 3)) if rng.random() < 0.35 else 0)),
    )


def aux_vector(d_mm: float, ecc_mm: float, pupil_r_mm: float, aperture_mm: float) -> np.ndarray:
    """The known rig numbers, scaled to about -1..1, as the network's second input."""
    return np.array([(1000.0 / d_mm - 2.5) / 1.5, (ecc_mm - 13.0) / 7.0, (pupil_r_mm - 2.75) / 1.25,
                     (aperture_mm - 3.0) / 1.5], dtype=np.float32)
