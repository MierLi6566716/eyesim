"""Checks that the fake eye photos follow the physics: blind band at -1/d, crescent flips sides, etc."""

import numpy as np

from eyesim.photorefraction import Eye, Geometry, Scene, reflex, render_pair

GEOM = Geometry(d_mm=500.0, ecc_mm=6.0, aperture_r_mm=1.5, source_r_mm=0.5)


def _total(rx, **kw):
    I, _, _ = reflex(Eye(M=rx, pupil_r_mm=3.0, scatter_mrad=0.5, **kw), GEOM, image_size=32)
    return float(I.sum())


def test_dark_zone_is_centred_on_minus_one_over_d():
    # Chen et al. 2003: with a small pupil the centre of the dark zone is -1/d.
    rx = np.arange(-6.0, 2.01, 0.25)
    tot = np.array([_total(r) for r in rx])
    dark = rx[tot < 1e-3 * tot.max()]
    assert abs(0.5 * (dark.min() + dark.max()) - (-2.0)) <= 0.25


def test_crescent_side_flips_across_the_null():
    # LED on +x: hyperopic of the null lights the -x side, myopic the +x side.
    def centroid_x(rx):
        I, _, xy = reflex(Eye(M=rx, pupil_r_mm=3.0), GEOM, image_size=48)
        return float((I[0] * xy[..., 0]).sum() / I[0].sum())
    assert centroid_x(+4.0) < -0.3
    assert centroid_x(-8.0) > +0.3


def test_spherical_aberration_moves_the_null_hyperopic():
    rx = np.arange(-5.0, 2.01, 0.1)
    plain = rx[np.argmin([_total(r) for r in rx])]
    with_sa = rx[np.argmin([_total(r, sa_d_per_mm2=0.16) for r in rx])]
    assert with_sa > plain


def test_astigmatism_separates_the_two_leds():
    # Pure J0 changes the horizontal and vertical meridians in opposite directions.
    I, _, _ = reflex(Eye(M=2.0, J0=1.5, pupil_r_mm=3.0), GEOM, image_size=32)
    assert abs(I[0].sum() - I[1].sum()) > 0.05 * max(I[0].sum(), I[1].sum())
    I, _, _ = reflex(Eye(M=2.0, pupil_r_mm=3.0), GEOM, image_size=32)
    assert abs(I[0].sum() - I[1].sum()) < 0.02 * I[0].sum()


def test_render_pair_is_an_8_bit_frame():
    img = render_pair(Eye(M=3.0), GEOM, Scene(image_size=32, glint_amp=2.0), np.random.default_rng(0))
    assert img.shape == (32, 32, 2) and img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert np.allclose(img * 255, np.round(img * 255), atol=1e-3)
