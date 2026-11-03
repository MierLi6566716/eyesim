"""
eyesim: simulate how an eye with a given prescription sees, and measure that
prescription from infrared photos of the pupil.

  optics.py           prescription -> blur the eye sees
  predistort.py       pre-blur a screen image so a given eye sees it sharp
  optimize.py         find the lens correction that makes the view sharpest
  photorefraction.py  fake infrared pupil photos for training the network
  cnn.py              the network (PupilNet) that reads those photos
  pipeline.py         the whole loop in simulation: photo -> network -> correction
"""
from .optics import (
    prescription_to_power_vector,
    psf_from_prescription,
    blur_image,
    simulate_vision,
)
from .predistort import (
    predistort,
    simulate_display_chain,
    wiener_deconvolve,
)
from .optimize import (
    sharpness,
    power_vector_to_prescription,
    run_closed_loop,
)

__version__ = "0.6.0"
__all__ = [
    "prescription_to_power_vector",
    "psf_from_prescription",
    "blur_image",
    "simulate_vision",
    "predistort",
    "simulate_display_chain",
    "wiener_deconvolve",
    "sharpness",
    "power_vector_to_prescription",
    "run_closed_loop",
]
