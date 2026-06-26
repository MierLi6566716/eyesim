"""eyesim — simulate how a human eye with a given prescription sees the world."""
from .optics import (
    prescription_to_power_vector,
    psf_from_prescription,
    blur_image,
    simulate_vision,
)
__version__ = "0.1.0"
__all__ = [
    "prescription_to_power_vector",
    "psf_from_prescription",
    "blur_image",
    "simulate_vision",
]
