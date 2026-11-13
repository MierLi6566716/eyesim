"""GET /api/psf: the blur kernel for a prescription, for the live slider."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..schemas import PSFResponse
from ..services.physics import psf_bench

router = APIRouter(prefix="/api", tags=["psf"])


@router.get("/psf", response_model=PSFResponse)
def get_psf(
    S: float = Query(0.0, ge=-8.0, le=6.0, description="Sphere, dioptres"),
    C: float = Query(0.0, ge=-4.0, le=0.0, description="Cylinder, dioptres"),
    axis: float = Query(0.0, ge=0.0, le=180.0, description="Axis, degrees"),
    pupil_r_mm: float = Query(1.5, ge=1.0, le=4.0),
):
    """
    Live physics for the 3D optics-bench page. Stateless — called on every
    debounced slider drag, never persisted. See services/physics.py.
    """
    return psf_bench(S, C, axis, pupil_r_mm)
