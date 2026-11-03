"""POST /api/cases/simulate: show how an uploaded image looks to a given prescription."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from eyesim.optics import prescription_to_power_vector

from ..db import get_db
from ..models import Case, CaseMode
from ..schemas import SimulateResponse
from ..services.images import load_upload_as_array
from ..services.physics import simulate_case
from ..services.storage import upload_array

router = APIRouter(prefix="/api", tags=["simulate"])


@router.post("/cases/simulate", response_model=SimulateResponse)
async def create_simulate_case(
    image: UploadFile = File(...),
    S: float = Form(..., ge=-8.0, le=6.0),
    C: float = Form(0.0, ge=-4.0, le=0.0),
    axis: float = Form(0.0, ge=0.0, le=180.0),
    pupil_r_mm: float = Form(2.0, ge=1.0, le=4.0),
    is_public: bool = Form(True),
    db: Session = Depends(get_db),
):
    """
    Upload any photo + a known/chosen prescription -> what that eye sees,
    the Wiener-predistorted "corrected" version, and a small blur kernel
    for the live screen-filter shader. No CNN involved — this mode never
    estimates anything, it only applies known physics (eyesim.optics /
    eyesim.predistort), so its accuracy is exact by construction.
    """
    arr = await load_upload_as_array(image)
    result = simulate_case(arr, S, C, axis, pupil_r_mm)
    # eyesim.optics returns numpy float64 for J0/J45 (np.cos/np.sin internally);
    # psycopg2 (unlike sqlite3) fails to adapt those, so coerce at this boundary.
    M, J0, J45 = (float(v) for v in prescription_to_power_vector(S, C, axis))

    source_url = upload_array(arr, prefix="simulate/source")
    blurred_url = upload_array(result["blurred"], prefix="simulate/blurred")
    predistorted_url = upload_array(result["predistorted"], prefix="simulate/predistorted")

    case = Case(
        mode=CaseMode.simulate,
        source_image_url=source_url,
        blurred_image_url=blurred_url,
        predistorted_image_url=predistorted_url,
        S=S, C=C, axis=axis,
        M=M, J0=J0, J45=J45,
        is_public=is_public,
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    return SimulateResponse(case=case, shader_kernel=result["shader_kernel"])
