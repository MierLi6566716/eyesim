"""POST /api/cases/diagnose: run the network on an uploaded eye photo (experimental)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from eyesim.optimize import power_vector_to_prescription

from ..config import settings
from ..db import get_db
from ..models import Case, CaseMode, ModelVersion
from ..schemas import DiagnoseResponse
from ..services import cnn_service, confidence
from ..services.images import load_upload_as_array, to_pupilnet_input
from ..services.physics import simulate_case
from ..services.storage import upload_array

router = APIRouter(prefix="/api", tags=["diagnose"])

DISCLAIMER = (
    "Experimental. PupilNet was trained only on synthetic eccentric-"
    "photorefraction images, not real photos — treat this estimate as a "
    "rough demo, not a measurement. See the confidence score and its "
    "breakdown before trusting any number here."
)


def _active_model(db: Session) -> ModelVersion:
    mv = db.scalar(
        select(ModelVersion).where(ModelVersion.tag == cnn_service.ACTIVE_MODEL_TAG)
    )
    if mv is None:
        mv = ModelVersion(
            tag=cnn_service.ACTIVE_MODEL_TAG,
            checkpoint_path=settings.model_checkpoint_path,
            **cnn_service.ACTIVE_MODEL_STATS,
        )
        db.add(mv)
        db.commit()
        db.refresh(mv)
    return mv


@router.post("/cases/diagnose", response_model=DiagnoseResponse)
async def create_diagnose_case(
    image: UploadFile = File(...),
    pupil_r_mm: float = Form(2.0, ge=1.0, le=4.0),
    is_public: bool = Form(True),
    db: Session = Depends(get_db),
):
    """
    Upload a real eye photo -> PupilNet estimate, clearly labeled
    experimental. Also renders what that estimated prescription implies
    visually (blurred / predistorted), reusing the same physics as
    Simulate mode, so a Diagnose result is never just a bare number.
    """
    arr = await load_upload_as_array(image)
    pupilnet_input = to_pupilnet_input(arr)

    M, J0, J45 = cnn_service.predict(pupilnet_input)
    conf, detail = confidence.score(pupilnet_input)
    S, C, axis = power_vector_to_prescription(M, J0, J45)

    result = simulate_case(arr, S, C, axis, pupil_r_mm)

    source_url = upload_array(arr, prefix="diagnose/source")
    blurred_url = upload_array(result["blurred"], prefix="diagnose/blurred")
    predistorted_url = upload_array(result["predistorted"], prefix="diagnose/predistorted")

    mv = _active_model(db)
    case = Case(
        mode=CaseMode.diagnose,
        model_version_id=mv.id,
        source_image_url=source_url,
        blurred_image_url=blurred_url,
        predistorted_image_url=predistorted_url,
        S=S, C=C, axis=axis,
        M=M, J0=J0, J45=J45,
        confidence=conf,
        is_public=is_public,
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    return DiagnoseResponse(
        case=case,
        shader_kernel=result["shader_kernel"],
        confidence_detail=detail,
        disclaimer=DISCLAIMER,
    )
