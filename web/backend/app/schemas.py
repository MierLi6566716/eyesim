"""Shapes of the JSON the API sends back."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PSFResponse(BaseModel):
    """Response for the live 3D-bench slider endpoint (no persistence)."""

    psf: list[list[float]]
    preview_png: str  # data: URL
    sharpness: float
    psf_crop: int


class CaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    mode: str
    source_image_url: str
    blurred_image_url: str | None
    predistorted_image_url: str | None
    S: float | None
    C: float | None
    axis: float | None
    M: float
    J0: float
    J45: float
    confidence: float | None
    share_slug: str
    created_at: datetime


class SimulateResponse(BaseModel):
    case: CaseOut
    shader_kernel: list[list[float]]


class DiagnoseResponse(BaseModel):
    case: CaseOut
    shader_kernel: list[list[float]]
    confidence_detail: dict[str, float]
    disclaimer: str


class GalleryPage(BaseModel):
    items: list[CaseOut]
    total: int
    limit: int
    offset: int


class StatsResponse(BaseModel):
    total_cases: int
    simulate_count: int
    diagnose_count: int
    m_histogram: list[dict]  # [{"bucket": "-3 to -2", "count": 5}, ...]
    avg_diagnose_confidence: float | None
