"""Gallery endpoints: list saved cases, overall stats, and the share card image."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Case, CaseMode
from ..schemas import CaseOut, GalleryPage, StatsResponse
from ..services.cardgen import render_card

router = APIRouter(prefix="/api", tags=["gallery"])


@router.get("/gallery", response_model=GalleryPage)
def list_gallery(
    mode: CaseMode | None = None,
    limit: int = Query(24, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    stmt = select(Case).where(Case.is_public.is_(True))
    if mode is not None:
        stmt = stmt.where(Case.mode == mode)

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    items = db.scalars(
        stmt.order_by(Case.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return GalleryPage(items=list(items), total=total, limit=limit, offset=offset)


@router.get("/gallery/{case_id}", response_model=CaseOut)
def get_case(case_id: str, db: Session = Depends(get_db)):
    case = db.get(Case, case_id)
    if case is None or not case.is_public:
        raise HTTPException(404, "Case not found.")
    return case


@router.get("/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    public = select(Case).where(Case.is_public.is_(True))
    total = db.scalar(select(func.count()).select_from(public.subquery())) or 0
    sim_count = db.scalar(
        select(func.count()).select_from(
            public.where(Case.mode == CaseMode.simulate).subquery()
        )
    ) or 0
    diag_count = total - sim_count

    m_values = db.scalars(select(Case.M).where(Case.is_public.is_(True))).all()
    histogram = []
    if m_values:
        counts, edges = np.histogram(m_values, bins=10, range=(-6, 4))
        histogram = [
            {"bucket": f"{edges[i]:.1f} to {edges[i + 1]:.1f}", "count": int(c)}
            for i, c in enumerate(counts)
        ]

    avg_conf = db.scalar(
        select(func.avg(Case.confidence)).where(
            Case.is_public.is_(True), Case.mode == CaseMode.diagnose
        )
    )

    return StatsResponse(
        total_cases=total,
        simulate_count=sim_count,
        diagnose_count=diag_count,
        m_histogram=histogram,
        avg_diagnose_confidence=float(avg_conf) if avg_conf is not None else None,
    )


@router.get("/share/{slug}", response_model=CaseOut)
def get_case_by_slug(slug: str, db: Session = Depends(get_db)):
    """JSON case detail for the Share page (distinct from the flat PNG at
    /share/{slug}/card.png, which is for social-preview embedding)."""
    case = db.scalar(
        select(Case).where(Case.share_slug == slug, Case.is_public.is_(True))
    )
    if case is None:
        raise HTTPException(404, "Not found.")
    return case


@router.get("/share/{slug}/card.png")
def get_share_card(slug: str, db: Session = Depends(get_db)):
    case = db.scalar(
        select(Case).where(Case.share_slug == slug, Case.is_public.is_(True))
    )
    if case is None:
        raise HTTPException(404, "Not found.")
    return Response(content=render_card(case), media_type="image/png")
