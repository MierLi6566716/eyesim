"""Tests for the diagnose endpoint."""

def test_diagnose_creates_case_with_confidence(client, synth_crescent_bytes):
    r = client.post(
        "/api/cases/diagnose",
        files={"image": ("a.png", synth_crescent_bytes, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    case = body["case"]
    assert case["mode"] == "diagnose"
    assert case["confidence"] is not None
    assert 0.0 <= case["confidence"] <= 100.0
    assert "experimental" in body["disclaimer"].lower()
    assert set(body["confidence_detail"].keys()) == {
        "mean", "std", "sharpness", "bright_frac", "corner_mean",
    }


def test_diagnose_confidence_deterministic(client, synth_crescent_bytes):
    r1 = client.post("/api/cases/diagnose", files={"image": ("a.png", synth_crescent_bytes, "image/png")})
    r2 = client.post("/api/cases/diagnose", files={"image": ("b.png", synth_crescent_bytes, "image/png")})
    assert r1.json()["case"]["confidence"] == r2.json()["case"]["confidence"]


def test_diagnose_flags_out_of_distribution_input(client, noise_image_bytes, synth_crescent_bytes):
    """A real, structured crescent should score meaningfully higher than
    unstructured noise with no zeroed corners — the whole point of the
    confidence heuristic (see services/confidence.py)."""
    noise_conf = client.post(
        "/api/cases/diagnose", files={"image": ("n.png", noise_image_bytes, "image/png")}
    ).json()["case"]["confidence"]
    crescent_conf = client.post(
        "/api/cases/diagnose", files={"image": ("c.png", synth_crescent_bytes, "image/png")}
    ).json()["case"]["confidence"]
    assert crescent_conf > noise_conf


def test_diagnose_seeds_model_version_once(client, synth_crescent_bytes):
    client.post("/api/cases/diagnose", files={"image": ("a.png", synth_crescent_bytes, "image/png")})
    client.post("/api/cases/diagnose", files={"image": ("b.png", synth_crescent_bytes, "image/png")})

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from app.db import engine
    from app.models import ModelVersion
    from app.services import cnn_service

    with Session(engine) as db:
        rows = db.scalars(select(ModelVersion).where(ModelVersion.tag == cnn_service.ACTIVE_MODEL_TAG)).all()
    assert len(rows) == 1
    assert rows[0].val_mae_m == cnn_service.ACTIVE_MODEL_STATS["val_mae_m"]
