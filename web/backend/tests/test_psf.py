"""Tests for the health and blur-kernel endpoints."""

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_psf_basic(client):
    r = client.get("/api/psf", params={"S": -2.0, "C": -0.5, "axis": 30})
    assert r.status_code == 200
    body = r.json()
    assert body["psf_crop"] >= 4
    assert len(body["psf"]) == body["psf_crop"]
    assert len(body["psf"][0]) == body["psf_crop"]
    assert body["preview_png"].startswith("data:image/png;base64,")
    assert body["sharpness"] >= 0


def test_psf_crop_scales_with_prescription_magnitude(client):
    """The kernel is sized against a real angular blur diameter (see
    services/physics.py::_psf_for_image), not an arbitrary constant — a
    much stronger prescription should produce a visibly larger kernel."""
    mild = client.get("/api/psf", params={"S": -0.5, "C": 0, "axis": 0}).json()
    strong = client.get("/api/psf", params={"S": -6.0, "C": 0, "axis": 0}).json()
    assert strong["psf_crop"] > mild["psf_crop"]


def test_psf_rejects_out_of_range_sphere(client):
    r = client.get("/api/psf", params={"S": 999, "C": 0, "axis": 0})
    assert r.status_code == 422


def test_psf_never_persists(client):
    """The live-slider bench must not write to the DB (see plan/schema notes
    in models.py) — hitting it repeatedly shouldn't grow the gallery."""
    before = client.get("/api/stats").json()["total_cases"]
    for _ in range(3):
        client.get("/api/psf", params={"S": -1.0, "C": 0, "axis": 0})
    after = client.get("/api/stats").json()["total_cases"]
    assert before == after
