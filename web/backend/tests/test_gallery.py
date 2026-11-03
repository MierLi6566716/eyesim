"""Tests for the gallery endpoints."""

def test_gallery_stats_and_share_card_reflect_created_cases(client, sample_image_bytes):
    r = client.post(
        "/api/cases/simulate",
        files={"image": ("t.png", sample_image_bytes, "image/png")},
        data={"S": "-1.0", "C": "0", "axis": "0"},
    )
    assert r.status_code == 200

    gallery = client.get("/api/gallery", params={"limit": 5}).json()
    assert gallery["total"] >= 1
    assert len(gallery["items"]) >= 1

    slug = gallery["items"][0]["share_slug"]
    card = client.get(f"/api/share/{slug}/card.png")
    assert card.status_code == 200
    assert card.headers["content-type"] == "image/png"
    assert len(card.content) > 1000  # a real rendered PNG, not an empty stub

    detail = client.get(f"/api/share/{slug}")
    assert detail.status_code == 200
    assert detail.json()["share_slug"] == slug

    stats = client.get("/api/stats").json()
    assert stats["total_cases"] >= 1
    assert stats["simulate_count"] >= 1
    assert len(stats["m_histogram"]) == 10


def test_gallery_filters_by_mode(client, sample_image_bytes, synth_crescent_bytes):
    client.post(
        "/api/cases/simulate",
        files={"image": ("t.png", sample_image_bytes, "image/png")},
        data={"S": "-1.0", "C": "0", "axis": "0"},
    )
    client.post("/api/cases/diagnose", files={"image": ("c.png", synth_crescent_bytes, "image/png")})

    sim_only = client.get("/api/gallery", params={"mode": "simulate"}).json()
    diag_only = client.get("/api/gallery", params={"mode": "diagnose"}).json()
    assert all(i["mode"] == "simulate" for i in sim_only["items"])
    assert all(i["mode"] == "diagnose" for i in diag_only["items"])


def test_gallery_case_not_found(client):
    r = client.get("/api/gallery/does-not-exist")
    assert r.status_code == 404


def test_share_card_not_found(client):
    r = client.get("/api/share/does-not-exist/card.png")
    assert r.status_code == 404


def test_share_detail_not_found(client):
    r = client.get("/api/share/does-not-exist")
    assert r.status_code == 404
