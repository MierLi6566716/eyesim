"""Tests for the simulate endpoint."""

def test_simulate_creates_case(client, sample_image_bytes):
    r = client.post(
        "/api/cases/simulate",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"S": "-2.25", "C": "-0.75", "axis": "30"},
    )
    assert r.status_code == 200
    body = r.json()
    case = body["case"]
    assert case["mode"] == "simulate"
    assert case["S"] == -2.25
    assert case["C"] == -0.75
    assert case["axis"] == 30.0
    assert case["blurred_image_url"]
    assert case["predistorted_image_url"]
    assert case["confidence"] is None  # Simulate mode never estimates
    assert len(body["shader_kernel"]) > 0


def test_simulate_power_vector_matches_eyesim(client, sample_image_bytes):
    from eyesim.optics import prescription_to_power_vector

    r = client.post(
        "/api/cases/simulate",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"S": "-1.5", "C": "-0.5", "axis": "90"},
    )
    case = r.json()["case"]
    M, J0, J45 = prescription_to_power_vector(-1.5, -0.5, 90)
    assert case["M"] == M
    assert abs(case["J0"] - J0) < 1e-9
    assert abs(case["J45"] - J45) < 1e-9


def test_simulate_rejects_bad_prescription(client, sample_image_bytes):
    r = client.post(
        "/api/cases/simulate",
        files={"image": ("test.png", sample_image_bytes, "image/png")},
        data={"S": "999", "C": "0", "axis": "0"},
    )
    assert r.status_code == 422


def test_simulate_rejects_non_image(client):
    r = client.post(
        "/api/cases/simulate",
        files={"image": ("test.txt", b"not an image", "text/plain")},
        data={"S": "-1.0", "C": "0", "axis": "0"},
    )
    assert r.status_code == 400
