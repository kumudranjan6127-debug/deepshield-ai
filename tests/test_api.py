"""The HTTP surface: every endpoint, and the shape of what comes back.

These are contract tests. The frontend reads `error` as a plain string and
`riskLevel` in title case; both have been true since before the fields were
documented, and both would break silently if someone tidied them.
"""
import io
import os

import pytest

pytestmark = pytest.mark.api


def upload(client, path, **fields):
    with open(path, "rb") as f:
        data = {"file": (io.BytesIO(f.read()), os.path.basename(path))}
    data.update(fields)
    return client.post("/api/analyze", data=data, content_type="multipart/form-data")


# ----------------------------------------------------------------- health

def test_health_reports_the_engine(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["engine"] in ("live", "echo")


def test_health_carries_the_model_identity(client, engine_ready):
    body = client.get("/api/health").get_json()
    for key in ("model_name", "architecture", "version", "runtime", "input_size"):
        assert body.get(key), f"/api/health is missing {key}"
    assert body["model"]["name"] == body["architecture"]


def test_health_publishes_the_certainty_bands(client):
    """The browser must not hold its own copy of these thresholds."""
    bands = client.get("/api/health").get_json()["certainty_bands"]
    assert bands and bands[0]["to"] == 100 and bands[-1]["from"] == 0
    assert {b["key"] for b in bands} >= {"very_strong", "strong", "uncertain"}


def test_health_admits_it_is_not_calibrated(client):
    assert client.get("/api/health").get_json()["calibrated"] is False


# ---------------------------------------------------------------- analyze

def test_analyze_multipart_image(client, engine_ready, fake_face):
    body = upload(client, fake_face, fileType="image").get_json()
    assert body["ok"] is True
    assert body["prediction"] in ("real", "deepfake")
    assert 50 <= body["confidence"] <= 100
    assert body["riskLevel"] in ("Low", "Medium", "High")
    assert body["risk"] == body["riskLevel"].lower()
    assert body["certainty"] in ("very_strong", "strong", "uncertain", "low_evidence")
    assert body["framesAnalyzed"] == 1


def test_analyze_via_staged_upload(client, engine_ready, fake_face):
    with open(fake_face, "rb") as f:
        staged = client.post("/api/upload", data={
            "file": (io.BytesIO(f.read()), "face.jpeg")},
            content_type="multipart/form-data").get_json()
    assert staged["ok"] is True and staged["uploadId"]

    body = client.post("/api/analyze", json={
        "uploadId": staged["uploadId"], "fileName": "face.jpeg",
        "fileType": "image"}).get_json()
    assert body["ok"] is True
    assert body["prediction"] in ("real", "deepfake")


def test_the_two_paths_agree(client, engine_ready, fake_face):
    """Staging a file and posting it directly must not give different
    answers — they are the same bytes through the same engine."""
    direct = upload(client, fake_face, fileType="image").get_json()
    with open(fake_face, "rb") as f:
        staged = client.post("/api/upload", data={
            "file": (io.BytesIO(f.read()), "face.jpeg")},
            content_type="multipart/form-data").get_json()
    via_id = client.post("/api/analyze", json={
        "uploadId": staged["uploadId"], "fileName": "face.jpeg",
        "fileType": "image"}).get_json()

    assert direct["prediction"] == via_id["prediction"]
    assert direct["confidence"] == via_id["confidence"]
    assert direct["ensemble"][0]["pFake"] == via_id["ensemble"][0]["pFake"]


def test_analyze_without_a_file_or_id_echoes(client):
    """Metadata alone is the demo path: a deterministic stand-in verdict,
    openly labelled, never a claim about real media."""
    body = client.post("/api/analyze", json={
        "fileName": "holiday.mp4", "fileType": "video", "fileSize": 1234}).get_json()
    assert body["ok"] is True
    assert body["prediction"] in ("real", "deepfake")
    # No model ran, so no model votes may be presented as if one had
    assert "ensemble" not in body and "explain" not in body


def test_the_echo_verdict_is_stable(client):
    payload = {"fileName": "holiday.mp4", "fileType": "video", "fileSize": 1234}
    first = client.post("/api/analyze", json=payload).get_json()
    second = client.post("/api/analyze", json=payload).get_json()
    assert (first["prediction"], first["confidence"]) == \
           (second["prediction"], second["confidence"])


def test_image_analysis_explains_itself(client, engine_ready, fake_face):
    explain = upload(client, fake_face, fileType="image").get_json().get("explain")
    assert explain, "an image verdict should come with an explanation"
    assert explain["method"] == "occlusion sensitivity"
    assert "sensitive" in explain["note"].lower()
    assert "attention" not in explain["note"].lower(), \
        "occlusion sensitivity does not measure attention"
    assert explain["heatmapDataUrl"].startswith("data:image/")


# --------------------------------------------------------------- feedback

def test_feedback_accepts_a_boolean(client):
    r = client.post("/api/feedback", json={
        "scanId": "SCAN-TEST", "prediction": "real", "confidence": 90,
        "fileType": "image", "agree": True})
    assert r.status_code == 200 and r.get_json()["ok"] is True


@pytest.mark.parametrize("agree", ["yes", 1, None, "", []])
def test_feedback_rejects_anything_else(client, agree):
    r = client.post("/api/feedback", json={"agree": agree})
    assert r.status_code == 400
    assert r.get_json()["error_code"] == "BAD_FIELD"


# ------------------------------------------------------------ error shape

def test_api_errors_share_one_shape(client):
    body = client.post("/api/feedback", json={"agree": "nope"}).get_json()
    assert body["ok"] is False
    assert isinstance(body["error"], str), \
        "the frontend renders `error` directly; it must stay a string"
    assert isinstance(body["error_code"], str)


def test_a_missing_api_route_is_json(client):
    r = client.get("/api/not-an-endpoint")
    assert r.status_code == 404
    assert r.is_json


def test_a_missing_page_stays_html(client):
    """Phase 1 broke this: the API error handler swallowed HTTPException
    and turned a plain 404 into a 500."""
    r = client.get("/definitely-not-here.html")
    assert r.status_code == 404
    assert not r.is_json


# ----------------------------------------------------------------- static

@pytest.mark.parametrize("path", ["/", "/dashboard.html", "/assets/js/utils.js"])
def test_static_routes_serve(client, path):
    assert client.get(path).status_code == 200


# ------------------------------------------------------- verdict provenance

def test_a_verdict_says_which_engine_produced_it(client, engine_ready, fake_face):
    """`/api/health` also reports the engine, but that is a different
    request at a different moment. A result read back from history has no
    status page to consult, so the verdict carries its own provenance."""
    body = upload(client, fake_face, fileType="image").get_json()
    assert body["engine"] == "live"


def test_a_demo_verdict_admits_it_is_a_demo(client):
    body = client.post("/api/analyze", json={
        "fileName": "holiday.mp4", "fileType": "video", "fileSize": 1234}).get_json()
    assert body["engine"] == "simulated", \
        "a stand-in verdict presented itself as a real one"


def test_the_explanation_ranks_every_region_it_leaned_on(client, engine_ready,
                                                         fake_face):
    """One region is a poor summary when a face gives itself away in two
    places. Each weight is the largest normalised drop that region caused."""
    explain = upload(client, fake_face, fileType="image").get_json()["explain"]
    regions = explain["regions"]

    assert regions, "no regions were reported"
    assert len(regions) <= 3, "the also-rans should be dropped"
    weights = [r["weight"] for r in regions]
    assert weights == sorted(weights, reverse=True), "regions are not ranked"
    assert weights[0] == 1.0, "the top region should be the normalisation point"
    assert all(w >= 0.25 for w in weights), "a negligible region survived"
    assert regions[0]["name"] == explain["focusRegion"], \
        "the ranked list disagrees with focusRegion"
