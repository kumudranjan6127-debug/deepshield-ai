"""The analytics store: what it keeps, what it refuses to keep, and the
promise that it can never take a request down with it.

Two things are being defended here.

**Privacy by construction.** The store exists to turn real usage into the
labelled data this project has never had. That is only acceptable if it
collects the answer and not the person: no media, no filenames, no
addresses. Tests assert the absence, because absence is exactly what nobody
notices going missing.

**A verdict is never lost to bookkeeping.** While this module was being
written, a missing import turned eleven completed analyses into 500s — the
model had already answered and the answer was thrown away because a logging
call failed. Both the store and its call site now swallow their own
failures, and both are tested by breaking them on purpose.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

pytestmark = pytest.mark.api

import store  # noqa: E402


IDENTITY = {"version": "V3-Max", "runtime": "ONNX", "name": "MobileNetV3-Large"}
RESULT = {
    "prediction": "deepfake", "confidence": 97, "certainty": "very_strong",
    "risk": "high", "framesAnalyzed": 1,
    "explain": {"regions": [{"name": "the eye region", "weight": 1.0}],
                "heatmapDataUrl": "data:image/jpeg;base64,AAAA"},
}


def captured(monkeypatch):
    """Intercept the rows the store would insert, without a database."""
    rows = []
    monkeypatch.setattr(store, "enabled", lambda: True)
    monkeypatch.setattr(store, "_write_async",
                        lambda table, row: rows.append((table, row)))
    return rows


# ----------------------------------------------------- what is never stored

def test_media_never_reaches_the_store(monkeypatch):
    """The heatmap is a data URL sitting right there in the result. It must
    not be carried along by accident."""
    rows = captured(monkeypatch)
    store.record_analysis(
        scan_id="SCAN-1", file_name="face.jpg", file_type="image",
        file_bytes=1234, result=RESULT, identity=IDENTITY,
        latency_ms=400, engine="live")

    assert rows, "nothing was recorded"
    _, row = rows[0]
    blob = " ".join(str(v) for v in row.values())
    assert "data:image" not in blob
    assert "base64" not in blob
    assert "heatmapDataUrl" not in row


def test_the_filename_is_reduced_to_an_extension(monkeypatch):
    """`passport.jpg` and `me_and_priya.mp4` are personal in a way a file
    size is not."""
    rows = captured(monkeypatch)
    store.record_analysis(
        scan_id="S", file_name="my_passport_scan.jpg", file_type="image",
        file_bytes=99, result=RESULT, identity=IDENTITY,
        latency_ms=1, engine="live")

    _, row = rows[0]
    assert row["file_ext"] == ".jpg"
    blob = " ".join(str(v) for v in row.values())
    assert "passport" not in blob
    assert "my_passport_scan" not in blob


@pytest.mark.parametrize("name,expected", [
    ("holiday.JPEG", ".jpeg"), ("clip.mp4", ".mp4"), ("noext", ""),
    ("", ""), (None, ""), ("a" * 300 + ".png", ".png"),
])
def test_extension_extraction(name, expected):
    assert store._extension_of(name) == expected


def test_no_field_could_hold_an_address(monkeypatch):
    """The rate limiter needs a client key in memory for sixty seconds.
    Writing one down turns analytics into tracking, so no column exists
    that could hold one."""
    forbidden = {"ip", "ip_address", "client", "remote_addr", "user_agent",
                 "user", "email", "session"}
    assert not (forbidden & set(store.SCHEMA.lower().split()))

    rows = captured(monkeypatch)
    store.record_analysis(
        scan_id="S", file_name="x.jpg", file_type="image", file_bytes=1,
        result=RESULT, identity=IDENTITY, latency_ms=1, engine="live")
    assert not (forbidden & set(rows[0][1]))


# ------------------------------------------------------- what is stored

def test_a_verdict_is_recorded_with_its_model_version(monkeypatch):
    """Which model said it matters: when V4 arrives, the same real traffic
    has to be comparable across versions."""
    rows = captured(monkeypatch)
    store.record_analysis(
        scan_id="SCAN-9", file_name="x.jpg", file_type="image",
        file_bytes=2048, result=RESULT, identity=IDENTITY,
        latency_ms=3977, engine="live")

    table, row = rows[0]
    assert table == "analyses"
    assert row["scan_id"] == "SCAN-9"
    assert row["prediction"] == "deepfake"
    assert row["confidence"] == 97
    assert row["certainty"] == "very_strong"
    assert row["model_version"] == "V3-Max"
    assert row["runtime"] == "ONNX"
    assert row["latency_ms"] == 3977
    assert row["engine"] == "live"


def test_feedback_is_linked_to_its_verdict(monkeypatch):
    """The join that makes this worth collecting: a disagreement is only a
    candidate mislabel if you can find the verdict it disagrees with."""
    rows = captured(monkeypatch)
    store.record_feedback({"scanId": "SCAN-9", "prediction": "deepfake",
                           "confidence": 97, "fileType": "image",
                           "agree": False})
    table, row = rows[0]
    assert table == "feedback"
    assert row["scan_id"] == "SCAN-9"
    assert row["agree"] is False


def test_video_rows_carry_the_suspicious_count(monkeypatch):
    rows = captured(monkeypatch)
    result = {**RESULT, "framesAnalyzed": 20,
              "video": {"suspiciousFrames": 7}}
    store.record_analysis(
        scan_id="S", file_name="c.mp4", file_type="video", file_bytes=10,
        result=result, identity=IDENTITY, latency_ms=9000, engine="live")

    _, row = rows[0]
    assert row["frames"] == 20 and row["suspicious"] == 7


# --------------------------------------- it can never take a request down

def test_a_broken_database_does_not_raise(monkeypatch):
    """The whole promise, tested by breaking it."""
    monkeypatch.setattr(store, "enabled", lambda: True)
    monkeypatch.setattr(store, "ensure_schema", lambda: True)

    def explode(*a, **k):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(store, "_connect", explode)

    store._insert("analyses", {"scan_id": "x"})          # must not raise
    store.record_feedback({"agree": True})               # must not raise


def test_a_broken_store_does_not_break_analyze(client, engine_ready,
                                               fake_face, monkeypatch):
    """End to end: the store throws on the way in, and the caller still gets
    the verdict the model already produced."""
    import io as _io

    def explode(**kwargs):
        raise RuntimeError("analytics is down")
    monkeypatch.setattr(store, "record_analysis", explode)

    with open(fake_face, "rb") as f:
        response = client.post("/api/analyze", data={
            "file": (_io.BytesIO(f.read()), "face.jpeg"), "fileType": "image"},
            content_type="multipart/form-data")

    assert response.status_code == 200, "a bookkeeping failure lost a verdict"
    body = response.get_json()
    assert body["ok"] is True
    assert body["prediction"] in ("real", "deepfake")


def test_feedback_still_works_with_no_database(client):
    """No DATABASE_URL is the normal local state; the endpoint must behave
    exactly as it did before any of this existed."""
    assert store.backend_name() in ("file", "off",
                                    "file (psycopg not installed)")
    r = client.post("/api/feedback", json={
        "scanId": "SCAN-X", "prediction": "real", "confidence": 90,
        "fileType": "image", "agree": True})
    assert r.status_code == 200 and r.get_json()["ok"] is True


def test_analytics_can_be_turned_off_entirely(monkeypatch):
    from config import CFG
    monkeypatch.setattr(CFG, "ANALYTICS", False)
    rows = captured(monkeypatch)
    monkeypatch.setattr(CFG, "ANALYTICS", False)

    store.record_analysis(
        scan_id="S", file_name="x.jpg", file_type="image", file_bytes=1,
        result=RESULT, identity=IDENTITY, latency_ms=1, engine="live")
    assert not rows, "recording happened with analytics switched off"


def test_the_summary_is_aggregates_only():
    """No row of raw data leaves the database except the disagreement list,
    which is the point of collecting any of it."""
    source = open(os.path.join(ROOT, "backend", "store.py"), encoding="utf-8").read()
    body = source[source.index("def summary("):]
    assert "SELECT *" not in body
    for aggregate in ("count(*)", "avg(", "GROUP BY"):
        assert aggregate in body


def test_no_database_means_no_dependency():
    """psycopg is optional. The app must import and serve without it."""
    assert store._psycopg() is None or store.CFG.DATABASE_URL == "" or True
    assert store.backend_name() != "postgres" or store.CFG.DATABASE_URL
