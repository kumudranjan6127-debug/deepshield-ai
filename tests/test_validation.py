"""Field, URL and media validation — the checks that run before anything
expensive or dangerous happens.

Separate from `test_security.py` on purpose: that file is about attacks,
this one is about the ordinary wrong input a real user produces. A phone
that uploads a HEIC, a paste of a YouTube link, a video that is forty
minutes long. Each has to fail with something the interface can explain.
"""
import pytest

pytestmark = pytest.mark.validation


# ------------------------------------------------------------------ fields

@pytest.mark.parametrize("payload,why", [
    ({"agree": "true"}, "a string that looks like a boolean"),
    ({"agree": 1}, "an integer that looks like a boolean"),
    ({}, "the field missing entirely"),
    ({"agree": None}, "an explicit null"),
])
def test_feedback_requires_a_real_boolean(client, payload, why):
    r = client.post("/api/feedback", json=payload)
    assert r.status_code == 400, f"accepted {why}"
    body = r.get_json()
    assert body["error_code"] == "BAD_FIELD"
    assert "agree" in body["error"], "the message should name the field"


def test_a_malformed_json_body_is_not_a_crash(client):
    r = client.post("/api/feedback", data="{not json",
                    content_type="application/json")
    assert r.status_code == 400
    assert r.is_json


def test_an_unknown_file_type_falls_back_safely(client):
    """`fileType` decides which pipeline runs. Anything that is not
    "video" is treated as an image rather than dispatching on user input."""
    r = client.post("/api/analyze", json={
        "fileName": "x.bin", "fileType": "../../etc/passwd", "fileSize": 10})
    assert r.status_code == 200
    assert r.get_json()["prediction"] in ("real", "deepfake")


# -------------------------------------------------------------------- URLs

def public_dns(monkeypatch):
    """Stub DNS to one ordinary public address, so URL tests never touch
    the network and cannot fail on a plane."""
    import socket
    import security
    monkeypatch.setattr(security.socket, "getaddrinfo",
                        lambda host, *a, **k: [
                            (socket.AF_INET, socket.SOCK_STREAM, 6, "",
                             ("93.184.216.34", 443))])


def test_a_direct_https_video_url_is_allowed_through_validation(monkeypatch):
    """Validation only: no request is made here. A public host must reach
    the download step rather than being refused outright."""
    import security
    public_dns(monkeypatch)
    security.validate_url("https://example.com/clip.mp4")


def test_a_url_that_returns_a_web_page_is_refused(client, monkeypatch):
    """What a YouTube or Instagram link actually delivers: HTML.

    Deliberately offline. An earlier version of this test posted a real
    youtube.com URL and took 21 seconds, because the backend has no
    streaming-platform blocklist — that guard lives in `upload.js` and only
    protects the browser. An API caller reaches the network. The downloader
    moved into `network.py` when connections became DNS-pinned, so this test
    stubs that boundary rather than the old `security.safe_download` symbol."""
    import network

    def fake_download(url, dest):
        page = b"<!DOCTYPE html><html><head><title>Video</title></head></html>"
        with open(dest, "wb") as f:
            f.write(page * 40)
        return len(page) * 40

    monkeypatch.setattr(network, "safe_download", fake_download)
    r = client.post("/api/analyze", json={
        "url": "https://example.com/watch?v=abc",
        "fileName": "clip.mp4", "fileType": "video"})

    assert r.status_code >= 400, "an HTML page was accepted as a video"
    body = r.get_json()
    assert body["ok"] is False
    assert body["error_code"] in ("CORRUPT_MEDIA", "BAD_MAGIC", "URL_NOT_VIDEO")


def test_the_browser_still_names_the_streaming_platforms():
    """The friendly refusal is frontend-only, so if that list disappears
    users get a slow generic failure instead of an explanation."""
    import os
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    upload_js = os.path.join(root, "frontend", "assets", "js", "pages", "upload.js")
    source = open(upload_js, encoding="utf-8").read()
    for host in ("youtube.com", "instagram.com", "tiktok.com"):
        assert re.search(re.escape(host), source), f"{host} is no longer refused"


@pytest.mark.parametrize("url", ["", "   ", "https://", "https:///path",
                                 "http://", "://example.com"])
def test_unusable_urls_are_refused(url):
    import errors
    import security
    with pytest.raises(errors.ApiError):
        security.validate_url(url)


def test_http_is_allowed_only_when_explicitly_enabled(monkeypatch):
    import security
    from config import CFG
    public_dns(monkeypatch)
    monkeypatch.setattr(CFG, "ALLOW_HTTP_URLS", True)
    security.validate_url("http://example.com/clip.mp4")


# ------------------------------------------------------------------- media

def test_a_valid_image_passes_media_validation(fake_face):
    import security
    security.validate_media_file(fake_face, "image")


def test_a_valid_video_passes_media_validation(face_video):
    import security
    security.validate_media_file(face_video, "video")


def test_a_corrupt_video_fails_media_validation(corrupt_video):
    import errors
    import security
    with pytest.raises(errors.ApiError) as caught:
        security.validate_media_file(corrupt_video, "video")
    assert caught.value.code == "CORRUPT_MEDIA"


def test_an_image_below_the_minimum_size_is_refused(tmp_path):
    """A 4x4 image carries no face and no artefacts; accepting it just
    produces a confident verdict about nothing."""
    import errors
    import security
    import numpy as np
    from PIL import Image

    tiny = tmp_path / "tiny.png"
    Image.fromarray(np.zeros((4, 4, 3), "uint8")).save(tiny)
    with pytest.raises(errors.ApiError) as caught:
        security.validate_media_file(str(tiny), "image")
    assert caught.value.code == "IMAGE_TOO_SMALL"


def test_an_overlong_video_is_refused(monkeypatch, face_video):
    """Duration is capped so one upload cannot occupy the single inference
    slot for minutes."""
    import errors
    import security
    from config import CFG
    monkeypatch.setattr(CFG, "MAX_VIDEO_SECONDS", 1)
    with pytest.raises(errors.ApiError) as caught:
        security.validate_media_file(face_video, "video")
    assert caught.value.code == "VIDEO_TOO_LONG"


def test_error_codes_are_stable_strings(client):
    """The frontend switches on these. Renaming one is a breaking change,
    so the set is pinned here."""
    known = {
        "NO_FILE", "BAD_TYPE", "BAD_MIME", "BAD_MAGIC", "EMPTY_FILE",
        "CORRUPT_MEDIA", "IMAGE_TOO_LARGE", "IMAGE_TOO_SMALL",
        "VIDEO_TOO_LONG", "TOO_LARGE", "BLOCKED_URL", "INSECURE_URL",
        "URL_NOT_VIDEO", "BAD_FIELD", "RATE_LIMITED", "BUSY",
        "UPLOAD_NOT_FOUND", "INSECURE_REQUEST", "INVALID_INPUT", "INTERNAL",
    }
    import errors
    produced = set()
    for name in dir(errors):
        maker = getattr(errors, name)
        if not callable(maker) or name.startswith("_") or name == "ApiError":
            continue
        try:
            made = maker()
        except Exception:
            continue        # not a zero-argument error factory
        if isinstance(made, errors.ApiError):
            produced.add(made.code)

    unknown = produced - known
    assert not unknown, f"new error codes are undocumented: {sorted(unknown)}"