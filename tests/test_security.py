"""Attacks that must be refused.

Two of these — the path-traversal pair — were never tested before, though
both defences were already in place. That is the dangerous shape: code that
is correct today, protected by nothing that would notice if it stopped being
correct. `os.path.basename` is one deletion away from a file-read primitive.

SSRF gets the most attention because the URL feature fetches whatever it is
pointed at, from inside the network the server lives in. Cloud metadata at
169.254.169.254 is the classic target and hands out credentials to anyone
who asks.
"""
import io
import os
import time

import pytest

pytestmark = pytest.mark.security


# ------------------------------------------------------------------- SSRF

BLOCKED_URLS = [
    ("https://127.0.0.1/video.mp4", "loopback"),
    ("https://localhost/video.mp4", "localhost by name"),
    ("https://0.0.0.0/video.mp4", "the unspecified address"),
    ("https://169.254.169.254/latest/meta-data/", "cloud metadata"),
    ("https://10.0.0.5/video.mp4", "private 10/8"),
    ("https://172.16.4.4/video.mp4", "private 172.16/12"),
    ("https://192.168.1.1/video.mp4", "private 192.168/16"),
    ("https://[::1]/video.mp4", "IPv6 loopback"),
    ("https://[fe80::1]/video.mp4", "IPv6 link-local"),
    ("https://[fc00::1]/video.mp4", "IPv6 unique-local"),
    ("https://[::ffff:127.0.0.1]/video.mp4", "IPv4-mapped loopback"),
]


@pytest.mark.parametrize("url,what", BLOCKED_URLS, ids=[w for _, w in BLOCKED_URLS])
def test_internal_destinations_are_refused(url, what):
    """Checked at the validator, so no request ever leaves the process."""
    import errors
    import security
    with pytest.raises(errors.ApiError) as caught:
        security.validate_url(url)
    assert caught.value.code == "BLOCKED_URL", \
        f"{what} was refused, but for the wrong reason ({caught.value.code})"


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/x.mp4",
    "gopher://example.com/x", "data:video/mp4;base64,AAAA",
    "javascript:alert(1)", "not-a-url", "",
])
def test_non_https_schemes_are_refused(url):
    import errors
    import security
    with pytest.raises(errors.ApiError):
        security.validate_url(url)


def test_plain_http_is_refused_by_default(monkeypatch):
    import errors
    import security
    from config import CFG
    monkeypatch.setattr(CFG, "ALLOW_HTTP_URLS", False)
    with pytest.raises(errors.ApiError):
        security.validate_url("http://example.com/video.mp4")


@pytest.mark.parametrize("resolved,what", [
    ("127.0.0.1", "loopback"),
    ("169.254.169.254", "cloud metadata"),
    ("10.1.2.3", "a private address"),
    ("192.168.0.7", "a home router"),
])
def test_a_public_name_that_resolves_inward_is_refused(monkeypatch, resolved, what):
    """The case a hostname blocklist cannot catch: `evil.example` is a
    perfectly ordinary name whose DNS answer points inside the network.
    Only resolving first and judging the address catches it.

    DNS is stubbed, so this runs offline and cannot be defeated by
    whoever currently owns a convenient wildcard domain."""
    import socket
    import errors
    import security

    monkeypatch.setattr(security.socket, "getaddrinfo",
                        lambda host, *a, **k: [
                            (socket.AF_INET, socket.SOCK_STREAM, 6, "",
                             (resolved, 443))])

    with pytest.raises(errors.ApiError) as caught:
        security.validate_url("https://looks-completely-fine.example/video.mp4")
    assert caught.value.code == "BLOCKED_URL", f"{what} was allowed through"


def test_one_bad_answer_poisons_the_whole_name(monkeypatch):
    """A host answering with one public and one internal address must be
    refused outright — taking the first answer is a race, not a check."""
    import socket
    import errors
    import security

    monkeypatch.setattr(security.socket, "getaddrinfo",
                        lambda host, *a, **k: [
                            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
                            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
                        ])
    with pytest.raises(errors.ApiError):
        security.validate_url("https://mixed-answers.example/video.mp4")


def test_a_name_that_resolves_publicly_is_allowed(monkeypatch):
    import socket
    import security
    monkeypatch.setattr(security.socket, "getaddrinfo",
                        lambda host, *a, **k: [
                            (socket.AF_INET, socket.SOCK_STREAM, 6, "",
                             ("93.184.216.34", 443))])
    security.validate_url("https://example.com/clip.mp4")


def test_a_blocked_url_is_refused_through_the_api(client):
    r = client.post("/api/analyze", json={
        "url": "https://169.254.169.254/latest/meta-data/",
        "fileName": "x.mp4", "fileType": "video"})
    assert r.status_code >= 400
    assert r.get_json()["error_code"] in ("BLOCKED_URL", "INSECURE_URL")


# --------------------------------------------------------- path traversal

@pytest.mark.parametrize("upload_id", [
    "../../backend/app.py",
    "..\\..\\backend\\app.py",
    "....//....//backend/app.py",
    "/etc/passwd",
    "C:\\Windows\\win.ini",
    "%2e%2e%2fbackend%2fapp.py",
])
def test_a_crafted_upload_id_cannot_escape_the_upload_folder(client, upload_id):
    """`uploadId` is attacker-controlled and becomes a filesystem path."""
    import app as app_module
    assert app_module.staged_upload_path(upload_id) is None

    r = client.post("/api/analyze", json={
        "uploadId": upload_id, "fileName": "x.jpg", "fileType": "image"})
    assert r.status_code >= 400, "a traversal id produced a 200"
    assert r.get_json()["ok"] is False


def test_a_crafted_upload_id_cannot_reach_a_real_file(client, tmp_path):
    """Aimed at a file that genuinely exists, so a passing result cannot be
    explained away by the target simply being absent."""
    import app as app_module
    from config import CFG

    sentinel = os.path.join(os.path.dirname(CFG.UPLOAD_DIR), "sentinel_secret.txt")
    with open(sentinel, "w", encoding="utf-8") as f:
        f.write("do not serve this")
    try:
        assert os.path.exists(sentinel)
        assert app_module.staged_upload_path("../sentinel_secret.txt") is None
        assert app_module.staged_upload_path("..\\sentinel_secret.txt") is None
    finally:
        os.remove(sentinel)


@pytest.mark.parametrize("path", [
    "../backend/app.py",
    "../../backend/config.py",
    "..%2f..%2fbackend%2fapp.py",
    "....//....//backend/app.py",
    "../models/deepshield.onnx.json",
    "../requirements.txt",
])
def test_static_routes_cannot_escape_the_frontend(client, path):
    """frontend/ is public; backend code, models and uploads are not."""
    r = client.get("/" + path)
    assert r.status_code in (301, 302, 400, 403, 404), \
        f"{path} returned {r.status_code}"
    if r.status_code == 200:
        pytest.fail(f"{path} was served")


def test_the_upload_folder_is_not_web_reachable(client, fake_face):
    with open(fake_face, "rb") as f:
        upload_id = client.post("/api/upload", data={
            "file": (io.BytesIO(f.read()), "face.jpeg")},
            content_type="multipart/form-data").get_json()["uploadId"]
    assert client.get(f"/uploads/{upload_id}").status_code != 200
    assert client.get(f"/../uploads/{upload_id}").status_code != 200


# ------------------------------------------------------------ rate limit

def test_a_burst_is_throttled(client):
    from config import CFG
    codes = [client.get("/api/health").status_code for _ in range(CFG.RATE_LIMIT + 3)]
    # health is not rate limited; analyze is
    payload = {"fileName": "x.mp4", "fileType": "video", "fileSize": 10}
    codes = [client.post("/api/analyze", json=payload).status_code
             for _ in range(CFG.RATE_LIMIT + 3)]
    assert codes[:CFG.RATE_LIMIT] == [200] * CFG.RATE_LIMIT, \
        f"the first {CFG.RATE_LIMIT} should be allowed, got {codes}"
    assert 429 in codes[CFG.RATE_LIMIT:], f"nothing was throttled: {codes}"


def test_the_limit_is_per_client():
    """One noisy caller must not lock everyone else out."""
    import errors
    import security
    limiter = security.RateLimiter(2, 60)
    limiter.check("1.2.3.4")
    limiter.check("1.2.3.4")
    with pytest.raises(errors.ApiError):
        limiter.check("1.2.3.4")
    limiter.check("5.6.7.8")            # a different client is unaffected


def test_a_throttled_response_says_when_to_retry(client):
    from config import CFG
    payload = {"fileName": "x.mp4", "fileType": "video", "fileSize": 10}
    last = None
    for _ in range(CFG.RATE_LIMIT + 3):
        last = client.post("/api/analyze", json=payload)
    assert last.status_code == 429
    assert last.get_json()["error_code"] == "RATE_LIMITED"


# ----------------------------------------------------------- concurrency

def test_only_so_many_analyses_run_at_once():
    import errors
    import security
    gate = security.InferenceGate(1, 0)
    with gate:
        with pytest.raises(errors.ApiError) as caught:
            with gate:
                pass
    assert caught.value.code == "BUSY"


def test_the_slot_is_released_afterwards():
    import security
    gate = security.InferenceGate(1, 0)
    with gate:
        pass
    with gate:                            # must not still be held
        pass


def test_a_failed_analysis_still_releases_its_slot():
    """An exception inside the gate must not leak the semaphore, or the
    server stops accepting work after the first error."""
    import security
    gate = security.InferenceGate(1, 0)
    with pytest.raises(RuntimeError):
        with gate:
            raise RuntimeError("boom")
    with gate:
        pass


# -------------------------------------------------------------- cleanup

def test_stale_uploads_are_swept():
    import security
    from config import CFG

    os.makedirs(CFG.UPLOAD_DIR, exist_ok=True)
    stale = os.path.join(CFG.UPLOAD_DIR, "stale_test_file.jpg")
    fresh = os.path.join(CFG.UPLOAD_DIR, "fresh_test_file.jpg")
    for path in (stale, fresh):
        with open(path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0")

    old = time.time() - (CFG.UPLOAD_TTL_SECONDS + 600)
    os.utime(stale, (old, old))
    try:
        security.cleanup_uploads()
        assert not os.path.exists(stale), "an expired upload was left behind"
        assert os.path.exists(fresh), "a fresh upload was swept too early"
    finally:
        for path in (stale, fresh):
            if os.path.exists(path):
                os.remove(path)


# ------------------------------------------------------------ magic bytes

@pytest.mark.parametrize("blob,ext,expected", [
    (b"\x89PNG\r\n\x1a\n", ".png", True),
    (b"\xff\xd8\xff\xe0", ".jpg", True),
    (b"RIFF\x00\x00\x00\x00WEBPVP8 ", ".webp", True),
    (b"\x00\x00\x00\x20ftypisom", ".mp4", True),
    (b"MZ\x90\x00", ".png", False),
    (b"PK\x03\x04", ".png", False),
    (b"<html>", ".jpg", False),
    (b"", ".png", False),
    (b"\x89PN", ".png", False),
    (b"RIFF\x00\x00\x00\x00AVI ", ".webp", False),    # RIFF, but not WEBP
    (b"\x00\x00\x00\x20junkisom", ".mp4", False),     # no ftyp box
    (b"\x89PNG\r\n\x1a\n", ".exe", False),            # extension not allowed at all
])
def test_magic_bytes_decide_the_type(blob, ext, expected):
    import security
    assert security.check_magic(blob, ext) is expected
