"""Security tests — every one is an attempted attack, not a happy path.

    python scripts/security_test.py            # unit + live tests
    python scripts/security_test.py --unit     # no server needed

Covers: SSRF (v4, v6, DNS, redirects), oversized files, forged
extensions, corrupt and empty media, malformed URLs, private addresses,
and rate limiting. A test passes only when the attack is *refused*.
"""
import io
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
API = os.environ.get("DS_TEST_URL", "http://127.0.0.1:5000")

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
    return condition


# ---------------------------------------------------------------- helpers

def post_json(path, payload):
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def post_file(path, filename, blob, fields=None):
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in (fields or {}).items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f"name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
        + blob + b"\r\n")
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        API + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except Exception:
            return e.code, {}
    except Exception as e:
        return 0, {"error": str(e)}


def tiny_png(width=64, height=64) -> bytes:
    """A real, valid PNG — used where a test needs to pass validation."""
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x7f\x7f\x7f" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def huge_png_header() -> bytes:
    """A decompression bomb: the header claims 30000×30000, ~900M pixels."""
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))
    ihdr = struct.pack(">IIBBBBB", 30000, 30000, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


# ---------------------------------------------------------------- unit tests
# These exercise the guards directly, so they run without a server.

def unit_tests():
    import security
    import errors
    from config import CFG

    print("\nSSRF — addresses that must never be fetched")
    # Deliberately https, so the scheme check cannot be what rejects
    # these: each one has to be stopped by the address check itself.
    blocked = [
        ("https://127.0.0.1/x.mp4", "loopback v4"),
        ("https://localhost/x.mp4", "localhost name"),
        ("https://0.0.0.0/x.mp4", "unspecified"),
        ("https://10.0.0.5/x.mp4", "private 10/8"),
        ("https://172.16.4.4/x.mp4", "private 172.16/12"),
        ("https://192.168.1.1/x.mp4", "private 192.168/16"),
        ("https://169.254.169.254/latest/meta-data/", "link-local metadata"),
        ("https://[::1]/x.mp4", "loopback v6"),
        ("https://[fe80::1]/x.mp4", "link-local v6"),
        ("https://[fc00::1]/x.mp4", "unique-local v6"),
        ("https://[::ffff:127.0.0.1]/x.mp4", "v4-mapped loopback"),
        ("https://[::]/x.mp4", "unspecified v6"),
        ("https://224.0.0.1/x.mp4", "multicast"),
        ("https://127.0.0.1.nip.io/x.mp4", "hostname resolving to loopback"),
    ]
    for url, label in blocked:
        try:
            security.validate_url(url)
            check(f"blocks {label}", False, "ACCEPTED — this is a hole")
        except errors.ApiError as e:
            # BLOCKED_URL means the address check did the work
            check(f"blocks {label}", e.code == "BLOCKED_URL", e.code)

    print("\nMalformed and hostile URLs")
    for url, label in [
        ("", "empty"),
        ("not-a-url", "no scheme"),
        ("file:///etc/passwd", "file scheme"),
        ("ftp://example.com/x.mp4", "ftp scheme"),
        ("gopher://example.com/x", "gopher scheme"),
        ("javascript:alert(1)", "javascript scheme"),
        ("http://" + "a" * 3000 + ".com/x.mp4", "over-long"),
    ]:
        try:
            security.validate_url(url)
            check(f"rejects {label} URL", False, "ACCEPTED")
        except errors.ApiError as e:
            check(f"rejects {label} URL", True, e.code)

    print("\nHTTPS policy")
    was = CFG.ALLOW_HTTP_URLS
    CFG.ALLOW_HTTP_URLS = False
    try:
        security.validate_url("http://example.com/x.mp4")
        check("plain http refused by default", False, "ACCEPTED")
    except errors.ApiError as e:
        check("plain http refused by default", e.code == "INSECURE_URL", e.code)
    finally:
        CFG.ALLOW_HTTP_URLS = was

    print("\nMagic bytes")
    check("png header recognised", security.check_magic(tiny_png()[:32], ".png"))
    check("exe renamed .png rejected", not security.check_magic(b"MZ\x90\x00" + b"\x00" * 28, ".png"))
    check("html renamed .jpg rejected", not security.check_magic(b"<!DOCTYPE html><html>", ".jpg"))
    check("zip renamed .mp4 rejected", not security.check_magic(b"PK\x03\x04" + b"\x00" * 28, ".mp4"))
    check("empty head rejected", not security.check_magic(b"", ".png"))

    print("\nRate limiter")
    rl = security.RateLimiter(limit=3, window_seconds=60)
    ok = True
    for _ in range(3):
        try:
            rl.check("1.2.3.4")
        except errors.ApiError:
            ok = False
    check("first 3 allowed", ok)
    try:
        rl.check("1.2.3.4")
        check("4th blocked", False, "ACCEPTED")
    except errors.ApiError as e:
        check("4th blocked", e.status == 429, f"{e.code} {e.status}")
    try:
        rl.check("5.6.7.8")
        check("other client unaffected", True)
    except errors.ApiError:
        check("other client unaffected", False)

    print("\nConcurrency gate")
    gate = security.InferenceGate(workers=1, wait_seconds=1)
    with gate:
        try:
            with gate:
                check("second concurrent analysis refused", False, "ADMITTED")
        except errors.ApiError as e:
            check("second concurrent analysis refused", e.status == 503, f"{e.code} {e.status}")
    with gate:                      # slot returned after release
        check("slot released afterwards", True)

    print("\nAbandoned upload cleanup")
    stale = os.path.join(CFG.UPLOAD_DIR, f"stale_{uuid.uuid4().hex}.jpg")
    fresh = os.path.join(CFG.UPLOAD_DIR, f"fresh_{uuid.uuid4().hex}.jpg")
    os.makedirs(CFG.UPLOAD_DIR, exist_ok=True)
    for p in (stale, fresh):
        with open(p, "wb") as f:
            f.write(b"x")
    old = time.time() - 3600
    os.utime(stale, (old, old))
    removed = security.cleanup_uploads(max_age_seconds=1800)
    check("stale upload deleted", not os.path.exists(stale), f"{removed} removed")
    check("fresh upload kept", os.path.exists(fresh))
    if os.path.exists(fresh):
        os.remove(fresh)


# ---------------------------------------------------------------- live tests

def live_tests():
    from config import CFG

    print("\nUploads over HTTP")
    status, body = post_file("/api/upload", "clean.png", tiny_png())
    check("a real png is accepted", status == 200 and body.get("uploadId"),
          f"{status} {body.get('error_code', '')}")

    status, body = post_file("/api/upload", "payload.png", b"MZ\x90\x00" + b"\x00" * 400)
    check("fake extension refused", status == 400 and body.get("error_code") == "BAD_MAGIC",
          f"{status} {body.get('error_code')}")

    status, body = post_file("/api/upload", "empty.png", b"")
    check("empty file refused", status == 400, f"{status} {body.get('error_code')}")

    truncated = tiny_png()[:40]                       # valid header, no image data
    status, body = post_file("/api/upload", "broken.png", truncated)
    check("corrupt image refused", status == 400,
          f"{status} {body.get('error_code')}")

    status, body = post_file("/api/upload", "bomb.png", huge_png_header())
    check("decompression bomb refused", status == 400,
          f"{status} {body.get('error_code')}")

    status, body = post_file("/api/upload", "clip.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 200)
    check("corrupt video refused", status == 400,
          f"{status} {body.get('error_code')}")

    status, body = post_file("/api/upload", "script.sh", b"#!/bin/sh\nrm -rf /\n")
    check("disallowed extension refused", status == 400 and body.get("error_code") == "BAD_TYPE",
          f"{status} {body.get('error_code')}")

    oversized = b"\x89PNG\r\n\x1a\n" + b"\x00" * (CFG.MAX_UPLOAD_BYTES + 1024)
    status, body = post_file("/api/upload", "huge.png", oversized)
    check("oversized upload refused", status in (413, 400, 0),
          f"{status} {body.get('error_code', body.get('error', ''))[:40]}")

    print("\nSSRF over HTTP")
    # https, so the address guard is what has to stop these end to end
    for url, label in [
        ("https://127.0.0.1:5000/api/health", "loopback"),
        ("https://169.254.169.254/latest/meta-data/", "cloud metadata"),
        ("https://192.168.1.1/video.mp4", "private LAN"),
        ("https://[::1]/video.mp4", "IPv6 loopback"),
        ("https://127.0.0.1.nip.io/video.mp4", "hostname → loopback"),
        ("file:///etc/passwd", "file scheme"),
        ("ftp://example.com/v.mp4", "ftp scheme"),
    ]:
        status, body = post_json("/api/analyze", {"url": url, "fileType": "video"})
        refused = status == 400 and body.get("error_code") in (
            "BLOCKED_URL", "INSECURE_URL", "INVALID_INPUT")
        check(f"SSRF via {label} refused", refused,
              f"{status} {body.get('error_code')}")

    print("\nRate limiting")
    # Runs last on purpose: it exhausts the window, so any check after it
    # would be measuring the limiter rather than itself.
    codes = []
    for _ in range(CFG.RATE_LIMIT + 4):
        status, _ = post_json("/api/analyze", {"fileName": "x.jpg", "fileType": "image"})
        codes.append(status)
    check("burst is throttled", 429 in codes,
          f"statuses: {codes}")


# ---------------------------------------------------------------- main

def server_up():
    try:
        urllib.request.urlopen(API + "/api/health", timeout=3).read()
        return True
    except Exception:
        return False


def main():
    unit_only = "--unit" in sys.argv
    print("DeepShield security tests")
    unit_tests()

    if unit_only:
        pass
    elif server_up():
        from config import CFG
        if CFG.RATE_LIMIT < 40:
            print(f"\nnote: the live suite sends ~25 requests but the limit is "
                  f"{CFG.RATE_LIMIT}/{CFG.RATE_WINDOW_SECONDS}s.\n"
                  f"      start both the server and this script with "
                  f"DS_RATE_LIMIT=50 to exercise them properly.")
        live_tests()
    else:
        print(f"\nnote: {API} not answering — HTTP tests skipped")

    print(f"\n{'=' * 52}")
    print(f"passed {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print("  - " + f)
        return 1
    print("every attack was refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
