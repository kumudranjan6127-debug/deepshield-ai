"""Everything that decides whether a request is allowed to reach the model.

Ordering matters: each check is cheap relative to the one after it, so a
hostile request is rejected as early as possible.

    upload:  size → extension → magic bytes → decoder → dimensions/duration
    url:     scheme → host policy → DNS → address class → per-redirect re-check → download
    traffic: rate limit → concurrency gate → analysis

Nothing here imports the model; it only imports config and errors.
"""
import ipaddress
import logging
import os
import socket
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque

import errors
from config import CFG

log = logging.getLogger("deepshield")


# Pages on these services are not direct media. The browser carries the same
# names only to give an earlier friendly hint; this backend list is the
# authoritative network boundary for API callers and redirect destinations.
STREAMING_PLATFORM_HOSTS = (
    "youtube.com", "youtu.be", "youtube-nocookie.com",
    "instagram.com", "instagr.am",
    "facebook.com", "fb.watch",
    "tiktok.com",
    "twitter.com", "x.com",
    "reddit.com", "redd.it",
    "vimeo.com",
    "dailymotion.com", "dai.ly",
    "snapchat.com",
    "t.me", "telegram.me",
)


# =====================================================================
# 1. Uploads
# =====================================================================

# First bytes each format must start with. An attacker controls the
# filename, never the container header.
MAGIC = {
    ".jpg":  [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png":  [b"\x89PNG\r\n\x1a\n"],
    ".webp": [b"RIFF"],                       # + b"WEBP" at offset 8
    ".mp4":  [b"\x00\x00\x00", b"ftyp"],      # box length, then 'ftyp'
    ".mov":  [b"\x00\x00\x00", b"ftyp"],
    ".webm": [b"\x1a\x45\xdf\xa3"],           # EBML
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".webm"}

# What a browser may claim. Checked, but never trusted on its own —
# the magic bytes below are the real gate.
ALLOWED_MIME_PREFIXES = ("image/", "video/", "application/octet-stream")


def _ext_of(filename: str) -> str:
    return os.path.splitext(filename or "")[1].lower()


def check_magic(head: bytes, ext: str) -> bool:
    """Does the file actually start like the format it claims?"""
    sigs = MAGIC.get(ext)
    if not sigs:
        return False
    if ext == ".webp":
        return head.startswith(b"RIFF") and head[8:12] == b"WEBP"
    if ext in (".mp4", ".mov"):
        return b"ftyp" in head[:16]           # 'ftyp' follows the box size
    return any(head.startswith(s) for s in sigs)


def validate_upload(file_storage) -> tuple[str, str]:
    """Cheap checks on an incoming upload, before anything touches disk.

    Returns (extension, kind) where kind is 'image' or 'video'."""
    name = file_storage.filename or ""
    ext = _ext_of(name)

    if ext not in CFG.ALLOWED_UPLOAD_EXTS:
        raise errors.bad_type()

    mime = (file_storage.mimetype or "").lower()
    if mime and not mime.startswith(ALLOWED_MIME_PREFIXES):
        raise errors.bad_mime(mime)

    head = file_storage.stream.read(32)
    file_storage.stream.seek(0)
    if not head:
        raise errors.empty_file()
    if not check_magic(head, ext):
        raise errors.bad_magic(ext)

    return ext, ("image" if ext in IMAGE_EXTS else "video")


def validate_media_file(path: str, kind: str):
    """Prove the bytes on disk really decode, and are not a bomb.

    A file can pass every header check and still be a 40,000 × 40,000 PNG
    that would exhaust memory, or a truncated MP4 that only fails deep
    inside the engine."""
    if os.path.getsize(path) == 0:
        raise errors.empty_file()

    if kind == "image":
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = CFG.MAX_IMAGE_PIXELS   # PIL's own bomb guard
        try:
            with Image.open(path) as im:
                im.verify()                              # structural check
            with Image.open(path) as im:
                w, h = im.size
                im.load()                                # full decode
        except errors.ApiError:
            raise
        except Exception as e:
            raise errors.corrupt_media(f"image could not be decoded: {type(e).__name__}")

        if w * h > CFG.MAX_IMAGE_PIXELS:
            raise errors.too_many_pixels(w, h)
        if w < 16 or h < 16:
            raise errors.too_small(w, h)
        return {"width": w, "height": h}

    import cv2
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise errors.corrupt_media("video could not be opened")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        ok, _ = cap.read()
        if not ok:
            raise errors.corrupt_media("video has no readable frames")
    finally:
        cap.release()

    seconds = (frames / fps) if fps > 0 else 0
    if seconds > CFG.MAX_VIDEO_SECONDS:
        raise errors.too_long(seconds, CFG.MAX_VIDEO_SECONDS)
    return {"seconds": round(seconds, 1), "fps": round(fps, 2)}


# =====================================================================
# 2. URL fetching (SSRF)
# =====================================================================

def _is_forbidden(ip: ipaddress._BaseAddress) -> str | None:
    """Reason this address must not be fetched, or None if it is fine.

    Covers IPv4 and IPv6, including the mapped forms an attacker reaches
    for once the obvious ones are blocked (::ffff:127.0.0.1, ::1, fc00::/7,
    fe80::/10)."""
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return _is_forbidden(ip.ipv4_mapped)
        if getattr(ip, "sixtofour", None):
            return _is_forbidden(ip.sixtofour)

    if ip.is_loopback:
        return "loopback address"
    if ip.is_link_local:                 # 169.254.0.0/16 — cloud metadata
        return "link-local address"
    if ip.is_private:                    # 10/8, 172.16/12, 192.168/16, fc00::/7
        return "private address"
    if ip.is_reserved or ip.is_multicast or ip.is_unspecified:
        return "reserved address"
    return None


def _is_streaming_platform_host(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    return any(host == root or host.endswith("." + root)
               for root in STREAMING_PLATFORM_HOSTS)


def resolve_public(host: str) -> list[str]:
    """Resolve a hostname and require that *every* answer is public.

    Every address is checked, not just the first: a host that resolves to
    one public and one internal address must not be fetched at all."""
    if not host:
        raise errors.blocked_url("no host in URL")

    # A literal address needs no DNS, but the same rules apply
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
        reason = _is_forbidden(ip)
        if reason:
            raise errors.blocked_url(f"{host} is a {reason}")
        return [str(ip)]
    except ValueError:
        pass

    if host.lower() in ("localhost", "localhost.localdomain"):
        raise errors.blocked_url("localhost is not allowed")

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise errors.blocked_url(f"could not resolve {host}: {e.strerror or 'DNS failure'}")

    addresses = []
    for family, _, _, _, sockaddr in infos:
        raw = sockaddr[0]
        ip = ipaddress.ip_address(raw)
        reason = _is_forbidden(ip)
        if reason:
            raise errors.blocked_url(f"{host} resolves to a {reason} ({raw})")
        addresses.append(str(ip))

    if not addresses:
        raise errors.blocked_url(f"{host} did not resolve")
    return sorted(set(addresses))


def validate_url(url: str) -> str:
    """Scheme, media-host policy and destination checks."""
    if not url or len(url) > 2048:
        raise errors.blocked_url("URL missing or too long")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise errors.blocked_url(f"unsupported scheme: {parsed.scheme or 'none'}")
    if parsed.scheme == "http" and not CFG.ALLOW_HTTP_URLS:
        raise errors.insecure_url()

    host = parsed.hostname or ""
    if _is_streaming_platform_host(host):
        raise errors.not_a_video(
            "Streaming-platform page URLs are not direct video files. "
            "Provide a direct video URL instead."
        )

    resolved = resolve_public(host)
    log.info("url allowed: %s -> %s", parsed.hostname, ",".join(resolved))
    return url


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Redirects are followed by hand so each hop can be re-validated."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirects)


def safe_download(url: str, dest: str) -> int:
    """Fetch a video with every hop validated. Returns bytes written."""
    current = validate_url(url)

    for hop in range(CFG.MAX_REDIRECTS + 1):
        req = urllib.request.Request(current, headers={"User-Agent": "DeepShield/1.0"})
        try:
            response = _opener.open(req, timeout=CFG.URL_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                target = e.headers.get("Location")
                if not target:
                    raise errors.blocked_url("redirect without a destination")
                current = validate_url(urllib.parse.urljoin(current, target))
                log.info("redirect %d -> %s", e.code, current)
                continue
            raise errors.blocked_url(f"server returned HTTP {e.code}")
        except urllib.error.URLError as e:
            raise errors.blocked_url(f"could not fetch URL: {e.reason}")

        with response:
            ctype = response.headers.get("Content-Type", "")
            if "video" not in ctype and not current.lower().split("?")[0].endswith(".mp4"):
                raise errors.not_a_video()

            declared = response.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > CFG.MAX_URL_BYTES:
                raise errors.too_large(CFG.MAX_URL_BYTES // (1024 * 1024))

            size = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > CFG.MAX_URL_BYTES:   # the header can lie
                        raise errors.too_large(CFG.MAX_URL_BYTES // (1024 * 1024))
                    f.write(chunk)
        return size

    raise errors.blocked_url("too many redirects")


# =====================================================================
# 3. Rate limiting
# =====================================================================

class RateLimiter:
    """Sliding window per client with bounded in-process bookkeeping.

    A one-off client key must not live forever after its window expires, and
    a flood of fresh keys must not turn the limiter itself into a memory DoS.
    When the real-client map is full, previously unseen callers share one
    overflow bucket until old client windows can be swept."""

    MAX_CLIENTS = 10_000
    SWEEP_INTERVAL_SECONDS = 5.0

    def __init__(self, limit: int, window_seconds: int, max_clients: int | None = None):
        self.limit = limit
        self.window = window_seconds
        self.max_clients = max(1, int(max_clients or self.MAX_CLIENTS))
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()
        self._overflow_key = object()
        self._next_sweep = 0.0

    def _expire(self, hits: deque, now: float):
        while hits and now - hits[0] > self.window:
            hits.popleft()

    def _sweep(self, now: float):
        for key, hits in list(self._hits.items()):
            self._expire(hits, now)
            if not hits:
                del self._hits[key]
        self._next_sweep = now + min(
            self.SWEEP_INTERVAL_SECONDS, max(1.0, float(self.window)))

    def check(self, key: str):
        now = time.monotonic()
        with self._lock:
            # Sweep old clients when the table is at capacity. Until the next
            # sweep is due, new one-off callers share an overflow bucket so
            # the map cannot keep growing with attacker-controlled keys.
            if len(self._hits) >= self.max_clients and now >= self._next_sweep:
                self._sweep(now)

            bucket_key = key
            if key not in self._hits and len(self._hits) >= self.max_clients:
                bucket_key = self._overflow_key

            hits = self._hits[bucket_key]
            self._expire(hits, now)
            if len(hits) >= self.limit:
                retry = int(self.window - (now - hits[0])) + 1
                log.warning("rate limit hit by %s", key)
                raise errors.rate_limited(retry)
            hits.append(now)


# =====================================================================
# 4. Concurrency
# =====================================================================

class InferenceGate:
    """Caps how many analyses run at once.

    Inference is CPU-bound and holds ~200 MB; letting requests in freely
    turns a burst into thrashing. Extra callers wait briefly, then are
    told the server is busy rather than being queued indefinitely."""

    def __init__(self, workers: int, wait_seconds: int):
        self._sem = threading.BoundedSemaphore(max(1, workers))
        self._wait = wait_seconds
        self.workers = max(1, workers)

    def __enter__(self):
        if not self._sem.acquire(timeout=self._wait):
            log.warning("inference gate full (%d workers)", self.workers)
            raise errors.server_busy()
        return self

    def __exit__(self, *exc):
        self._sem.release()
        return False


# =====================================================================
# 5. Cleanup
# =====================================================================

def cleanup_uploads(max_age_seconds: int | None = None) -> int:
    """Delete staged files nobody came back for. Returns how many went.

    An upload is staged before the user reaches the processing page; if
    they close the tab, nothing else would ever remove it."""
    max_age = max_age_seconds if max_age_seconds is not None else CFG.UPLOAD_TTL_SECONDS
    cutoff = time.time() - max_age
    removed = 0
    try:
        names = os.listdir(CFG.UPLOAD_DIR)
    except OSError:
        return 0

    for name in names:
        path = os.path.join(CFG.UPLOAD_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
                removed += 1
        except OSError:
            pass                                # in use, or already gone
    if removed:
        log.info("cleanup removed %d abandoned upload(s)", removed)
    return removed


def start_cleanup_thread():
    """Sweep on a timer for the life of the process."""
    def loop():
        while True:
            time.sleep(CFG.CLEANUP_INTERVAL_SECONDS)
            try:
                cleanup_uploads()
            except Exception:
                log.exception("cleanup sweep failed")

    t = threading.Thread(target=loop, name="upload-cleanup", daemon=True)
    t.start()
    return t

# =====================================================================
# 6. Response hardening
# =====================================================================

def security_headers(is_secure: bool) -> dict:
    """Headers every response carries.

    The CSP is the important one. This app renders user-supplied file names
    and server error strings, and its own scripts are all local files — so
    there is no reason to allow inline script or any remote origin, and
    saying so turns a future escaping mistake into a blocked request rather
    than a stolen session.

    `data:` is allowed for images because the occlusion heatmap is delivered
    as a data URL, and `blob:` because previews are read from the file the
    user picked. Neither can execute.
    """
    headers = {
        "Content-Security-Policy": "; ".join([
            "default-src 'self'",
            "script-src 'self'",
            "style-src 'self' 'unsafe-inline'",   # inline styles set bar widths
            "img-src 'self' data: blob:",
            "media-src 'self' blob:",
            "font-src 'self'",
            "connect-src 'self'",
            "object-src 'none'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
        ]),
        # Stop a browser guessing that an upload echoed back is a script
        "X-Content-Type-Options": "nosniff",
        # Belt and braces alongside frame-ancestors, for older browsers
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        # Nothing here needs a camera, a microphone or a location
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), "
                              "payment=(), usb=()",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-origin",
    }

    # Only over TLS. Sending HSTS on a plain-HTTP local run would pin
    # localhost to https in the developer's browser and be a nuisance to undo.
    if is_secure and CFG.HSTS_SECONDS > 0:
        headers["Strict-Transport-Security"] = (
            f"max-age={CFG.HSTS_SECONDS}; includeSubDomains")
    return headers


def cors_headers(origin: str) -> dict:
    """Cross-origin headers for an allow-listed caller, or nothing.

    Never `*`. This API accepts uploads and fetches URLs on the caller's
    behalf; a wildcard would let any page on the internet drive it. An
    origin that is not on the list gets no CORS headers at all, and the
    browser refuses the response — which is the correct outcome, not an
    error to report."""
    if not origin or origin not in CFG.CORS_ORIGINS:
        return {}
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "600",
        "Vary": "Origin",
    }
