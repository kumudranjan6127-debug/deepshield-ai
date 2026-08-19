"""DeepShield application security facade.

The older, well-tested upload/header primitives remain in ``security_core``.
This layer owns the controls that need stronger process/network semantics:
streaming-page refusal, DNS-pinned remote downloads, bounded rate-limit state,
and idempotent WSGI housekeeping.
"""
from __future__ import annotations

import http.client
import os
import ssl
import threading
import time
import urllib.parse

import security_core as _core
from config import CFG

log = _core.log
errors = _core.errors

STREAMING_PAGE_HOSTS = (
    "youtube.com", "youtu.be", "instagram.com", "tiktok.com",
    "facebook.com", "fb.watch", "twitter.com", "x.com",
)

_base_validate_url = _core.validate_url


def _host_matches(host: str, domain: str) -> bool:
    host = (host or "").lower().rstrip(".")
    domain = domain.lower().rstrip(".")
    return host == domain or host.endswith("." + domain)


def _parsed(url: str):
    try:
        parsed = urllib.parse.urlparse(url or "")
        host = parsed.hostname or ""
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise errors.blocked_url(f"invalid URL: {exc}")
    if parsed.username is not None or parsed.password is not None:
        raise errors.blocked_url("credentials in media URLs are not allowed")
    return parsed, host, port


def validate_url(url: str) -> str:
    _parsed_url, host, _ = _parsed(url)
    if any(_host_matches(host, domain) for domain in STREAMING_PAGE_HOSTS):
        raise errors.blocked_url(
            "Streaming-platform page URLs are not direct media files. "
            "Upload the video or provide a direct downloadable media URL instead."
        )
    return _base_validate_url(url)


def _validated_target(url: str):
    current = validate_url(url)
    parsed, host, explicit_port = _parsed(current)
    if not host:
        raise errors.blocked_url("no host in URL")
    port = explicit_port or (443 if parsed.scheme == "https" else 80)
    if not (1 <= int(port) <= 65535):
        raise errors.blocked_url("invalid port")
    addresses = _core.resolve_public(host)
    return current, parsed, host, int(port), addresses


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host, port, pinned_ip, timeout):
        super().__init__(host, port=port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self):
        self.sock = _core.socket.create_connection(
            (self._pinned_ip, self.port), self.timeout, self.source_address)
        if self._tunnel_host:
            self._tunnel()


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host, port, pinned_ip, timeout):
        super().__init__(host, port=port, timeout=timeout,
                         context=ssl.create_default_context())
        self._pinned_ip = pinned_ip

    def connect(self):
        raw = _core.socket.create_connection(
            (self._pinned_ip, self.port), self.timeout, self.source_address)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


def _open_pinned(parsed, host, port, addresses):
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    default_port = 443 if parsed.scheme == "https" else 80
    host_header = host if port == default_port else f"{host}:{port}"
    headers = {"User-Agent": "DeepShield/1.0", "Host": host_header,
               "Accept": "video/*,application/octet-stream;q=0.8"}

    last_error = None
    for ip in addresses:
        conn = None
        try:
            cls = _PinnedHTTPSConnection if parsed.scheme == "https" else _PinnedHTTPConnection
            conn = cls(host, port, ip, CFG.URL_TIMEOUT_SECONDS)
            conn.request("GET", target, headers=headers)
            return conn, conn.getresponse(), ip
        except Exception as exc:
            last_error = exc
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    raise errors.blocked_url(
        f"could not connect to validated destination: {type(last_error).__name__ if last_error else 'connection failure'}")


def safe_download(url: str, dest: str) -> int:
    """Download a direct video without ever re-resolving the validated host."""
    current = url
    try:
        for _hop in range(CFG.MAX_REDIRECTS + 1):
            current, parsed, host, port, addresses = _validated_target(current)
            conn, response, pinned_ip = _open_pinned(parsed, host, port, addresses)
            try:
                status = int(response.status)
                if status in (301, 302, 303, 307, 308):
                    target = response.getheader("Location")
                    if not target:
                        raise errors.blocked_url("redirect without a destination")
                    current = urllib.parse.urljoin(current, target)
                    log.info("remote video redirect %d from %s via %s", status, host, pinned_ip)
                    continue
                if status < 200 or status >= 300:
                    raise errors.blocked_url(f"server returned HTTP {status}")

                ctype = response.getheader("Content-Type", "")
                if "video" not in ctype.lower() and not parsed.path.lower().endswith(".mp4"):
                    raise errors.not_a_video()
                declared = response.getheader("Content-Length")
                if declared and declared.isdigit() and int(declared) > CFG.MAX_URL_BYTES:
                    raise errors.too_large(CFG.MAX_URL_BYTES // (1024 * 1024))

                size = 0
                with open(dest, "wb") as output:
                    while True:
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > CFG.MAX_URL_BYTES:
                            raise errors.too_large(CFG.MAX_URL_BYTES // (1024 * 1024))
                        output.write(chunk)
                return size
            finally:
                try:
                    response.close()
                finally:
                    conn.close()
        raise errors.blocked_url("too many redirects")
    except Exception:
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except OSError:
            pass
        raise


class RateLimiter(_core.RateLimiter):
    PRUNE_EVERY = 256
    MAX_TRACKED_CLIENTS = 10_000

    def __init__(self, limit: int, window_seconds: int):
        super().__init__(limit, window_seconds)
        self._checks = 0

    def _prune_locked(self, now: float):
        stale = []
        for key, hits in self._hits.items():
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if not hits:
                stale.append(key)
        for key in stale:
            self._hits.pop(key, None)

    def check(self, key: str):
        now = time.monotonic()
        with self._lock:
            self._checks += 1
            if (self._checks % self.PRUNE_EVERY == 0
                    or len(self._hits) >= self.MAX_TRACKED_CLIENTS):
                self._prune_locked(now)

            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry = max(1, int(self.window - (now - hits[0])) + 1)
                log.warning("rate limit hit")
                raise errors.rate_limited(retry)
            hits.append(now)

            while len(self._hits) > self.MAX_TRACKED_CLIENTS:
                oldest = min(self._hits, key=lambda k: self._hits[k][0] if self._hits[k] else now)
                self._hits.pop(oldest, None)


cleanup_uploads = _core.cleanup_uploads
_cleanup_thread = None
_cleanup_lock = threading.Lock()


def start_cleanup_thread():
    """Start exactly one upload sweeper in this serving process."""
    global _cleanup_thread
    with _cleanup_lock:
        if _cleanup_thread is not None and _cleanup_thread.is_alive():
            return _cleanup_thread
        cleanup_uploads()

        def loop():
            while True:
                time.sleep(CFG.CLEANUP_INTERVAL_SECONDS)
                try:
                    cleanup_uploads()
                except Exception:
                    log.exception("cleanup sweep failed")

        _cleanup_thread = threading.Thread(
            target=loop, name="upload-cleanup", daemon=True)
        _cleanup_thread.start()
        return _cleanup_thread


def __getattr__(name):
    return getattr(_core, name)
