"""DNS-rebinding-safe outbound video downloads.

URL analysis is an SSRF boundary. A hostname is allowed only after every
resolved address is public, and the socket is then connected to one of those
validated addresses directly. HTTPS still verifies the certificate against
the original hostname via SNI.

Using urllib.request after validating DNS is not enough: urllib resolves the
hostname again when it opens the socket, which creates a time-of-check /
time-of-use window for DNS rebinding.
"""
import http.client
import os
import ssl
import urllib.parse

import errors
import security
from config import CFG

_REDIRECTS = {301, 302, 303, 307, 308}


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection whose TCP peer is an already-validated IP address."""

    def __init__(self, ip: str, server_hostname: str, port: int, timeout: int):
        self._tls_server_hostname = server_hostname
        super().__init__(
            host=ip,
            port=port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )

    def connect(self):
        self.sock = self._create_connection(
            (self.host, self.port), self.timeout, self.source_address
        )
        if self._tunnel_host:
            self._tunnel()
        self.sock = self._context.wrap_socket(
            self.sock, server_hostname=self._tls_server_hostname
        )


def _port(parts: urllib.parse.SplitResult) -> int:
    """Return an explicit valid TCP port, or the scheme default."""
    try:
        port = parts.port
    except ValueError:
        raise errors.blocked_url("invalid port")
    if port is None:
        return 443 if parts.scheme == "https" else 80
    # Port zero is not a valid remote service destination. Treating it as
    # falsy and falling back to 80/443 silently changes the URL the caller
    # supplied and can make validation describe a different connection.
    if not 1 <= port <= 65535:
        raise errors.blocked_url("invalid port")
    return port


def _host_header(parts: urllib.parse.SplitResult) -> str:
    host = parts.hostname or ""
    rendered = f"[{host}]" if ":" in host and not host.startswith("[") else host
    port = _port(parts)
    default = 443 if parts.scheme == "https" else 80
    return rendered if port == default else f"{rendered}:{port}"


def _target(parts: urllib.parse.SplitResult) -> str:
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    return target


def _open_once(url: str):
    """Open one request while pinning the connection to a validated address."""
    parts = urllib.parse.urlsplit(url)
    hostname = parts.hostname or ""
    addresses = security.resolve_public(hostname)
    port = _port(parts)
    last_error = None

    for address in addresses:
        conn = None
        try:
            if parts.scheme == "https":
                conn = _PinnedHTTPSConnection(
                    address, hostname, port, CFG.URL_TIMEOUT_SECONDS
                )
            else:
                conn = http.client.HTTPConnection(
                    address, port=port, timeout=CFG.URL_TIMEOUT_SECONDS
                )
            conn.request(
                "GET",
                _target(parts),
                headers={
                    "Host": _host_header(parts),
                    "User-Agent": "DeepShield/1.0",
                    "Accept-Encoding": "identity",
                },
            )
            return conn, conn.getresponse()
        except (OSError, ssl.SSLError, http.client.HTTPException) as exc:
            last_error = exc
            if conn is not None:
                conn.close()

    detail = type(last_error).__name__ if last_error else "no address"
    raise errors.blocked_url(f"could not connect to validated host: {detail}")


def safe_download(url: str, dest: str) -> int:
    """Fetch a video with SSRF, redirect, type, timeout and size protections."""
    current = security.validate_url(url)

    try:
        for _hop in range(CFG.MAX_REDIRECTS + 1):
            conn = response = None
            try:
                conn, response = _open_once(current)

                if response.status in _REDIRECTS:
                    target = response.getheader("Location")
                    if not target:
                        raise errors.blocked_url("redirect without a destination")
                    current = security.validate_url(
                        urllib.parse.urljoin(current, target)
                    )
                    continue

                if not 200 <= response.status < 300:
                    raise errors.blocked_url(
                        f"server returned HTTP {response.status}"
                    )

                ctype = response.getheader("Content-Type", "")
                if (
                    "video" not in ctype.lower()
                    and not current.lower().split("?")[0].endswith(".mp4")
                ):
                    raise errors.not_a_video()

                declared = response.getheader("Content-Length")
                if (
                    declared
                    and declared.isdigit()
                    and int(declared) > CFG.MAX_URL_BYTES
                ):
                    raise errors.too_large(
                        CFG.MAX_URL_BYTES // (1024 * 1024)
                    )

                size = 0
                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > CFG.MAX_URL_BYTES:
                            raise errors.too_large(
                                CFG.MAX_URL_BYTES // (1024 * 1024)
                            )
                        f.write(chunk)
                return size
            finally:
                if response is not None:
                    response.close()
                if conn is not None:
                    conn.close()

        raise errors.blocked_url("too many redirects")
    except Exception:
        # A failed or oversized download must not leave a partial file behind.
        try:
            if os.path.exists(dest):
                os.remove(dest)
        except OSError:
            pass
        raise