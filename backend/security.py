"""Application URL policy layered on the existing request validation code."""
from __future__ import annotations

import urllib.parse
import security_core as _core

STREAMING_PAGE_HOSTS = (
    "youtube.com", "youtu.be", "instagram.com", "tiktok.com",
    "facebook.com", "fb.watch", "twitter.com", "x.com",
)

_base_validate_url = _core.validate_url


def _host_matches(host: str, domain: str) -> bool:
    host = (host or "").lower().rstrip(".")
    domain = domain.lower().rstrip(".")
    return host == domain or host.endswith("." + domain)


def validate_url(url: str) -> str:
    host = urllib.parse.urlparse(url or "").hostname or ""
    if any(_host_matches(host, domain) for domain in STREAMING_PAGE_HOSTS):
        raise _core.errors.blocked_url(
            "Streaming-platform page URLs are not direct media files. "
            "Upload the video or provide a direct downloadable media URL instead."
        )
    return _base_validate_url(url)


# Redirect hops use the same application-level policy.
_core.validate_url = validate_url
safe_download = _core.safe_download


def __getattr__(name):
    return getattr(_core, name)
