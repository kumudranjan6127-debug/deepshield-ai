import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import errors
import security


@pytest.mark.parametrize("url", [
    "https://youtube.com/watch?v=abc",
    "https://www.youtube.com/watch?v=abc",
    "https://m.tiktok.com/v/123",
    "https://sub.instagram.com/reel/abc",
    "https://x.com/example/status/1",
    "https://vimeo.com/12345",
])
def test_streaming_platform_pages_are_rejected_before_dns(monkeypatch, url):
    def must_not_resolve(*_args, **_kwargs):
        raise AssertionError("streaming-platform policy reached DNS")

    monkeypatch.setattr(security.socket, "getaddrinfo", must_not_resolve)

    with pytest.raises(errors.ApiError) as exc:
        security.validate_url(url)

    assert exc.value.code == "URL_NOT_VIDEO"
    assert "direct video" in exc.value.message.lower()


def test_unrelated_public_hosts_still_reach_dns(monkeypatch):
    seen = []

    def public_dns(host, *_args, **_kwargs):
        seen.append(host)
        import socket
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(security.socket, "getaddrinfo", public_dns)
    security.validate_url("https://media.example.com/clip.mp4")
    assert seen == ["media.example.com"]
