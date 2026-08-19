import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import security


def test_known_streaming_pages_are_rejected_before_network(monkeypatch):
    called = {"base": False}

    def should_not_run(url):
        called["base"] = True
        return url

    monkeypatch.setattr(security, "_base_validate_url", should_not_run)
    urls = [
        "https://www.youtube.com/watch?v=abc",
        "https://youtu.be/abc",
        "https://instagram.com/reel/abc",
        "https://m.tiktok.com/v/abc",
        "https://www.facebook.com/watch/?v=1",
        "https://fb.watch/abc",
        "https://x.com/user/status/1",
    ]
    for url in urls:
        try:
            security.validate_url(url)
        except Exception:
            pass
        else:
            raise AssertionError(f"streaming page was accepted: {url}")
    assert called["base"] is False


def test_substring_lookalike_host_is_not_blocked(monkeypatch):
    monkeypatch.setattr(security, "_base_validate_url", lambda url: url)
    url = "https://notyoutube.com/video.mp4"
    assert security.validate_url(url) == url


def test_allowed_host_still_uses_existing_validation(monkeypatch):
    seen = []
    monkeypatch.setattr(security, "_base_validate_url", lambda url: seen.append(url) or url)
    url = "https://cdn.example.org/video.mp4"
    assert security.validate_url(url) == url
    assert seen == [url]
