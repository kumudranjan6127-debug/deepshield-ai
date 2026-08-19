import os
import socket
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module
import errors
import network
import security
from config import CFG


class _FakeResponse:
    def __init__(self, chunks=(b"video",), status=200, headers=None):
        self.status = status
        self._chunks = list(chunks)
        self._headers = {
            "Content-Type": "video/mp4",
            **(headers or {}),
        }
        self.closed = False

    def getheader(self, name, default=None):
        return self._headers.get(name, default)

    def read(self, _size):
        return self._chunks.pop(0) if self._chunks else b""

    def close(self):
        self.closed = True


def _public_dns(*_args, **_kwargs):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "",
         ("93.184.216.34", 0))
    ]


def test_url_download_pins_socket_to_validated_ip(monkeypatch, tmp_path):
    created = []
    response = _FakeResponse(chunks=(b"abc",))

    class FakeConnection:
        def __init__(self, host, port=None, timeout=None):
            self.host, self.port, self.timeout = host, port, timeout
            self.headers = None
            created.append(self)

        def request(self, method, target, headers=None):
            assert method == "GET"
            assert target == "/clip.mp4"
            self.headers = headers

        def getresponse(self):
            return response

        def close(self):
            pass

    monkeypatch.setattr(CFG, "ALLOW_HTTP_URLS", True)
    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(network.http.client, "HTTPConnection", FakeConnection)

    dest = tmp_path / "clip.mp4"
    assert network.safe_download("http://example.com/clip.mp4", str(dest)) == 3
    assert dest.read_bytes() == b"abc"
    assert created[0].host == "93.184.216.34"
    assert created[0].headers["Host"] == "example.com"


def test_dns_rebinding_to_loopback_is_rejected_before_connect(monkeypatch, tmp_path):
    answers = iter([
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))],
        [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))],
    ])
    connections = []

    def rebinding_dns(*_args, **_kwargs):
        return next(answers)

    class MustNotConnect:
        def __init__(self, *args, **kwargs):
            connections.append((args, kwargs))
            raise AssertionError("private rebound address reached the connector")

    monkeypatch.setattr(CFG, "ALLOW_HTTP_URLS", True)
    monkeypatch.setattr(security.socket, "getaddrinfo", rebinding_dns)
    monkeypatch.setattr(network.http.client, "HTTPConnection", MustNotConnect)

    dest = tmp_path / "rebind.mp4"
    with pytest.raises(errors.ApiError) as exc:
        network.safe_download("http://rebind.example/clip.mp4", str(dest))

    assert exc.value.code == "BLOCKED_URL"
    assert connections == []
    assert not dest.exists()


def test_failed_download_removes_partial_file(monkeypatch, tmp_path):
    response = _FakeResponse(chunks=(b"12345",))

    class FakeConnection:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            return response

        def close(self):
            pass

    monkeypatch.setattr(CFG, "ALLOW_HTTP_URLS", True)
    monkeypatch.setattr(CFG, "MAX_URL_BYTES", 4)
    monkeypatch.setattr(security.socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(network.http.client, "HTTPConnection", FakeConnection)

    dest = tmp_path / "partial.mp4"
    with pytest.raises(errors.ApiError) as exc:
        network.safe_download("http://example.com/clip.mp4", str(dest))

    assert exc.value.code == "TOO_LARGE"
    assert not dest.exists()


def test_wsgi_housekeeping_starts_once(monkeypatch):
    calls = {"cleanup": 0, "thread": 0}

    def cleanup():
        calls["cleanup"] += 1
        return 0

    def start_thread():
        calls["thread"] += 1
        return object()

    monkeypatch.setattr(app_module.security, "cleanup_uploads", cleanup)
    monkeypatch.setattr(app_module.security, "start_cleanup_thread", start_thread)
    monkeypatch.setattr(app_module, "_housekeeping_started", False)

    client = app_module.app.test_client()
    client.get("/definitely-missing-one")
    client.get("/definitely-missing-two")

    assert calls == {"cleanup": 1, "thread": 1}


def test_feedback_writes_are_rate_limited(monkeypatch):
    monkeypatch.setattr(app_module, "_housekeeping_started", True)
    monkeypatch.setattr(app_module.limiter, "limit", 1)
    monkeypatch.setattr(app_module.store, "record_feedback", lambda _entry: None)
    app_module.limiter._hits.clear()

    client = app_module.app.test_client()
    first = client.post("/api/feedback", json={"agree": True})
    second = client.post("/api/feedback", json={"agree": True})

    try:
        assert first.status_code == 200
        assert second.status_code == 429
        assert second.get_json()["error_code"] == "RATE_LIMITED"
    finally:
        app_module.limiter._hits.clear()


def test_shared_inference_calls_are_serialized(monkeypatch):
    state_lock = threading.Lock()
    barrier = threading.Barrier(2)
    active = 0
    max_active = 0

    def fake_analyze(*_args):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with state_lock:
            active -= 1
        return {"prediction": "real", "confidence": 90, "framesAnalyzed": 1}

    monkeypatch.setattr(app_module.inference, "analyze_file", fake_analyze)

    def worker():
        barrier.wait()
        app_module._run_inference("x.jpg", "image", 1.0)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_active == 1
