import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import errors
import security
import store
from config import CFG


def test_expired_one_off_clients_are_reclaimed(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    limiter = security.RateLimiter(limit=2, window_seconds=10, max_clients=3)

    for key in ("client-a", "client-b", "client-c"):
        limiter.check(key)
    assert len(limiter._hits) == 3

    # None of those callers ever return. Crossing the window and admitting a
    # new caller must sweep their stale deques instead of retaining them forever.
    now[0] = 11.0
    limiter.check("client-d")

    assert "client-d" in limiter._hits
    assert all(key not in limiter._hits for key in ("client-a", "client-b", "client-c"))
    assert len(limiter._hits) == 1


def test_fresh_one_off_client_flood_cannot_grow_map_without_bound(monkeypatch):
    monkeypatch.setattr(security.time, "monotonic", lambda: 0.0)
    limiter = security.RateLimiter(limit=2, window_seconds=60, max_clients=3)

    for key in ("client-a", "client-b", "client-c"):
        limiter.check(key)

    # New callers now share a single overflow bucket. It gets rate-limited
    # like any other client instead of allocating attacker-controlled keys.
    limiter.check("client-d")
    limiter.check("client-e")
    with pytest.raises(errors.ApiError) as exc:
        limiter.check("client-f")

    assert exc.value.code == "RATE_LIMITED"
    assert len(limiter._hits) <= limiter.max_clients + 1
    assert "client-d" not in limiter._hits
    assert "client-e" not in limiter._hits
    assert "client-f" not in limiter._hits


def test_disabling_analytics_suppresses_feedback_file_and_database(monkeypatch, tmp_path):
    feedback_path = tmp_path / "feedback.jsonl"
    writes = []

    monkeypatch.setattr(CFG, "ANALYTICS", False)
    monkeypatch.setattr(CFG, "FEEDBACK_PATH", str(feedback_path))
    monkeypatch.setattr(store, "_write_async", lambda table, row: writes.append((table, row)))

    store.record_feedback({
        "scanId": "SCAN-1",
        "prediction": "real",
        "confidence": 90,
        "fileType": "image",
        "agree": True,
    })

    assert not feedback_path.exists()
    assert writes == []
    assert store.backend_name() == "off"
