import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import CFG
from inference import aggregate_frames, certainty_bands


def test_aggregate_normalizes_weights_and_bounds_scores():
    result = aggregate_frames([0.1, 0.2, 0.9], weights={"median": 4, "mean": 2, "top_k": 4})
    assert abs(sum(result["weights"].values()) - 1.0) < 1e-6
    assert 0.0 <= result["score"] <= 1.0
    assert result["k"] >= 1


def test_aggregate_rejects_empty_input():
    try:
        aggregate_frames([])
    except ValueError:
        return
    assert False, "empty frame sequence must fail explicitly"


def test_certainty_bands_are_ordered_and_cover_zero_to_hundred():
    bands = certainty_bands()
    assert bands[0]["from"] == 90
    assert bands[-1]["from"] == 0
    assert all(0 <= b["from"] <= 100 for b in bands)


def test_video_threshold_is_configured():
    assert 0.0 < CFG.VIDEO_SUSPICIOUS_AT < 1.0
