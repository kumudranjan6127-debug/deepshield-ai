"""Video: sampling, aggregation, and the clips that are not really clips.

`aggregate_frames` and `temporal_signals` take plain data, so most of this
runs without decoding anything. That separation is deliberate — the
interesting logic is testable against sequences whose right answer is
obvious by construction, and the file-handling is tested separately against
real files.

The case that matters most: 59 calm frames and one disaster must stay
"real". The obvious implementation — max over frames — fails it, and a
single blurred frame would then accuse an authentic video.
"""
import pytest

pytestmark = pytest.mark.video


def rec(t, p, cx=320.0, cy=240.0, size=120.0, frame=(640, 480),
        landmarks=None, thumb=None):
    """One frame record shaped exactly as predict_video emits."""
    x, y = cx - size / 2, cy - size / 2
    return {"index": int(t), "time": float(t), "pFake": p,
            "box": (x, y, size, size), "origin": (int(x), int(y)),
            "frame": frame, "landmarks": landmarks, "thumb": thumb}


@pytest.fixture
def agg():
    from inference import aggregate_frames
    return aggregate_frames


# --------------------------------------------------------- the obvious cases

def test_a_clean_clip_stays_real(agg):
    result = agg([0.02] * 60)
    assert result["score"] < 0.5
    assert result["suspicious"] == 0


def test_a_wholly_manipulated_clip_goes_fake(agg):
    result = agg([0.95] * 60)
    assert result["score"] >= 0.5
    assert result["suspicious"] == 60


def test_one_bad_frame_cannot_accuse_a_real_video(agg):
    """The false-positive case this design exists for."""
    result = agg([0.03] * 59 + [0.99])
    assert result["score"] < 0.5, f"scored {result['score']:.4f}"
    assert result["peak"] == 0.99, "the bad frame should still be reported"
    assert result["suspicious"] == 1


def test_a_tenth_of_the_frames_is_still_not_a_verdict(agg):
    assert agg([0.03] * 54 + [0.99] * 6)["score"] < 0.5


def test_sustained_manipulation_does_go_fake(agg):
    assert agg([0.95] * 30 + [0.05] * 30)["score"] >= 0.5


def test_a_max_over_frames_would_have_failed_this(agg):
    """Spelled out because it is the whole reason for top-k: the naive
    implementation calls the clean clip fake."""
    frames = [0.03] * 59 + [0.99]
    assert max(frames) >= 0.5           # what max() would have decided
    assert agg(frames)["score"] < 0.5   # what actually happens


# ---------------------------------------------------------------- components

def test_components_are_what_they_claim(agg):
    # middle two deliberately different, so the median has to average them
    result = agg([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9])
    assert result["k"] == 2
    assert result["components"]["median"] == pytest.approx(0.7)
    assert result["components"]["mean"] == pytest.approx(0.6)
    assert result["components"]["top_k"] == pytest.approx(0.9)
    assert (result["peak"], result["lowest"]) == (0.9, 0.1)


def test_the_score_is_reproducible_from_the_response(agg):
    """No hidden term: the published score is exactly the published
    weighting of the published components."""
    result = agg([0.1, 0.4, 0.6, 0.9, 0.95])
    weights = result["weights"]
    rebuilt = sum(result["components"][n] * w for n, w in weights.items())
    assert result["score"] == pytest.approx(rebuilt / sum(weights.values()))


def test_suspicious_uses_the_phase_5_boundary(agg):
    from config import CFG
    result = agg([0.1, 0.5, 0.69, 0.70, 0.95])
    assert result["suspiciousAt"] == 0.70
    assert result["suspicious"] == 2
    strong = next(low for low, key, _ in CFG.CERTAINTY_BANDS if key == "strong")
    assert result["suspiciousAt"] * 100 == strong, \
        "the suspicious threshold drifted away from the 'strong evidence' band"


# ---------------------------------------------------------------- properties

def test_properties_hold_for_any_clip(agg):
    import random
    rnd = random.Random(6)
    for _ in range(200):
        n = rnd.randint(1, 90)
        frames = [rnd.random() for _ in range(n)]
        result = agg(frames)

        assert 0.0 <= result["score"] <= 1.0
        assert result["lowest"] <= result["score"] <= result["peak"] + 1e-12
        assert result["k"] <= n

        louder = list(frames)
        i = rnd.randrange(n)
        louder[i] = min(1.0, louder[i] + 0.1)
        assert agg(louder)["score"] >= result["score"] - 1e-12, \
            "raising a frame lowered the score"


@pytest.mark.parametrize("frames,expected", [
    ([0.8], 0.8),
    ([0.4] * 5, 0.4),
    ([0.0] * 3, 0.0),
    ([1.0] * 3, 1.0),
])
def test_uniform_clips_give_that_value_back(agg, frames, expected):
    assert agg(frames)["score"] == pytest.approx(expected)
    assert agg(frames)["variance"] == pytest.approx(0.0)


def test_an_empty_clip_is_refused(agg):
    with pytest.raises(ValueError):
        agg([])


def test_weights_need_not_be_normalised(agg):
    result = agg([0.3] * 4, weights={"median": 2, "mean": 2, "top_k": 2})
    assert result["score"] == pytest.approx(0.3)


# --------------------------------------------------------------- timestamps

@pytest.mark.parametrize("seconds,expected", [
    (0, "00:00"), (7, "00:07"), (61, "01:01"), (599, "09:59"),
    (3599, "59:59"), (3600, "60:00"), (-5, "00:00"), (9.6, "00:10"),
])
def test_timestamps_format(seconds, expected):
    from inference import timestamp
    assert timestamp(seconds) == expected


# ----------------------------------------------------------------- temporal

def test_a_motionless_face_has_no_jitter():
    import numpy as np
    from inference import temporal_signals

    marks = {"right_eye": (30.0, 40.0), "left_eye": (70.0, 40.0),
             "nose": (50.0, 60.0), "mouth_right": (35.0, 80.0),
             "mouth_left": (65.0, 80.0)}
    thumb = np.tile(np.arange(32, dtype=float), (32, 1))
    signals = temporal_signals(
        [rec(t, 0.1, landmarks=marks, thumb=thumb) for t in range(10)])

    assert signals["facePositionJitter"] == pytest.approx(0.0)
    assert signals["faceSizeJitter"] == pytest.approx(0.0)
    assert signals["landmarkJitter"] == pytest.approx(0.0)
    assert signals["appearanceContinuity"] == pytest.approx(1.0, abs=1e-6)
    assert signals["facesFound"] == 10


def test_a_moving_face_shows_up_as_movement():
    import numpy as np
    from inference import temporal_signals

    marks = {"nose": (50.0, 60.0), "left_eye": (70.0, 40.0)}
    thumb = np.tile(np.arange(32, dtype=float), (32, 1))
    steady = temporal_signals(
        [rec(t, 0.1, landmarks=marks, thumb=thumb) for t in range(10)])
    jumpy = temporal_signals(
        [rec(t, 0.1, cx=320 + (120 if t % 2 else -120), landmarks=marks, thumb=thumb)
         for t in range(10)])

    assert jumpy["facePositionJitter"] > steady["facePositionJitter"]
    assert jumpy["landmarkJitter"] > steady["landmarkJitter"]


def test_size_jitter_does_not_leak_into_landmark_jitter():
    """Landmark movement is measured in face widths, so a subject walking
    towards the camera must not read as facial instability."""
    from inference import temporal_signals
    marks = {"nose": (50.0, 60.0), "left_eye": (70.0, 40.0)}
    signals = temporal_signals(
        [rec(t, 0.1, size=80 + 12 * t, landmarks=marks) for t in range(10)])
    assert signals["faceSizeJitter"] > 0.1
    assert signals["landmarkJitter"] < 0.5


def test_unrelated_frames_have_low_appearance_continuity():
    import numpy as np
    from inference import temporal_signals
    rng = np.random.default_rng(2)
    signals = temporal_signals(
        [rec(t, 0.1, thumb=rng.random((32, 32))) for t in range(10)])
    assert signals["appearanceContinuity"] < 0.3


def test_signals_are_none_when_there_is_nothing_to_measure():
    from inference import temporal_signals
    blind = [{"index": t, "time": float(t), "pFake": 0.2, "box": None,
              "origin": (0, 0), "frame": (640, 480), "landmarks": None,
              "thumb": None} for t in range(8)]
    signals = temporal_signals(blind)

    assert signals["facesFound"] == 0 and signals["framesSampled"] == 8
    for key in ("facePositionJitter", "faceSizeJitter", "landmarkJitter",
                "appearanceContinuity"):
        assert signals[key] is None, f"{key} invented a value from no faces"

    assert temporal_signals([])["facesFound"] == 0
    assert temporal_signals([rec(0, 0.2)])["landmarkJitter"] is None


# ------------------------------------------------------------- real clips

@pytest.mark.slow
def test_a_real_clip_is_analysed(engine_ready, face_video):
    result = engine_ready.analyze_file(face_video, "video")
    video = result["video"]

    assert result["prediction"] in ("real", "deepfake")
    assert result["framesAnalyzed"] == video["framesAnalyzed"] > 0
    assert len(video["timeline"]) == video["framesAnalyzed"]
    assert video["lowestFakeScore"] <= video["medianFakeScore"] <= video["peakFakeScore"]
    assert video["topTimestamps"], "no suspicious timestamps were surfaced"
    assert all(":" in m["timestamp"] for m in video["topTimestamps"])


@pytest.mark.slow
def test_the_timeline_is_ordered_in_time(engine_ready, face_video):
    timeline = engine_ready.analyze_file(face_video, "video")["video"]["timeline"]
    times = [f["t"] for f in timeline]
    assert times == sorted(times)


@pytest.mark.slow
def test_the_top_timestamps_are_the_worst_frames(engine_ready, face_video):
    video = engine_ready.analyze_file(face_video, "video")["video"]
    worst = sorted((f["p"] for f in video["timeline"]), reverse=True)
    surfaced = [m["score"] for m in video["topTimestamps"]]
    assert surfaced == pytest.approx(worst[:len(surfaced)], abs=1e-4)


@pytest.mark.slow
def test_frame_sampling_is_capped(engine_ready, long_video):
    """75 encoded frames at 1 fps against a limit of 60."""
    from config import CFG
    result = engine_ready.analyze_file(long_video, "video")
    assert result["framesAnalyzed"] == CFG.MAX_VIDEO_FRAMES, \
        f"sampled {result['framesAnalyzed']}, cap is {CFG.MAX_VIDEO_FRAMES}"


@pytest.mark.slow
def test_a_clip_with_no_faces_still_returns_a_verdict(engine_ready, no_face_video):
    result = engine_ready.analyze_file(no_face_video, "video")
    assert result["prediction"] in ("real", "deepfake")
    assert result["video"]["temporal"]["facesFound"] == 0
    for key in ("facePositionJitter", "landmarkJitter"):
        assert result["video"]["temporal"][key] is None


def test_a_corrupt_clip_is_refused(engine_ready, corrupt_video):
    with pytest.raises(Exception):
        engine_ready.analyze_file(corrupt_video, "video")


def test_an_empty_clip_is_refused(engine_ready, empty_video):
    with pytest.raises(Exception):
        engine_ready.analyze_file(empty_video, "video")


@pytest.mark.slow
def test_video_has_no_heatmap(engine_ready, face_video):
    """Occlusion sensitivity is 36 extra forwards per frame; running it
    per-frame would make a one-minute clip unusable on CPU."""
    result = engine_ready.analyze_file(face_video, "video")
    assert result.get("explain") is None
