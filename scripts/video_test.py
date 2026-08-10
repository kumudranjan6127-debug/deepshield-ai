"""Video aggregation and temporal-signal tests.

    python scripts/video_test.py

No video files involved. The point of pulling `aggregate_frames` and
`temporal_signals` out of `predict_video` was that both take plain data,
so their behaviour can be pinned against sequences whose right answer is
obvious by construction:

    sixty calm frames and one bad one        -> must stay real
    thirty calm frames and thirty bad ones   -> must go fake

The first of those is the one that matters. A single frame that scores
0.99 because the subject blinked mid-blur must not be able to call a real
video a deepfake, and `max()` over frames — the obvious implementation —
does exactly that.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from config import CFG                                   # noqa: E402
from inference import (aggregate_frames, temporal_signals,  # noqa: E402
                       timestamp)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
    return ok


def close(a, b, tol=1e-9):
    return a is not None and b is not None and abs(a - b) <= tol


def rec(t, p, cx=320.0, cy=240.0, size=120.0, frame=(640, 480),
        landmarks=None, thumb=None):
    """One frame record shaped exactly like predict_video emits."""
    x, y = cx - size / 2, cy - size / 2
    return {"index": int(t), "time": float(t), "pFake": p,
            "box": (x, y, size, size), "origin": (int(x), int(y)),
            "frame": frame, "landmarks": landmarks, "thumb": thumb}


# --------------------------------------------------------------- aggregation

def test_obvious_cases():
    print("\nSequences whose answer is not in doubt")

    calm = aggregate_frames([0.02] * 60)
    check("a clean clip stays real", calm["score"] < 0.5, f"score {calm['score']:.4f}")
    check("and flags no suspicious frames", calm["suspicious"] == 0)

    faked = aggregate_frames([0.95] * 60)
    check("a wholly manipulated clip goes fake", faked["score"] >= 0.5,
          f"score {faked['score']:.4f}")
    check("and flags every frame", faked["suspicious"] == 60)

    # THE false-positive case: 59 calm frames, one disaster
    spike = aggregate_frames([0.03] * 59 + [0.99])
    check("one bad frame cannot call a real video fake", spike["score"] < 0.5,
          f"score {spike['score']:.4f} with peak {spike['peak']}")
    check("but the bad frame is still reported", spike["peak"] == 0.99 and
          spike["suspicious"] == 1)

    # ...and the same shape at ten times the rate must still not flip it
    burst_small = aggregate_frames([0.03] * 54 + [0.99] * 6)
    check("ten percent of frames bad is still not a verdict",
          burst_small["score"] < 0.5, f"score {burst_small['score']:.4f}")

    # Sustained manipulation across half the clip
    half = aggregate_frames([0.95] * 30 + [0.05] * 30)
    check("half the clip manipulated does go fake", half["score"] >= 0.5,
          f"score {half['score']:.4f}")


def test_components():
    print("\nComponents, computed by hand")
    # Ten frames, middle two deliberately different so the median has to
    # average them rather than pick one.
    ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]   # n = 10
    a = aggregate_frames(ps)

    check("k is 15% of the frames, rounded", a["k"] == 2, f"k={a['k']}")
    check("median of ten values averages the middle two: (0.5+0.9)/2",
          close(a["components"]["median"], 0.7), str(a["components"]["median"]))
    check("mean is the mean", close(a["components"]["mean"], 0.6, 1e-6),
          str(a["components"]["mean"]))
    check("top-k averages the k largest", close(a["components"]["top_k"], 0.9),
          str(a["components"]["top_k"]))
    check("peak and lowest are the extremes",
          a["peak"] == 0.9 and a["lowest"] == 0.1)

    # The published score must be exactly the published weighting of the
    # published components — no hidden term.
    w = a["weights"]
    rebuilt = sum(a["components"][n] * w[n] for n in w) / sum(w.values())
    check("score is reproducible from the response alone",
          close(a["score"], rebuilt, 1e-6), f"{a['score']:.6f} vs {rebuilt:.6f}")

    check("suspicious counts frames at or above the threshold",
          a["suspicious"] == 5, f"{a['suspicious']} at >= {a['suspiciousAt']}")
    check("the threshold is the Phase 5 'strong evidence' boundary",
          close(a["suspiciousAt"] * 100, 70.0),
          f"{a['suspiciousAt']} vs band {dict((k, l) for _, k, l in CFG.CERTAINTY_BANDS)}")


def test_properties():
    print("\nProperties that must hold for any clip")
    import random
    rnd = random.Random(6)

    for trial in range(200):
        n = rnd.randint(1, 90)
        ps = [rnd.random() for _ in range(n)]
        a = aggregate_frames(ps)

        if not (0.0 <= a["score"] <= 1.0):
            check(f"trial {trial}: score within [0, 1]", False, str(a["score"]))
            return
        if not (a["lowest"] <= a["score"] <= a["peak"] + 1e-12):
            check(f"trial {trial}: score lies between the extremes", False)
            return
        if a["k"] > n:
            check(f"trial {trial}: k never exceeds the frame count", False)
            return

        # More evidence must never mean a lower score
        i = rnd.randrange(n)
        louder = list(ps)
        louder[i] = min(1.0, louder[i] + 0.1)
        if aggregate_frames(louder)["score"] < a["score"] - 1e-12:
            check(f"trial {trial}: raising a frame cannot lower the score", False)
            return

    check("score within [0, 1]  (200 random clips)", True)
    check("score lies between the lowest and highest frame", True)
    check("k never exceeds the frame count", True)
    check("raising any frame never lowers the score", True)


def test_edges():
    print("\nEdges")
    one = aggregate_frames([0.8])
    check("a single frame works", one["frames"] == 1 and one["k"] == 1 and
          close(one["score"], 0.8, 1e-9), f"score {one['score']}")
    check("a single frame has no variance", one["variance"] == 0.0)

    flat = aggregate_frames([0.4] * 5)
    check("identical frames give that value back", close(flat["score"], 0.4, 1e-9))
    check("identical frames have zero variance", close(flat["variance"], 0.0))

    try:
        aggregate_frames([])
        check("an empty clip is refused", False, "no error raised")
    except ValueError:
        check("an empty clip is refused", True)

    check("weights need not be normalised",
          close(aggregate_frames([0.3] * 4, weights={"median": 2, "mean": 2,
                                                     "top_k": 2})["score"], 0.3, 1e-9))


def test_timestamps():
    print("\nTimestamps")
    for seconds, want in ((0, "00:00"), (7, "00:07"), (61, "01:01"),
                          (599, "09:59"), (3599, "59:59"), (3600, "60:00")):
        check(f"{seconds}s -> {want}", timestamp(seconds) == want, timestamp(seconds))
    check("a negative time does not produce a negative clock",
          timestamp(-5) == "00:00", timestamp(-5))
    check("fractional seconds round", timestamp(9.6) == "00:10", timestamp(9.6))


# ------------------------------------------------------------------ temporal

def test_temporal():
    print("\nTemporal signals — descriptive, never a vote")
    import numpy as np

    marks = {"right_eye": (30.0, 40.0), "left_eye": (70.0, 40.0),
             "nose": (50.0, 60.0), "mouth_right": (35.0, 80.0),
             "mouth_left": (65.0, 80.0)}
    steady_thumb = np.tile(np.arange(32, dtype=float), (32, 1))

    steady = [rec(t, 0.1, landmarks=marks, thumb=steady_thumb) for t in range(10)]
    s = temporal_signals(steady)
    check("a motionless face has no position jitter",
          close(s["facePositionJitter"], 0.0), str(s["facePositionJitter"]))
    check("...no size jitter", close(s["faceSizeJitter"], 0.0))
    check("...no landmark jitter", close(s["landmarkJitter"], 0.0))
    check("...and perfect appearance continuity",
          close(s["appearanceContinuity"], 1.0, 1e-6),
          str(s["appearanceContinuity"]))
    check("every frame counted a face", s["facesFound"] == 10 and
          s["framesSampled"] == 10)

    jumpy = [rec(t, 0.1, cx=320 + (120 if t % 2 else -120), landmarks=marks,
                 thumb=steady_thumb) for t in range(10)]
    j = temporal_signals(jumpy)
    check("a face that jumps has more position jitter",
          j["facePositionJitter"] > s["facePositionJitter"],
          f"{j['facePositionJitter']} vs {s['facePositionJitter']}")
    check("and more landmark movement",
          j["landmarkJitter"] > s["landmarkJitter"],
          f"{j['landmarkJitter']} vs {s['landmarkJitter']}")

    growing = [rec(t, 0.1, size=80 + 12 * t, landmarks=marks) for t in range(10)]
    g = temporal_signals(growing)
    check("a face approaching the camera shows size jitter",
          g["faceSizeJitter"] > 0.1, str(g["faceSizeJitter"]))
    check("...without inflating landmark jitter, which is face-relative",
          g["landmarkJitter"] is not None and g["landmarkJitter"] < 0.5,
          str(g["landmarkJitter"]))

    rng = np.random.default_rng(2)
    noisy = [rec(t, 0.1, landmarks=marks, thumb=rng.random((32, 32)))
             for t in range(10)]
    n = temporal_signals(noisy)
    check("unrelated frames have low appearance continuity",
          n["appearanceContinuity"] < 0.3, str(n["appearanceContinuity"]))


def test_temporal_without_faces():
    print("\nTemporal signals when there is nothing to measure")
    blind = [{"index": t, "time": float(t), "pFake": 0.2, "box": None,
              "origin": (0, 0), "frame": (640, 480), "landmarks": None,
              "thumb": None} for t in range(8)]
    s = temporal_signals(blind)
    check("no face found is reported, not hidden",
          s["facesFound"] == 0 and s["framesSampled"] == 8)
    check("and every signal is None rather than 0",
          all(s[k] is None for k in ("facePositionJitter", "faceSizeJitter",
                                     "landmarkJitter", "appearanceContinuity")))

    one = temporal_signals([rec(0, 0.2)])
    check("a single face cannot show frame-to-frame anything",
          one["facesFound"] == 1 and one["landmarkJitter"] is None)

    check("no records at all does not crash",
          temporal_signals([])["facesFound"] == 0)

    # Faces without landmarks or thumbnails still yield what they can
    partial = [rec(t, 0.2) for t in range(5)]
    p = temporal_signals(partial)
    check("geometry survives missing landmarks",
          p["facePositionJitter"] is not None and p["landmarkJitter"] is None)


def main():
    print("DeepShield video tests")
    test_obvious_cases()
    test_components()
    test_properties()
    test_edges()
    test_timestamps()
    test_temporal()
    test_temporal_without_faces()

    total = len(PASS) + len(FAIL)
    print("\n" + "=" * 52)
    print(f"passed {len(PASS)} / {total}")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print("  - " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
