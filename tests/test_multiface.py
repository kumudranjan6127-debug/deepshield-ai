"""A photograph containing a manipulated face is a manipulated photograph.

The detector used to return only the largest face and the image path scored
only that one, so a group photo with a single swapped face was decided by
whichever head happened to be a few pixels wider. That is the commonest real
deepfake there is.

These tests stub the classifier rather than feeding it composite images.
A composite is not a valid probe: pasting two faces onto a canvas changes
their resolution and their background, and both move the score more than the
manipulation does. What is being tested here is the selection rule - given
per-face scores, which one decides - and that is exactly what broke.
"""
import os
import sys

import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

pytestmark = pytest.mark.inference

import inference  # noqa: E402


def faces(*scores):
    """Fake detections whose crops the stubbed classifier can tell apart."""
    return [{"crop": Image.new("RGB", (100, 100), (i, i, i)),
             "landmarks": {}, "box": (0, 0, 10, 10), "origin": (0, 0),
             "frame": (100, 100), "found": True}
            for i, _ in enumerate(scores)]


@pytest.fixture
def stub(monkeypatch):
    """Score faces by position in the list, without touching the model."""
    def install(detections, p_fakes):
        import numpy as np
        eng = inference._get_engine()
        monkeypatch.setattr(eng, "_detect_faces",
                            lambda im, limit=None: detections)
        order = {id(d["crop"]): p for d, p in zip(detections, p_fakes)}
        fake_i = eng.classes.index("fake")

        def scores(crop):
            p = order[id(crop)]
            out = np.zeros(len(eng.classes))
            out[fake_i] = p
            out[1 - fake_i] = 1 - p
            return out
        monkeypatch.setattr(eng, "_probs_raw", scores)
        monkeypatch.setattr(inference, "_get_hf_engines", lambda: [])
        monkeypatch.setattr(eng, "explain", lambda *a, **k: None)
        return eng
    return install


def test_one_fake_face_among_real_ones_decides(stub, fake_face):
    """The whole point. Three convincing faces and one swap is a deepfake."""
    detections = faces(0, 0, 0, 0)
    stub(detections, [0.02, 0.01, 0.97, 0.03])   # the third is manipulated

    result = inference.analyze_file(fake_face, "image")
    assert result["prediction"] == "deepfake", \
        "a manipulated face was outvoted by the people standing next to it"
    assert result["facesFound"] == 4


def test_the_largest_face_does_not_win_by_being_largest(stub, fake_face):
    """The old rule, stated as a test so it cannot come back: detections
    arrive largest-first, and the first one being clean must not settle it."""
    detections = faces(0, 0)
    stub(detections, [0.01, 0.99])               # largest is clean, other is not

    assert inference.analyze_file(fake_face, "image")["prediction"] == "deepfake"


def test_all_real_faces_stay_real(stub, fake_face):
    """The cost of the rule above is false positives on group photos, so the
    ordinary case is pinned too."""
    detections = faces(0, 0, 0)
    stub(detections, [0.03, 0.01, 0.04])

    result = inference.analyze_file(fake_face, "image")
    assert result["prediction"] == "real"
    assert result["facesFound"] == 3


def test_a_single_face_behaves_exactly_as_before(stub, fake_face):
    detections = faces(0)
    stub(detections, [0.91])
    result = inference.analyze_file(fake_face, "image")
    assert result["prediction"] == "deepfake"
    assert result["facesFound"] == 1
    assert result["faceFound"] is True


def test_no_face_is_reported_rather_than_hidden(fake_face, monkeypatch):
    """With no face the whole frame is scored, which is a reasonable
    fallback and a terrible thing to present as a verdict on a face."""
    eng = inference._get_engine()
    miss = [{"crop": Image.new("RGB", (64, 64)), "landmarks": None,
             "box": None, "origin": (0, 0), "frame": (64, 64), "found": False}]
    monkeypatch.setattr(eng, "_detect_faces", lambda im, limit=None: miss)

    result = inference.analyze_file(fake_face, "image")
    assert result["faceFound"] is False
    assert result["facesFound"] == 0


def test_the_face_cap_is_honoured(monkeypatch):
    """A crowd shot must not turn into a hundred forward passes."""
    from config import CFG
    monkeypatch.setattr(CFG, "MAX_FACES", 3)
    eng = inference._get_engine()

    import numpy as np
    fake_rows = np.zeros((10, 15), dtype=np.float32)
    for i in range(10):
        fake_rows[i, :4] = [10 * i, 10, 40 + i, 40 + i]

    class Detector:
        def setInputSize(self, size): pass
        def detect(self, bgr): return 1, fake_rows
    monkeypatch.setattr(eng, "_detector", Detector())

    found = eng._detect_faces(Image.new("RGB", (400, 300), (128, 128, 128)))
    assert len(found) == 3, f"cap ignored: {len(found)} faces scored"
