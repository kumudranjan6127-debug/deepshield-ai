"""Shared fixtures for the DeepShield test suite.

Two decisions shape everything here.

**No running server.** The suites this replaced needed `python backend/app.py`
in another terminal and `DS_RATE_LIMIT=50` in both, and they throttled
themselves when you forgot. Flask's test client exercises the same routes,
the same validation and the same engine in-process, so `pytest` is the whole
command.

**No committed test media.** Every image and clip below is generated from
material already in the repository — the sample StyleGAN2 faces and the
authentic-portrait clip — so the fixtures are deterministic, cost nothing in
git, and cannot rot into "some binary someone added once".

The one thing that cannot be generated is an authentic face. `real_face` is
extracted from `training/video_test/authentic_portrait.mp4`, a real recording
of a real person; the engine scores it 0.02 fake. Without it the suite could
only ever check that fakes look fake.
"""
import glob
import os
import shutil
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SAMPLE_FACES = os.path.join(ROOT, "training", "tpdn_test")
SAMPLE_VIDEOS = os.path.join(ROOT, "training", "video_test")


# --------------------------------------------------------------- machinery

def _built(name):
    os.makedirs(FIXTURES, exist_ok=True)
    return os.path.join(FIXTURES, name)


def _faces():
    return sorted(glob.glob(os.path.join(SAMPLE_FACES, "*.jpeg")))


def _authentic_clip():
    path = os.path.join(SAMPLE_VIDEOS, "authentic_portrait.mp4")
    return path if os.path.exists(path) else None


@pytest.fixture(scope="session", autouse=True)
def _clean_fixtures():
    """Rebuild the fixture folder each session, so a half-written file left
    by an interrupted run cannot quietly become the thing under test."""
    if os.path.isdir(FIXTURES):
        shutil.rmtree(FIXTURES, ignore_errors=True)
    os.makedirs(FIXTURES, exist_ok=True)
    yield


# ------------------------------------------------------------------- app

@pytest.fixture(scope="session")
def flask_app():
    import app as app_module
    app_module.app.config.update(TESTING=True)
    return app_module.app


@pytest.fixture
def client(flask_app):
    """A fresh client with the rate limiter cleared.

    The limiter is per-process and per-client-key, so without this the
    sixth request of the session starts failing and every later test
    reports a rate-limit error instead of whatever it meant to check."""
    import app as app_module
    app_module.limiter._hits.clear()
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(scope="session")
def engine_ready():
    import inference
    if not inference.engine_available():
        pytest.skip("no model available")
    return inference


# ---------------------------------------------------------------- images

@pytest.fixture(scope="session")
def fake_face():
    """A StyleGAN2 face. The model should call this one fake."""
    faces = _faces()
    if not faces:
        pytest.skip("no sample faces in training/tpdn_test")
    return faces[0]


@pytest.fixture(scope="session")
def real_face():
    """A frame of a genuine recording. The model should leave it alone."""
    clip = _authentic_clip()
    if not clip:
        pytest.skip("authentic_portrait.mp4 not present")
    import cv2
    target = _built("real_face.jpg")
    if not os.path.exists(target):
        cap = cv2.VideoCapture(clip)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            pytest.skip("could not read a frame from the authentic clip")
        cv2.imwrite(target, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return target


@pytest.fixture(scope="session")
def no_face_image():
    """A picture with nothing face-like in it — a smooth colour gradient."""
    import numpy as np
    from PIL import Image
    target = _built("no_face.png")
    if not os.path.exists(target):
        h = w = 480
        y, x = np.mgrid[0:h, 0:w]
        rgb = np.stack([x * 255 // w, y * 255 // h,
                        (x + y) * 255 // (w + h)], axis=-1).astype("uint8")
        Image.fromarray(rgb).save(target)
    return target


@pytest.fixture(scope="session")
def multi_face_image():
    """Two faces side by side. Exactly one verdict may come back."""
    faces = _faces()
    if len(faces) < 2:
        pytest.skip("need two sample faces")
    from PIL import Image
    target = _built("multi_face.jpg")
    if not os.path.exists(target):
        canvas = Image.new("RGB", (800, 400))
        canvas.paste(Image.open(faces[0]).convert("RGB").resize((400, 400)), (0, 0))
        canvas.paste(Image.open(faces[1]).convert("RGB").resize((400, 400)), (400, 0))
        canvas.save(target, "JPEG", quality=92)
    return target


@pytest.fixture(scope="session")
def tiny_face_image():
    """One small face adrift in a large frame — the detector's hard case."""
    faces = _faces()
    if not faces:
        pytest.skip("no sample faces")
    from PIL import Image
    target = _built("tiny_face.jpg")
    if not os.path.exists(target):
        canvas = Image.new("RGB", (900, 900), (30, 30, 40))
        canvas.paste(Image.open(faces[0]).convert("RGB").resize((64, 64)), (420, 400))
        canvas.save(target, "JPEG", quality=90)
    return target


@pytest.fixture(scope="session")
def large_image():
    """Well past the 1024px working cap — which exists because a 2687px
    portrait once scored 0.94 fake purely from the downsampling path."""
    faces = _faces()
    if not faces:
        pytest.skip("no sample faces")
    from PIL import Image
    target = _built("large_face.jpg")
    if not os.path.exists(target):
        Image.open(faces[0]).convert("RGB").resize(
            (3000, 3000), Image.LANCZOS).save(target, "JPEG", quality=90)
    return target


@pytest.fixture(scope="session")
def compressed_image():
    """The same face after a messaging app has had it."""
    faces = _faces()
    if not faces:
        pytest.skip("no sample faces")
    from PIL import Image
    target = _built("compressed.jpg")
    if not os.path.exists(target):
        Image.open(faces[0]).convert("RGB").resize((640, 640)).save(
            target, "JPEG", quality=20)
    return target


@pytest.fixture(scope="session")
def png_bytes():
    """A small valid PNG, comfortably above the minimum accepted size."""
    import io
    import numpy as np
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.zeros((128, 128, 3), "uint8")).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- videos

def _write_clip(target, frames, fps, size):
    import cv2
    writer = cv2.VideoWriter(target, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()
    return os.path.exists(target) and os.path.getsize(target) > 0


@pytest.fixture(scope="session")
def face_video():
    """A short clip of faces — the ordinary case."""
    faces = _faces()
    if not faces:
        pytest.skip("no sample faces")
    import cv2
    target = _built("faces.mp4")
    if not os.path.exists(target):
        imgs = [cv2.resize(cv2.imread(p), (320, 320)) for p in faces[:2]]
        frames = [im for _ in range(3) for im in imgs for _ in range(5)]
        if not _write_clip(target, frames, 5.0, (320, 320)):
            pytest.skip("no working mp4 encoder")
    return target


@pytest.fixture(scope="session")
def long_video():
    """More sampled frames than MAX_VIDEO_FRAMES, so the cap is exercised.

    Encoded at 1 fps, so the sampler takes every frame: 75 encoded frames
    means 75 candidates against a limit of 60."""
    faces = _faces()
    if not faces:
        pytest.skip("no sample faces")
    import cv2
    target = _built("long.mp4")
    if not os.path.exists(target):
        img = cv2.resize(cv2.imread(faces[0]), (128, 128))
        if not _write_clip(target, [img] * 75, 1.0, (128, 128)):
            pytest.skip("no working mp4 encoder")
    return target


@pytest.fixture(scope="session")
def no_face_video(no_face_image):
    """A clip the face detector will find nothing in."""
    import cv2
    target = _built("no_face.mp4")
    if not os.path.exists(target):
        img = cv2.resize(cv2.imread(no_face_image), (256, 256))
        if not _write_clip(target, [img] * 20, 5.0, (256, 256)):
            pytest.skip("no working mp4 encoder")
    return target


@pytest.fixture(scope="session")
def empty_video():
    """Zero bytes behind a video extension."""
    target = _built("empty.mp4")
    if not os.path.exists(target):
        open(target, "wb").close()
    return target


@pytest.fixture(scope="session")
def corrupt_video():
    """A valid mp4 header followed by nonsense — it passes the magic-byte
    check on purpose, so the decoder is what has to refuse it."""
    target = _built("corrupt.mp4")
    if not os.path.exists(target):
        with open(target, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftypisom" + bytes(range(256)) * 8)
    return target
