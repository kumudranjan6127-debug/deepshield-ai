"""What the model does with the images it will actually be given.

These are behaviour tests, not accuracy tests. `scripts/evaluate.py` measures
accuracy; this file checks the properties that must hold whatever the numbers
turn out to be — the same image gives the same answer, one face comes back as
one verdict, and the preprocessing that was added to stop two specific false
positives still works.

Where the current behaviour is known to be wrong, the test says so and pins
what it does today rather than pretending. A test that quietly encodes a bug
as correct is worse than no test.
"""
import pytest

pytestmark = [pytest.mark.inference, pytest.mark.slow]


def p_fake(inference, path):
    return inference.score_image(path)


# -------------------------------------------------------------- the basics

def test_a_generated_face_is_called_fake(engine_ready, fake_face):
    score = p_fake(engine_ready, fake_face)
    assert score > 0.5, f"a StyleGAN2 face scored {score:.4f}"


def test_an_authentic_face_is_left_alone(engine_ready, real_face):
    """The expensive error. A frame of a genuine recording must not be
    called a deepfake."""
    score = p_fake(engine_ready, real_face)
    assert score < 0.5, f"an authentic face scored {score:.4f} fake"


def test_the_two_are_clearly_separated(engine_ready, fake_face, real_face):
    """Not just on opposite sides of 0.5 — far enough apart that a small
    shift in preprocessing cannot swap them."""
    gap = p_fake(engine_ready, fake_face) - p_fake(engine_ready, real_face)
    assert gap > 0.5, f"only {gap:.4f} between a real and a generated face"


def test_the_same_image_gives_the_same_answer(engine_ready, fake_face):
    scores = [p_fake(engine_ready, fake_face) for _ in range(3)]
    assert max(scores) - min(scores) == 0, f"scores drifted: {scores}"


def test_a_reloaded_engine_agrees_with_a_warm_one(engine_ready, fake_face):
    warm = p_fake(engine_ready, fake_face)
    engine_ready._engine = None
    engine_ready._engine_mtime = None
    assert p_fake(engine_ready, fake_face) == warm


# ------------------------------------------------------------------ faces

def test_a_face_is_found_and_cropped(engine_ready, fake_face):
    from PIL import Image
    with Image.open(fake_face) as im:
        found = engine_ready._get_engine()._detect_face(im)
    assert found["found"] is True
    assert found["box"] is not None
    assert len(found["landmarks"]) == 5
    assert found["crop"].size[0] < im.size[0], "the crop is the whole frame"


def test_no_face_is_reported_as_no_face(engine_ready, no_face_image):
    from PIL import Image
    with Image.open(no_face_image) as im:
        found = engine_ready._get_engine()._detect_face(im)
    assert found["found"] is False
    assert found["box"] is None


def test_an_image_with_no_face_still_returns_a_verdict(engine_ready, no_face_image):
    """Known gap (KNOWN_ISSUES #6): the whole frame is analysed and the
    verdict comes back as though a face had been found. Pinned here so the
    day someone adds a `faceFound` flag, this test fails and gets updated
    rather than the behaviour changing unnoticed."""
    result = engine_ready.analyze_file(no_face_image, "image")
    assert result["prediction"] in ("real", "deepfake")
    assert "faceFound" not in result, \
        "a faceFound flag now exists — update this test and KNOWN_ISSUES #6"


def test_two_faces_produce_one_verdict(engine_ready, multi_face_image):
    """The largest face is the subject. Whatever else is in frame, the API
    contract is one verdict per image."""
    result = engine_ready.analyze_file(multi_face_image, "image")
    assert result["prediction"] in ("real", "deepfake")
    assert result["framesAnalyzed"] == 1
    assert len([v for v in result["ensemble"] if v["model"].endswith("(ours)")]) == 1


def test_the_larger_face_is_the_one_chosen(engine_ready, fake_face):
    """Two faces at different sizes: the crop must follow the bigger one."""
    from PIL import Image
    with Image.open(fake_face) as src:
        big = src.convert("RGB").resize((400, 400))
        small = src.convert("RGB").resize((90, 90))
    canvas = Image.new("RGB", (800, 400), (20, 20, 20))
    canvas.paste(big, (0, 0))
    canvas.paste(small, (600, 150))

    found = engine_ready._get_engine()._detect_face(canvas)
    assert found["found"] is True
    x, _, w, _ = found["box"]
    assert w > 150, f"the detector locked onto a {w:.0f}px face"
    assert x < 400, "the chosen face is on the wrong side of the frame"


def test_a_tiny_face_does_not_crash_the_pipeline(engine_ready, tiny_face_image):
    """A 64px face in a 900px frame. It may or may not be detected; what
    must not happen is an exception or a missing verdict."""
    result = engine_ready.analyze_file(tiny_face_image, "image")
    assert result["prediction"] in ("real", "deepfake")
    assert 50 <= result["confidence"] <= 100


# ----------------------------------------------------------- preprocessing

def test_a_large_image_is_capped_before_analysis(engine_ready, large_image):
    """A 2687px authentic portrait once scored 0.94 fake purely because of
    the downsampling path. Inputs are capped at 1024px to remove that."""
    from PIL import Image
    from config import CFG

    with Image.open(large_image) as im:
        assert max(im.size) > CFG.MAX_IMAGE_SIDE
        found = engine_ready._get_engine()._detect_face(im)
    assert max(found["frame"]) <= CFG.MAX_IMAGE_SIDE, \
        f"a {max(im.size)}px image reached the detector at {found['frame']}"


def test_scaling_an_image_up_does_not_change_the_verdict(engine_ready,
                                                         fake_face, large_image):
    """Same face, 3000px versus its original. The resolution cap exists so
    these agree; before it, they did not."""
    small = p_fake(engine_ready, fake_face)
    large = p_fake(engine_ready, large_image)
    assert (small > 0.5) == (large > 0.5), \
        f"verdict flipped with resolution: {small:.4f} vs {large:.4f}"


def test_heavy_compression_does_not_flip_the_verdict(engine_ready,
                                                     fake_face, compressed_image):
    """q20 is well past what a messaging app does. Detection should survive
    it — the compression normalisation exists for exactly this."""
    original = p_fake(engine_ready, fake_face)
    squashed = p_fake(engine_ready, compressed_image)
    assert (original > 0.5) == (squashed > 0.5), \
        f"verdict flipped under compression: {original:.4f} vs {squashed:.4f}"


def test_inputs_are_normalised_into_one_compression_domain(engine_ready):
    """A pristine camera original carries detail the model never saw as
    normal, and once scored 0.95 fake. Every input takes one JPEG round
    trip so nothing arrives sharper than training data."""
    import numpy as np
    from PIL import Image
    engine = engine_ready._get_engine()

    # High-frequency detail, which is exactly what a lossy round trip
    # removes — a flat colour would survive it untouched and prove nothing.
    rng = np.random.default_rng(11)
    pristine = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype="uint8"))
    normalised = engine._normalize_compression(pristine)

    assert normalised.size == pristine.size
    assert normalised.mode == "RGB"
    changed = np.abs(np.asarray(normalised, int) - np.asarray(pristine, int))
    assert changed.mean() > 1.0, "the round trip left the image untouched"

    # And it settles: compressing an already-compressed image barely moves
    # it, so a second pass is not quietly degrading every input further.
    again = np.asarray(engine._normalize_compression(normalised), int)
    settled = np.abs(again - np.asarray(normalised, int)).mean()
    assert settled < changed.mean(), \
        f"a second round trip changed as much as the first ({settled:.2f})"


# ------------------------------------------------------- explain + labels

def test_the_explanation_is_grounded_in_a_real_region(engine_ready, fake_face):
    explain = engine_ready.analyze_file(fake_face, "image")["explain"]
    assert explain["method"] == "occlusion sensitivity"
    assert explain["focusRegion"] in (
        "the eye region", "the nose area", "the mouth area", "the face overall")
    assert explain["note"].startswith("Prediction was most sensitive to")


def test_the_heatmap_is_a_real_image(engine_ready, fake_face):
    import base64
    url = engine_ready.analyze_file(fake_face, "image")["explain"]["heatmapDataUrl"]
    assert url.startswith("data:image/jpeg;base64,")
    blob = base64.b64decode(url.split(",", 1)[1])
    assert blob.startswith(b"\xff\xd8\xff"), "the heatmap is not a JPEG"


@pytest.mark.parametrize("confidence,expected", [
    (100, "very_strong"), (95, "very_strong"), (90, "very_strong"),
    (89, "strong"), (75, "strong"), (70, "strong"),
    (69, "uncertain"), (50, "uncertain"), (30, "uncertain"),
    (29, "low_evidence"), (0, "low_evidence"),
])
def test_certainty_bands_are_total(engine_ready, confidence, expected):
    assert engine_ready.certainty_for(confidence) == expected


def test_risk_and_certainty_are_independent(engine_ready):
    """A confident 'real' is low risk; a confident 'deepfake' is high. The
    certainty band says nothing about which."""
    assert engine_ready.risk_for("real", 95) == "Low"
    assert engine_ready.risk_for("deepfake", 95) == "High"
    assert engine_ready.certainty_for(95) == "very_strong"
