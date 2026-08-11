"""The training preprocessor must produce exactly what production produces.

`training/deepshield_preprocess.py` is a deliberate copy: it has to run
inside a Kaggle session where `backend/` does not exist, so it cannot import
the engine. A copy can drift, and drift here is invisible — the model still
trains, the numbers still look fine, and they describe a pipeline nobody is
served.

So the copy is pinned. These tests assert **bit-identical** output, stage by
stage and end to end, on the sample images in the repository. Change one side
without the other and this file fails.

The tolerance is zero. Not `allclose` — identical bytes. Two float pipelines
that differ by 1e-7 are two different pipelines, and the whole point of this
module is that there is only one.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "training"))

pytestmark = pytest.mark.parity

YUNET = os.path.join(ROOT, "models", "face_detection_yunet.onnx")


@pytest.fixture(scope="module")
def trainer():
    from deepshield_preprocess import Preprocessor
    if not os.path.exists(YUNET):
        pytest.skip("YuNet model not present")
    return Preprocessor(YUNET)


@pytest.fixture(scope="module")
def production(engine_ready):
    return engine_ready._get_engine()


def open_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


# ------------------------------------------------------- the constants agree

def test_the_config_mirrors_production():
    """Every constant, checked against `backend/config.py` rather than
    against a comment."""
    from config import CFG
    from deepshield_preprocess import PRODUCTION

    assert PRODUCTION.max_side == CFG.MAX_IMAGE_SIDE
    assert PRODUCTION.jpeg_quality == CFG.JPEG_NORMALISE_QUALITY
    assert PRODUCTION.margin == 0.35
    assert PRODUCTION.score_threshold == 0.6
    assert PRODUCTION.nms_threshold == 0.3
    assert PRODUCTION.top_k == 5000
    assert PRODUCTION.detector_input == (320, 320)


def test_the_model_contract_matches_the_shipped_metadata(engine_ready):
    """Input size, normalisation and class order come from the model's own
    metadata; hardcoding a different value here would train for a model that
    is not the one deployed."""
    from deepshield_preprocess import PRODUCTION
    info = engine_ready.engine_info()

    assert PRODUCTION.input_size == info["input_size"]
    assert list(PRODUCTION.classes) == list(info["classes"])
    assert PRODUCTION.classes[0] == "fake", "index 0 must stay fake"

    engine = engine_ready._get_engine()
    assert list(PRODUCTION.mean) == list(engine.norm["mean"])
    assert list(PRODUCTION.std) == list(engine.norm["std"])


# ----------------------------------------------------------- stage by stage

def test_resolution_cap_is_identical(trainer, production, large_image):
    """Truncating ints, INTER_AREA, and the same threshold — an off-by-one
    here shifts every crop that follows."""
    import cv2
    from config import CFG

    source = open_rgb(large_image)
    ours = trainer.cap_resolution(source)

    rgb = np.array(source)
    longest = max(rgb.shape[:2])
    scale = CFG.MAX_IMAGE_SIDE / longest
    theirs = cv2.resize(rgb, (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA)

    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)
    assert max(ours.shape[:2]) == CFG.MAX_IMAGE_SIDE


def test_a_small_image_is_not_resized(trainer, fake_face):
    source = open_rgb(fake_face)
    if max(source.size) > trainer.cfg.max_side:
        pytest.skip("sample is larger than the cap")
    assert np.array_equal(trainer.cap_resolution(source), np.array(source))


def test_jpeg_normalisation_is_identical(trainer, production, fake_face):
    source = open_rgb(fake_face)
    ours = trainer.normalize_compression(source)
    theirs = production._normalize_compression(source)
    assert np.array_equal(np.array(ours), np.array(theirs))


def test_tensor_conversion_is_identical(trainer, production, fake_face):
    """Resize interpolation, /255, mean/std, and the HWC→CHW transpose."""
    source = open_rgb(fake_face)
    ours = trainer.to_input(source)
    theirs = production._to_input(source)

    assert ours.dtype == theirs.dtype == np.float32
    assert ours.shape == theirs.shape == (3, 224, 224)
    assert np.array_equal(ours, theirs), \
        f"max difference {np.abs(ours - theirs).max():.3e}"


@pytest.mark.parametrize("fixture_name", ["fake_face", "real_face",
                                          "large_image", "compressed_image",
                                          "multi_face_image", "tiny_face_image"])
def test_the_crop_is_identical(trainer, production, request, fixture_name):
    """The crop is where a mismatch would do the most damage: a different
    box means the model trains on a different framing than it is served."""
    source = open_rgb(request.getfixturevalue(fixture_name))

    ours = trainer.detect_face(source)
    theirs = production._detect_face(source)

    assert ours.found == theirs["found"], "detectors disagree on whether a face is there"
    assert ours.frame == theirs["frame"]
    assert ours.origin == theirs["origin"]
    if ours.found:
        assert ours.box == pytest.approx(theirs["box"], abs=1e-6)
        assert ours.crop.size == theirs["crop"].size
        assert set(ours.landmarks) == set(theirs["landmarks"])
        for name, point in ours.landmarks.items():
            assert point == pytest.approx(theirs["landmarks"][name], abs=1e-4)
    assert np.array_equal(np.array(ours.crop), np.array(theirs["crop"]))


@pytest.mark.parametrize("fixture_name", ["fake_face", "real_face",
                                          "large_image", "compressed_image",
                                          "no_face_image"])
def test_the_whole_baseline_is_identical(trainer, production, request, fixture_name):
    """End to end: cap → detect → crop → q88 → resize → normalise.

    This is the assertion the module exists for. Zero tolerance."""
    source = open_rgb(request.getfixturevalue(fixture_name))

    ours = trainer.baseline_tensor(source)
    crop, _ = production._face_crop_ex(source)
    theirs = production._to_input(production._normalize_compression(crop.convert("RGB")))

    assert np.array_equal(ours, theirs), \
        f"{fixture_name}: max difference {np.abs(ours - theirs).max():.3e}"


def test_the_baseline_matches_what_the_engine_actually_scores(trainer, production,
                                                              fake_face):
    """Production averages over the image and its mirror (TTA). The
    deterministic half of that must be our baseline exactly, so a training
    sample and the un-mirrored inference pass see the same tensor."""
    source = open_rgb(fake_face)
    ours = trainer.baseline_tensor(source)

    crop, _ = production._face_crop_ex(source)
    prepared = production._normalize_compression(crop.convert("RGB"))
    theirs = production.np.stack([production._to_input(prepared),
                                  production._to_input(prepared.transpose(0))])
    assert np.array_equal(ours, theirs[0])


# --------------------------------------------------------------- properties

def test_the_evaluation_transform_is_deterministic(trainer, fake_face):
    """Run it five times; get the same bytes five times. Anything else and
    a benchmark number is not reproducible."""
    source = open_rgb(fake_face)
    first = trainer.baseline_tensor(source)
    for _ in range(4):
        assert np.array_equal(trainer.baseline_tensor(source), first)


def test_evaluation_never_augments(trainer, fake_face):
    """`training_tensor(augment=None)` and `baseline_tensor` are the same
    path — there is no third pipeline."""
    source = open_rgb(fake_face)
    assert np.array_equal(trainer.training_tensor(source, augment=None),
                          trainer.baseline_tensor(source))


def test_augmentation_actually_changes_the_image(trainer, fake_face):
    from deepshield_preprocess import TrainingAugmentation
    source = open_rgb(fake_face)
    aug = TrainingAugmentation(seed=7)

    baseline = trainer.baseline_tensor(source)
    augmented = trainer.training_tensor(source, augment=aug)
    assert not np.array_equal(baseline, augmented), "the augmentation did nothing"
    assert augmented.shape == baseline.shape
    assert augmented.dtype == np.float32


def test_augmentation_is_seeded(trainer, fake_face):
    """Two runs with the same seed produce the same sample, so a training
    run is reproducible."""
    from deepshield_preprocess import TrainingAugmentation
    source = open_rgb(fake_face)
    a = trainer.training_tensor(source, augment=TrainingAugmentation(seed=3))
    b = trainer.training_tensor(source, augment=TrainingAugmentation(seed=3))
    c = trainer.training_tensor(source, augment=TrainingAugmentation(seed=4))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_the_q88_round_trip_stays_last(trainer, fake_face):
    """The ordering requirement. Augmentation sits between the crop and the
    q88 tail; if it ran after, q88 would no longer be the final compression
    step and the parity this module exists for would be gone.

    Checked by construction: an augmented sample must still equal
    `to_input(normalize_compression(augmented_crop))`."""
    from deepshield_preprocess import TrainingAugmentation
    source = open_rgb(fake_face)

    aug = TrainingAugmentation(seed=11)
    crop = trainer.detect_face(source).crop
    expected = trainer.to_input(trainer.normalize_compression(aug(crop)))

    aug_again = TrainingAugmentation(seed=11)
    produced = trainer.training_tensor(source, augment=aug_again)
    assert np.array_equal(produced, expected)


def test_a_prepared_crop_skips_detection(trainer, fake_face):
    """Dataset preparation saves crops once; training must not re-detect on
    an already-cropped image, which would crop a crop."""
    source = open_rgb(fake_face)
    crop = trainer.detect_face(source).crop

    assert np.array_equal(trainer.baseline_tensor(crop, already_cropped=True),
                          trainer.to_input(trainer.normalize_compression(crop)))
    assert np.array_equal(trainer.baseline_tensor(source),
                          trainer.baseline_tensor(crop, already_cropped=True))


def test_no_face_is_reported_not_hidden(trainer, no_face_image):
    """Preparation needs to know, so those samples can be dropped rather
    than teaching the model to classify backgrounds."""
    result = trainer.detect_face(open_rgb(no_face_image))
    assert result.found is False
    assert result.box is None
    assert result.crop.size == result.frame


def test_detection_reports_what_quality_filtering_needs(trainer, fake_face):
    result = trainer.detect_face(open_rgb(fake_face))
    assert result.found is True
    assert result.n_faces >= 1
    assert result.score is None or 0.0 <= result.score <= 1.0
    assert len(result.landmarks) == 5


# ------------------------------------------------------------ the model file

def test_a_missing_yunet_fails_loudly(tmp_path):
    """Requirement 9: no silent download, and no silent fallback to a
    different crop either."""
    from deepshield_preprocess import Preprocessor
    with pytest.raises(FileNotFoundError) as caught:
        Preprocessor(tmp_path / "not_here.onnx")
    message = str(caught.value)
    assert "models/face_detection_yunet.onnx" in message
    assert "opencv_zoo" in message


def test_the_module_needs_no_backend_import():
    """It has to run in a Kaggle session where `backend/` does not exist."""
    import re
    source = open(os.path.join(ROOT, "training", "deepshield_preprocess.py"),
                  encoding="utf-8").read()
    code = re.sub(r'"""[\s\S]*?"""', "", source)          # strip docstrings
    for forbidden in ("import inference", "from inference",
                      "import config", "from config",
                      "import app", "from app", "import torch"):
        assert forbidden not in code, f"module imports {forbidden!r}"
