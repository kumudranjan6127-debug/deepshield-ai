"""The model is what it says it is, and both runtimes agree.

Deployment runs ONNX through OpenCV; the checkpoint it was exported from is
PyTorch. If those two ever disagree, every number this project has published
is about a model nobody is running. Parity is skipped rather than failed when
PyTorch is absent, because a deployment does not install it — that is the
point of the ONNX export.

Identity is checked in three places at once (metadata file, loaded engine,
API response) because the bug it replaced was exactly a drift between them:
`/api` reported MobileNetV3-Small / 2.5M / PyTorch for a Large ONNX model.
"""
import json
import os

import pytest

pytestmark = pytest.mark.parity

PARITY_THRESHOLD = 1e-4


@pytest.fixture(scope="session")
def metadata():
    from config import CFG
    if not os.path.exists(CFG.ONNX_META_PATH):
        pytest.skip("no ONNX metadata present")
    with open(CFG.ONNX_META_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- identity

@pytest.mark.parametrize("field", [
    "model_name", "architecture", "version", "runtime", "input_size", "classes",
])
def test_the_identity_block_is_complete(engine_ready, field):
    info = engine_ready.engine_info()
    assert info.get(field) not in (None, "", []), f"engine_info() lost {field}"


@pytest.mark.parametrize("field", [
    "model_name", "architecture", "version", "runtime", "input_size",
])
def test_the_engine_agrees_with_its_metadata_file(engine_ready, metadata, field):
    info = engine_ready.engine_info()
    assert str(info.get(field)) == str(metadata.get(field)), \
        f"{field}: engine says {info.get(field)!r}, file says {metadata.get(field)!r}"


@pytest.mark.parametrize("field", [
    "model_name", "architecture", "version", "runtime", "input_size",
])
def test_the_api_agrees_with_the_engine(client, engine_ready, field):
    published = client.get("/api/health").get_json()
    assert str(published.get(field)) == str(engine_ready.engine_info().get(field))


def test_class_order_is_fixed(engine_ready):
    """Index 0 is fake. Swapping these silently inverts every verdict."""
    assert list(engine_ready.engine_info()["classes"]) == ["fake", "real"]


def test_the_runtime_matches_the_active_backend(engine_ready):
    info = engine_ready.engine_info()
    assert (info["runtime"] == "ONNX") == (info["backend"] == "onnx")


def test_the_architecture_is_not_a_stale_small(engine_ready):
    info = engine_ready.engine_info()
    if "small" not in info["arch"]:
        assert "Small" not in info["architecture"], \
            "a Large model is describing itself as Small again"


def test_the_version_comes_from_the_model_not_the_app(engine_ready):
    version = engine_ready.engine_info()["version"]
    assert version and version != "1.0.0", \
        "model.version is reporting the application version again"


# ------------------------------------------------------------ reproducible

def test_repeated_analysis_is_bit_identical(engine_ready, fake_face):
    runs = [engine_ready.analyze_file(fake_face, "image") for _ in range(3)]
    scores = [r["ensemble"][0]["pFake"] for r in runs]
    assert max(scores) - min(scores) == 0, f"probability drifted: {scores}"
    assert len({(r["prediction"], r["confidence"]) for r in runs}) == 1
    assert len({r["explain"]["focusRegion"] for r in runs}) == 1


# ---------------------------------------------------------------- parity

@pytest.fixture(scope="session")
def torch_model():
    from config import CFG
    pytest.importorskip("torch", reason="PyTorch is not installed in a deployment")
    pytest.importorskip("torchvision")
    import torch
    from torchvision import models

    if not os.path.exists(CFG.CKPT_PATH):
        pytest.skip("no .pth checkpoint to compare against")

    checkpoint = torch.load(CFG.CKPT_PATH, map_location="cpu", weights_only=False)
    builders = {"mobilenet_v3_small": models.mobilenet_v3_small,
                "mobilenet_v3_large": models.mobilenet_v3_large}
    net = builders[checkpoint["arch"]](weights=None)
    net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features,
                                        len(checkpoint["classes"]))
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net


@pytest.mark.slow
def test_onnx_and_pytorch_agree(engine_ready, torch_model, fake_face, real_face):
    """Measured on real images through the real preprocessing, not on random
    tensors — a graph can match on noise and diverge on faces."""
    import torch
    from PIL import Image

    engine = engine_ready._get_engine()
    worst = 0.0
    for path in (fake_face, real_face):
        with Image.open(path) as im:
            face, _ = engine._face_crop_ex(im)
        prepared = engine._normalize_compression(face.convert("RGB"))
        batch = engine.np.stack([engine._to_input(prepared)])

        onnx_probs = engine._forward(batch)[0]
        with torch.no_grad():
            torch_probs = torch.softmax(torch_model(torch.from_numpy(batch)),
                                        dim=1)[0].numpy()

        diff = float(abs(onnx_probs - torch_probs).max())
        worst = max(worst, diff)
        assert diff < PARITY_THRESHOLD, (
            f"{os.path.basename(path)}: {diff:.2e} — onnx {onnx_probs[0]:.6f} "
            f"vs torch {torch_probs[0]:.6f}")

    assert worst < PARITY_THRESHOLD


# ---------------------------------------------------------- certainty bands

def test_the_bands_are_contiguous_and_total(engine_ready):
    bands = engine_ready.certainty_bands()
    assert bands[0]["to"] == 100 and bands[-1]["from"] == 0
    for upper, lower in zip(bands, bands[1:]):
        assert lower["to"] == upper["from"], "a gap opened between bands"


def test_every_confidence_maps_to_exactly_one_band(engine_ready):
    bands = engine_ready.certainty_bands()
    for confidence in range(0, 101):
        owners = [b for b in bands
                  if b["from"] <= confidence and
                  (confidence < b["to"] or b["to"] == 100)]
        assert len({b["key"] for b in owners}) == 1, \
            f"confidence {confidence} matched {[b['key'] for b in owners]}"
        assert owners[0]["key"] == engine_ready.certainty_for(confidence)


def test_the_api_publishes_the_same_band_table(client, engine_ready):
    assert client.get("/api/health").get_json()["certainty_bands"] == \
        engine_ready.certainty_bands()
