"""Model correctness tests.

    python scripts/model_test.py

Three properties that must hold for a result to be trustworthy:

  identity        the model reports what it actually is, consistently,
                  everywhere it is exposed
  reproducibility the same image analysed repeatedly gives the same
                  probability — no hidden randomness in preprocessing,
                  augmentation or the runtime
  parity          the shipped ONNX graph and the PyTorch checkpoint it
                  was exported from agree to better than 1e-4

Parity is skipped, not failed, when PyTorch is absent — the deployment
does not install it.
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

PARITY_THRESHOLD = 1e-4
PASS, FAIL, SKIP = [], [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


def skip(name, why):
    SKIP.append(name)
    print(f"  [SKIP] {name}  — {why}")


def sample_images(limit=4):
    files = sorted(glob.glob(os.path.join(ROOT, "training", "tpdn_test", "*")))
    return [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))][:limit]


# ------------------------------------------------------------------ identity

def test_identity():
    import inference
    from config import CFG

    print("\nIdentity — the model reports what it is")
    if not inference.engine_available():
        skip("identity", "no model available")
        return

    info = inference.engine_info()
    required = ("model_name", "architecture", "version", "runtime",
                "input_size", "classes")
    missing = [k for k in required if info.get(k) in (None, "")]
    check("identity block complete", not missing,
          f"missing: {missing}" if missing else ", ".join(
              f"{k}={info[k]}" for k in required))

    with open(CFG.ONNX_META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    # The file on disk and the loaded engine must not drift apart
    for key in ("model_name", "architecture", "version", "runtime", "input_size"):
        check(f"{key} matches the metadata file",
              str(info.get(key)) == str(meta.get(key)),
              f"engine={info.get(key)!r} file={meta.get(key)!r}")

    check("classes are ['fake', 'real'] in that order",
          list(info["classes"]) == ["fake", "real"], str(info["classes"]))
    check("architecture is not a stale Small",
          "Small" not in str(info["architecture"]) or "small" in info["arch"],
          info["architecture"])
    check("runtime matches the active backend",
          (info["runtime"] == "ONNX") == (info["backend"] == "onnx"),
          f"runtime={info['runtime']} backend={info['backend']}")

    # And what the API hands the frontend must agree with both
    import app
    with app.app.test_request_context():
        identity = app.model_identity()
    for key in ("model_name", "architecture", "version", "runtime", "input_size"):
        check(f"/api/health agrees on {key}",
              str(identity.get(key)) == str(info.get(key)),
              f"api={identity.get(key)!r}")


# ------------------------------------------------------------ reproducibility

def test_reproducibility(runs=3):
    import inference

    print(f"\nReproducibility — {runs} runs per image must be identical")
    if not inference.engine_available():
        skip("reproducibility", "no model available")
        return

    for path in sample_images():
        results = [inference.analyze_file(path, "image") for _ in range(runs)]
        pfakes = [r["ensemble"][0]["pFake"] for r in results]
        verdicts = {(r["prediction"], r["confidence"]) for r in results}
        regions = {(r.get("explain") or {}).get("focusRegion") for r in results}
        spread = max(pfakes) - min(pfakes)

        name = os.path.basename(path)[:22]
        check(f"{name}: probability stable", spread == 0,
              f"spread {spread:.2e}  values {pfakes}")
        check(f"{name}: verdict stable", len(verdicts) == 1, str(verdicts))
        check(f"{name}: heatmap region stable", len(regions) == 1, str(regions))

    # A fresh engine must agree with the warm one — no state accumulating
    path = sample_images()[0]
    warm = inference.analyze_file(path, "image")["ensemble"][0]["pFake"]
    inference._engine = None
    inference._engine_mtime = None
    cold = inference.analyze_file(path, "image")["ensemble"][0]["pFake"]
    check("a reloaded engine gives the same answer", warm == cold,
          f"warm={warm} cold={cold}")


# --------------------------------------------------------------------- parity

def test_parity():
    print(f"\nParity — ONNX vs PyTorch, max difference must be < {PARITY_THRESHOLD}")
    import inference
    from config import CFG

    if not inference.onnx_available():
        skip("parity", "no ONNX export present")
        return
    try:
        import torch
        from torchvision import models
    except ImportError:
        skip("parity", "PyTorch not installed (expected in a deployment)")
        return
    if not os.path.exists(CFG.CKPT_PATH):
        skip("parity", "no .pth checkpoint to compare against")
        return

    ck = torch.load(CFG.CKPT_PATH, map_location="cpu", weights_only=False)
    builders = {"mobilenet_v3_small": models.mobilenet_v3_small,
                "mobilenet_v3_large": models.mobilenet_v3_large}
    net = builders[ck["arch"]](weights=None)
    net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features, len(ck["classes"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()

    engine = inference._get_engine()
    from PIL import Image

    worst = 0.0
    for path in sample_images():
        with Image.open(path) as im:
            face, _ = engine._face_crop_ex(im)
        prepared = engine._normalize_compression(face.convert("RGB"))
        batch = engine.np.stack([engine._to_input(prepared)])

        onnx_probs = engine._forward(batch)[0]
        with torch.no_grad():
            torch_probs = torch.softmax(net(torch.from_numpy(batch)), dim=1)[0].numpy()

        diff = float(abs(onnx_probs - torch_probs).max())
        worst = max(worst, diff)
        check(f"{os.path.basename(path)[:22]}: agrees", diff < PARITY_THRESHOLD,
              f"max diff {diff:.2e}  (onnx {onnx_probs[0]:.6f} vs torch {torch_probs[0]:.6f})")

    check("worst-case difference within threshold", worst < PARITY_THRESHOLD,
          f"{worst:.2e} < {PARITY_THRESHOLD}")


def main():
    print("DeepShield model tests")
    test_identity()
    test_reproducibility()
    test_parity()

    total = len(PASS) + len(FAIL)
    print("\n" + "=" * 52)
    print(f"passed {len(PASS)} / {total}" + (f"   ({len(SKIP)} skipped)" if SKIP else ""))
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print("  - " + f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
