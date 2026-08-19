"""Full-frame AI-generated media detector.

This is deliberately separate from the face-manipulation model in
``inference.py``.  DeepShield V3 was trained on face crops, so a synthetic
image can contain useful generation artefacts outside the largest face.  The
origin detector looks at the complete frame and provides a second, independent
signal.

The model is never downloaded at request time.  ``scripts/fetch_origin_model.py``
installs a pinned, hash-verified ONNX artefact at build/setup time.  If the
artefact or onnxruntime is missing, this module simply reports itself
unavailable and the existing DeepShield engine keeps working.

The returned number is a model score, *not* a calibrated probability.
"""
from __future__ import annotations

import os
import statistics
import threading
from typing import Any

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "ai_origin_int8.onnx",
)
MODEL_NAME = "AI Image Detect Distilled (INT8)"
MODEL_SOURCE = "onnx-community/ai-image-detect-distilled-ONNX"
MODEL_REVISION = "7f067e23521eeb6d6525221af82c613fb746aaff"
MODEL_SHA256 = "7273cb9cd81e17eae04771010d2199ba6ae34ea2a75a275518c0bc4a2c26ffd2"

# Conservative defaults until DeepShield has its own held-out calibration set.
TRIGGER_SCORE = float(os.environ.get("DS_AI_ORIGIN_THRESHOLD", "0.85"))
VIDEO_FRAMES = max(1, int(os.environ.get("DS_AI_ORIGIN_FRAMES", "4")))

_session = None
_session_stamp = None
_lock = threading.Lock()


def _runtime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def available() -> bool:
    return os.path.exists(MODEL_PATH) and _runtime_available()


def info() -> dict[str, Any]:
    return {
        "available": available(),
        "model": MODEL_NAME,
        "source": MODEL_SOURCE,
        "revision": MODEL_REVISION,
        "threshold": TRIGGER_SCORE,
        "calibrated": False,
    }


def _get_session():
    global _session, _session_stamp
    if not available():
        return None

    import onnxruntime as ort

    stamp = (MODEL_PATH, os.path.getmtime(MODEL_PATH), os.path.getsize(MODEL_PATH))
    if _session is None or _session_stamp != stamp:
        opts = ort.SessionOptions()
        # The hosting target is small and CPU constrained.  Do not let this
        # auxiliary model create a second large thread pool beside OpenCV.
        opts.intra_op_num_threads = max(1, int(os.environ.get("DS_AI_ORIGIN_THREADS", "1")))
        opts.inter_op_num_threads = 1
        _session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        _session_stamp = stamp
    return _session


def _to_input(image):
    """Match the published ViT preprocessor: RGB, 224², /255, mean/std=.5."""
    import numpy as np
    from PIL import Image

    image = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)


def _softmax(logits):
    import numpy as np

    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 1:
        logits = logits[None, :]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def score_pil(image) -> float | None:
    """Return the model's AI/fake score for a full PIL image, or None."""
    session = _get_session()
    if session is None:
        return None

    x = _to_input(image)
    input_name = session.get_inputs()[0].name
    with _lock:
        output = session.run(None, {input_name: x})[0]
    probs = _softmax(output)
    if probs.shape[-1] != 2:
        raise ValueError(f"origin detector expected 2 classes, got {probs.shape}")
    # Published config: class 0 = fake, class 1 = real.
    return float(probs[0, 0])


def score_image(path: str) -> float | None:
    from PIL import Image

    if not available():
        return None
    with Image.open(path) as image:
        return score_pil(image)


def score_video(path: str, max_frames: int | None = None) -> dict[str, Any] | None:
    """Score a small, evenly-spaced set of complete video frames.

    This is intentionally a second pass.  The face detector samples according
    to the user's frame-rate setting; the origin detector is much heavier, so
    it examines only a few frames spread across the clip.
    """
    if not available():
        return None

    import cv2
    from PIL import Image

    wanted = max(1, int(max_frames or VIDEO_FRAMES))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None

    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count > 1:
            if wanted == 1:
                indices = [count // 2]
            else:
                indices = sorted({round(i * (count - 1) / (wanted - 1)) for i in range(wanted)})
        else:
            indices = list(range(wanted))

        scores = []
        sampled = []
        for idx in indices:
            if count > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                if count <= 1:
                    break
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            score = score_pil(Image.fromarray(rgb))
            if score is not None:
                scores.append(float(score))
                sampled.append(int(idx))
            if count <= 1 and len(scores) >= wanted:
                break

        if not scores:
            return None
        return {
            "score": float(statistics.median(scores)),
            "mean": float(sum(scores) / len(scores)),
            "peak": float(max(scores)),
            "lowest": float(min(scores)),
            "frames": len(scores),
            "frameIndices": sampled,
            "threshold": TRIGGER_SCORE,
            "calibrated": False,
        }
    finally:
        cap.release()
