"""DeepShield inference facade.

The large, measured inference implementation that shipped with multi-face
support is preserved in ``inference_legacy``.  A later hardening change
accidentally replaced most of it while adding three useful policies:

- no-face media is inconclusive rather than whole-frame classified;
- every detected face is considered, including per video frame;
- video aggregation validates/clamps caller-supplied tuning values.

This module keeps those policies while re-exposing the proven preprocessing,
explainability, optional verifier and temporal-diagnostic implementation.
Keeping the policy layer small makes future hardening changes less likely to
delete unrelated model behavior again.
"""
from __future__ import annotations

import os
import statistics
from typing import Iterable

import inference_legacy as _legacy
from config import CFG

BASE_DIR = _legacy.BASE_DIR
CKPT_PATH = _legacy.CKPT_PATH
ONNX_PATH = _legacy.ONNX_PATH
ONNX_META_PATH = _legacy.ONNX_META_PATH
YUNET_PATH = _legacy.YUNET_PATH
ARCH_NAMES = _legacy.ARCH_NAMES
ARCH_PARAMS = _legacy.ARCH_PARAMS
HF_MODELS = _legacy.HF_MODELS
_HFEngine = _legacy._HFEngine

_engine = None
_engine_mtime = None


def risk_for(prediction: str, confidence: int) -> str:
    if prediction == "deepfake":
        return "High" if confidence >= 85 else "Medium"
    if prediction == "real":
        return "Low" if confidence >= 80 else "Medium"
    return "Medium"


def certainty_for(confidence: int) -> str:
    for lower, key, _ in CFG.CERTAINTY_BANDS:
        if confidence >= lower:
            return key
    return CFG.CERTAINTY_BANDS[-1][1]


def certainty_bands() -> list:
    """Publish the same contiguous half-open ranges used by certainty_for."""
    ordered = sorted(CFG.CERTAINTY_BANDS, key=lambda x: x[0], reverse=True)
    return [
        {
            "from": lower,
            "to": ordered[i - 1][0] if i else 100,
            "key": key,
            "label": label,
        }
        for i, (lower, key, label) in enumerate(ordered)
    ]


def aggregate_frames(
    p_fakes: Iterable[float], weights=None, topk_fraction=None,
    suspicious_at=None,
) -> dict:
    """Robust video aggregation with validated, normalized weights."""
    ps = [min(1.0, max(0.0, float(p))) for p in p_fakes]
    if not ps:
        raise ValueError("no frame scores to aggregate")

    allowed = ("median", "mean", "top_k")
    raw = dict(weights or CFG.VIDEO_WEIGHTS)
    clean = {k: max(0.0, float(raw.get(k, 0.0))) for k in allowed}
    total = sum(clean.values())
    if total <= 0:
        clean = {"median": 1.0, "mean": 0.0, "top_k": 0.0}
        total = 1.0
    clean = {k: v / total for k, v in clean.items()}

    fraction = CFG.VIDEO_TOPK_FRACTION if topk_fraction is None else float(topk_fraction)
    fraction = min(1.0, max(1.0 / len(ps), fraction))
    k = max(1, min(len(ps), round(fraction * len(ps))))
    top = sorted(ps, reverse=True)[:k]

    parts = {
        "median": statistics.median(ps),
        "mean": sum(ps) / len(ps),
        "top_k": sum(top) / len(top),
    }
    score = sum(parts[name] * clean[name] for name in allowed)

    threshold = (
        CFG.VIDEO_SUSPICIOUS_AT if suspicious_at is None
        else min(1.0, max(0.0, float(suspicious_at)))
    )
    return {
        "score": score,
        "components": {k: round(v, 4) for k, v in parts.items()},
        "weights": {k: round(v, 4) for k, v in clean.items()},
        "k": k,
        "frames": len(ps),
        "suspicious": sum(p >= threshold for p in ps),
        "suspiciousAt": threshold,
        "peak": max(ps),
        "lowest": min(ps),
        "variance": statistics.pvariance(ps) if len(ps) > 1 else 0.0,
    }


timestamp = _legacy.timestamp
temporal_signals = _legacy.temporal_signals
version_from = _legacy.version_from
torch_available = _legacy.torch_available
onnx_available = _legacy.onnx_available
verifiers_enabled = _legacy.verifiers_enabled


def engine_available() -> bool:
    if CFG.FORCE_ECHO:
        return False
    return onnx_available() or (os.path.exists(CKPT_PATH) and torch_available())


def _face_summary(face: dict) -> dict:
    """Serializable geometry from a detector record; never expose PIL crops."""
    return {
        key: face.get(key)
        for key in ("box", "origin", "frame", "landmarks")
    }


class _Engine(_legacy._Engine):
    """Known-good engine plus hardened no-face/multi-face video sampling."""

    def __init__(self):
        super().__init__()
        self.info["calibrated"] = bool(self.meta.get("calibrated", False))

    def predict_image(self, path):
        """Return a neutral compatibility value when no face is detectable."""
        from PIL import Image

        with Image.open(path) as image:
            detected = self._detect_faces(image)
            found = [d for d in detected if d["found"]]
            if not found:
                return {
                    "prediction": "real",
                    "confidence": 50,
                    "faceFound": False,
                    "facesFound": 0,
                    "framesAnalyzed": 1,
                    "insufficientEvidence": True,
                    "reason": (
                        "No face detected; the face-trained model cannot "
                        "provide a reliable verdict."
                    ),
                }

            fake_i = self.classes.index("fake")
            scored = [(self._probs_raw(d["crop"]), d) for d in found]
            probs, selected = max(scored, key=lambda pair: float(pair[0][fake_i]))
            prediction, confidence = self._verdict(probs)
            return {
                "prediction": prediction,
                "confidence": confidence,
                "faceFound": True,
                "facesFound": len(found),
                "framesAnalyzed": 1,
                "selectedFace": _face_summary(selected),
            }

    def predict_video(
        self, path, frame_rate=CFG.DEFAULT_FRAME_RATE,
        max_frames=CFG.MAX_VIDEO_FRAMES,
    ):
        """Score the most suspicious detected face per sampled frame.

        No-face frames contribute neutral 0.5 evidence.  Geometry, landmarks
        and a tiny thumbnail from the deciding face are retained solely for
        descriptive temporal diagnostics; they never alter the verdict.
        """
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps == fps and fps > 0 else 25.0
        step = max(1, round(fps / max(0.25, float(frame_rate))))
        fake_i = self.classes.index("fake")
        records, idx = [], 0

        try:
            while len(records) < max_frames:
                if idx % step:
                    if not cap.grab():
                        break
                    idx += 1
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected = self._detect_faces(Image.fromarray(rgb))
                faces = [d for d in detected if d["found"]]

                selected = None
                if not faces:
                    p_fake = 0.5
                else:
                    scored = [
                        (float(self._probs_raw(d["crop"])[fake_i]), d)
                        for d in faces
                    ]
                    p_fake, selected = max(scored, key=lambda pair: pair[0])

                if selected is not None:
                    grey = cv2.cvtColor(
                        self.np.asarray(selected["crop"].convert("RGB")),
                        cv2.COLOR_RGB2GRAY,
                    )
                    thumb = cv2.resize(
                        grey, (32, 32), interpolation=cv2.INTER_AREA
                    )
                    box = selected["box"]
                    origin = selected["origin"]
                    frame_size = selected["frame"]
                    landmarks = selected["landmarks"]
                else:
                    miss = detected[0] if detected else None
                    thumb = None
                    box = None
                    origin = (0, 0)
                    frame_size = miss["frame"] if miss else (rgb.shape[1], rgb.shape[0])
                    landmarks = None

                records.append({
                    "index": idx,
                    "time": idx / fps,
                    "pFake": float(p_fake),
                    "facesFound": len(faces),
                    "box": box,
                    "origin": origin,
                    "frame": frame_size,
                    "landmarks": landmarks,
                    "thumb": thumb,
                })
                idx += 1
        finally:
            cap.release()

        if not records:
            raise ValueError("No readable frames in video")
        return records, {
            "fps": float(fps),
            "step": int(step),
            "duration": idx / fps,
        }


_HFEngine = _legacy._HFEngine


def _get_hf_engines():
    """Compatibility hook kept local so tests/callers can monkeypatch it."""
    return _legacy._get_hf_engines()


def _active_model_path():
    return ONNX_PATH if onnx_available() else CKPT_PATH


def _get_engine():
    global _engine, _engine_mtime
    path = _active_model_path()
    try:
        stamp = (path, os.path.getmtime(path))
    except OSError:
        if _engine is not None:
            return _engine
        raise
    if _engine is None or _engine_mtime != stamp:
        _engine = _Engine()
        _engine_mtime = stamp
    return _engine


def engine_info() -> dict:
    if not engine_available():
        return {}
    return {**_get_engine().info, "verifiers": verifiers_enabled()}


def score_image(path) -> float:
    """P(fake) through the same face selection/preprocessing as serving."""
    from PIL import Image

    eng = _get_engine()
    with Image.open(path) as image:
        detected = eng._detect_faces(image)
        faces = [d for d in detected if d["found"]]
        if not faces:
            return 0.5
        fake_i = eng.classes.index("fake")
        return max(float(eng._probs_raw(d["crop"])[fake_i]) for d in faces)


def _combine_image_votes(votes):
    own_weight = CFG.OWN_WEIGHT
    if len(votes) > 1:
        verifier_weight = (1.0 - own_weight) / (len(votes) - 1)
        weights = [own_weight] + [verifier_weight] * (len(votes) - 1)
    else:
        weights = [1.0]

    for vote, weight in zip(votes, weights):
        vote["weight"] = round(weight, 3)

    p_fake = sum(v["pFake"] * v["weight"] for v in votes)
    verifiers = [v["pFake"] for v in votes[1:]]
    if verifiers and all(v >= CFG.VERIFIER_OVERRULE_AT for v in verifiers):
        p_fake = max(p_fake, sum(verifiers) / len(verifiers))
    return min(1.0, max(0.0, p_fake))


def analyze_file(path, file_type, frame_rate=CFG.DEFAULT_FRAME_RATE):
    """Main serving entry with restored observable inference contracts."""
    eng = _get_engine()

    if file_type == "video":
        records, meta = eng.predict_video(path, frame_rate)
        agg = aggregate_frames([r["pFake"] for r in records])
        any_face = any(r["facesFound"] > 0 for r in records)
        score = float(agg["score"])
        if not any_face:
            score = 0.5
            prediction = "real"
            confidence = 50
        else:
            prediction = "deepfake" if score >= 0.5 else "real"
            confidence = int(round(max(score, 1.0 - score) * 100))
        hottest = sorted(records, key=lambda r: r["pFake"], reverse=True)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "framesAnalyzed": len(records),
            "faceFound": any_face,
            "facesFound": max((r["facesFound"] for r in records), default=0),
            "insufficientEvidence": not any_face,
            "reason": (
                "No faces were detected in the sampled frames; result is "
                "inconclusive."
                if not any_face else None
            ),
            "video": {
                "framesAnalyzed": len(records),
                "suspiciousFrames": agg["suspicious"],
                "suspiciousAt": agg["suspiciousAt"],
                "peakFakeScore": round(agg["peak"], 4),
                "medianFakeScore": round(agg["components"]["median"], 4),
                "meanFakeScore": round(agg["components"]["mean"], 4),
                "topKFakeScore": round(agg["components"]["top_k"], 4),
                "lowestFakeScore": round(agg["lowest"], 4),
                "scoreVariance": round(agg["variance"], 6),
                "combinedScore": round(score, 4),
                "k": agg["k"],
                "weights": agg["weights"],
                "topTimestamps": [
                    {
                        "time": round(r["time"], 2),
                        "timestamp": timestamp(r["time"]),
                        "score": round(r["pFake"], 4),
                    }
                    for r in hottest[:CFG.VIDEO_TOP_TIMESTAMPS]
                ],
                "timeline": [
                    {"t": round(r["time"], 2), "p": round(r["pFake"], 4)}
                    for r in records
                ],
                "temporal": temporal_signals(records),
                "fps": round(meta["fps"], 2),
                "sampledEveryNthFrame": meta["step"],
                "durationSeconds": round(meta["duration"], 2),
            },
            "ensemble": [{
                "model": "MobileNetV3 (ours)",
                "pFake": round(score, 4),
                "weight": 1.0,
                "note": "video — median / mean / top-k over frames",
            }],
        }

    from PIL import Image

    with Image.open(path) as image:
        detected = eng._detect_faces(image)
        faces = [d for d in detected if d["found"]]
        if not faces:
            return {
                "prediction": "real",
                "confidence": 50,
                "framesAnalyzed": 1,
                "faceFound": False,
                "facesFound": 0,
                "insufficientEvidence": True,
                "reason": (
                    "No face detected; the face-trained model cannot provide "
                    "a reliable verdict."
                ),
            }

        fake_i = eng.classes.index("fake")
        scored = [(eng._probs_raw(d["crop"]), d) for d in faces]
        probs, selected = max(scored, key=lambda pair: float(pair[0][fake_i]))
        face = selected["crop"]
        landmarks = selected["landmarks"]

        votes = [{
            "model": "MobileNetV3 (ours)",
            "pFake": round(float(probs[fake_i]), 4),
        }]
        for verifier in _get_hf_engines():
            try:
                votes.append({
                    "model": verifier.name,
                    "pFake": round(verifier.p_fake(face), 4),
                })
            except Exception:
                pass

        try:
            explain = eng.explain(face, landmarks)
        except Exception:
            explain = None

    p_fake = _combine_image_votes(votes)
    prediction = "deepfake" if p_fake >= 0.5 else "real"
    confidence = int(round(max(p_fake, 1.0 - p_fake) * 100))
    disputed = any((v["pFake"] >= 0.5) != (p_fake >= 0.5) for v in votes)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "framesAnalyzed": 1,
        "faceFound": True,
        "facesFound": len(faces),
        "ensemble": votes,
        "disputed": disputed,
        "combiner": "own-led + verifier consensus",
        "explain": explain,
        "selectedFace": _face_summary(selected),
    }