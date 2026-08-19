"""Unified DeepShield synthetic-media inference policy.

V3 remains the face-manipulation engine in ``face_inference.py``.  A second,
optional full-frame AI-origin model supplies independent evidence for fully
synthetic images and video frames.  The scores are never blindly averaged:
strong evidence from either detector can flag media, while weak disagreement
becomes an inconclusive result rather than a confident accusation.
"""
from __future__ import annotations

import face_inference as _face
import origin_detector as _origin
from config import CFG

# Explicit aliases keep imports/monkeypatches used by tooling and the historic
# test suite working.  __getattr__ remains as a final compatibility fallback.
risk_for = _face.risk_for
certainty_for = _face.certainty_for
certainty_bands = _face.certainty_bands
aggregate_frames = _face.aggregate_frames
timestamp = _face.timestamp
temporal_signals = _face.temporal_signals
torch_available = _face.torch_available
onnx_available = _face.onnx_available
engine_available = _face.engine_available
version_from = _face.version_from
verifiers_enabled = _face.verifiers_enabled
_Engine = _face._Engine
_get_engine = _face._get_engine
_get_hf_engines = _face._get_hf_engines


def __getattr__(name):
    return getattr(_face, name)


def engine_info() -> dict:
    info = dict(_face.engine_info())
    if info:
        info["origin_detector"] = _origin.info()
    return info


def _p_fake(prediction: str, confidence: int) -> float:
    c = min(1.0, max(0.0, float(confidence) / 100.0))
    return c if prediction == "deepfake" else 1.0 - c


def _face_score_from(base: dict) -> float | None:
    if not base.get("faceFound"):
        return None
    for vote in base.get("ensemble") or []:
        if isinstance(vote, dict) and vote.get("model", "").endswith("(ours)"):
            value = vote.get("pFake")
            if isinstance(value, (int, float)):
                return min(1.0, max(0.0, float(value)))
    return _p_fake(base.get("prediction", "real"), base.get("confidence", 50))


def _combined_votes(base: dict, origin_score: float | None) -> list:
    votes = [dict(v) for v in (base.get("ensemble") or []) if isinstance(v, dict)]
    for vote in votes:
        if vote.get("model", "").endswith("(ours)"):
            vote.setdefault("kind", "face-manipulation")
    if origin_score is not None:
        votes.append({
            "model": _origin.MODEL_NAME,
            "pFake": round(float(origin_score), 4),
            "kind": "ai-origin/full-frame",
            "note": "uncalibrated full-frame AI-generation score",
        })
    return votes


def fuse_evidence(base: dict, origin_score: float | None,
                  threshold: float | None = None) -> dict:
    """Fuse face and full-frame evidence without erasing either provenance."""
    out = dict(base)
    threshold = float(_origin.TRIGGER_SCORE if threshold is None else threshold)
    threshold = min(0.999, max(0.501, threshold))
    strong_real = 1.0 - threshold

    face_score = _face_score_from(base)
    if origin_score is not None:
        origin_score = min(1.0, max(0.0, float(origin_score)))
    votes = _combined_votes(base, origin_score)
    if votes:
        out["ensemble"] = votes

    # Strong V3 fake evidence remains authoritative for face manipulation.
    if face_score is not None and face_score >= 0.5:
        winning = face_score
        if origin_score is not None and origin_score >= threshold:
            winning = max(winning, origin_score)
            out["findingType"] = "synthetic_or_manipulated"
        else:
            out["findingType"] = "face_manipulation"
        out["prediction"] = "deepfake"
        out["confidence"] = int(round(winning * 100))
        out["insufficientEvidence"] = False
        out.pop("reason", None)
        out["disputed"] = bool(origin_score is not None and origin_score < 0.5)
        return out

    # Strong whole-frame synthetic evidence catches the failure mode a
    # face-only crop fundamentally cannot see.
    if origin_score is not None and origin_score >= threshold:
        out["prediction"] = "deepfake"
        out["confidence"] = int(round(origin_score * 100))
        out["findingType"] = "ai_generated"
        out["insufficientEvidence"] = False
        out.pop("reason", None)
        out["disputed"] = bool(face_score is not None and face_score < 0.5)
        return out

    # A face-trained model must not call a no-face image real.  The origin
    # detector can answer only when it is strongly on one side.
    if face_score is None:
        if origin_score is not None and origin_score <= strong_real:
            out["prediction"] = "real"
            out["confidence"] = int(round((1.0 - origin_score) * 100))
            out["findingType"] = "real_media"
            out["insufficientEvidence"] = False
            out.pop("reason", None)
        else:
            out["prediction"] = "real"  # transport compatibility; see flag below
            out["confidence"] = 50
            out["findingType"] = "inconclusive"
            out["insufficientEvidence"] = True
            out["reason"] = (
                "No reliable face evidence and the full-frame AI-origin "
                "detector did not reach a strong decision."
            )
        return out

    # Face evidence says real.  Mild disagreement is not enough to accuse the
    # media, but it is enough to remove the confident-real presentation.
    out["findingType"] = "real_media"
    if origin_score is None:
        return out
    if origin_score > 0.5:
        out["confidence"] = 50
        out["insufficientEvidence"] = True
        out["reason"] = "Face and full-frame detectors disagree; result is inconclusive."
        out["disputed"] = True
    else:
        out["confidence"] = min(
            int(base.get("confidence", 50)), int(round((1.0 - origin_score) * 100))
        )
        out["disputed"] = False
    return out


def analyze_file(path, file_type, frame_rate=None):
    frame_rate = CFG.DEFAULT_FRAME_RATE if frame_rate is None else frame_rate
    base = _face.analyze_file(path, file_type, frame_rate)

    if file_type == "video":
        origin = _origin.score_video(path)
        result = fuse_evidence(base, origin.get("score") if origin else None)
        if origin:
            result.setdefault("video", {})["originDetector"] = origin
        return result

    origin_score = _origin.score_image(path)
    return fuse_evidence(base, origin_score)


def score_image(path) -> float:
    """Combined fast score for evaluation; never computes a heatmap."""
    face_score = _face.score_image(path)
    if not _origin.available():
        return face_score
    origin_score = _origin.score_image(path)
    if origin_score is None:
        return face_score
    threshold = _origin.TRIGGER_SCORE
    if face_score > 0.5:
        return max(face_score, origin_score if origin_score >= threshold else face_score)
    if origin_score >= threshold:
        return origin_score
    if origin_score > 0.5:
        return 0.5
    return min(face_score, origin_score)
