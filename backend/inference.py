"""Unified DeepShield inference policy.

The original V3 face-manipulation engine lives in ``face_inference.py``.
This module keeps that detector intact and adds an optional full-frame
AI-origin detector so fully synthetic images/videos are not forced through a
face-only classifier.

The two signals are intentionally *not* averaged.  Each detector has a
separate job:
- face engine: evidence of manipulation/synthesis in detected faces;
- origin engine: evidence that the complete frame is AI-generated.

A strong signal from either detector is enough to flag the media.  A weak or
conflicting auxiliary score lowers certainty instead of inventing confidence.
The origin detector is uncalibrated, so its threshold is deliberately
conservative and every response exposes the individual model scores.
"""
from __future__ import annotations

import face_inference as _face
import origin_detector as _origin
from config import CFG

# Preserve the public surface used by app.py, scripts and the existing tests.
risk_for = _face.risk_for
certainty_for = _face.certainty_for
certainty_bands = _face.certainty_bands
aggregate_frames = _face.aggregate_frames
torch_available = _face.torch_available
onnx_available = _face.onnx_available
engine_available = _face.engine_available
version_from = _face.version_from
_Engine = _face._Engine
_get_engine = _face._get_engine


def __getattr__(name):
    """Backwards-compatible access to implementation details/tests."""
    return getattr(_face, name)


def engine_info() -> dict:
    info = dict(_face.engine_info())
    if info:
        info["origin_detector"] = _origin.info()
    return info


def _p_fake(prediction: str, confidence: int) -> float:
    c = min(1.0, max(0.0, float(confidence) / 100.0))
    return c if prediction == "deepfake" else 1.0 - c


def fuse_evidence(base: dict, origin_score: float | None,
                  threshold: float | None = None) -> dict:
    """Fuse face-manipulation and full-frame AI-origin evidence.

    This function is pure on purpose: policy tests can exercise the decision
    boundary without loading either model.
    """
    out = dict(base)
    threshold = float(_origin.TRIGGER_SCORE if threshold is None else threshold)
    threshold = min(0.999, max(0.501, threshold))
    strong_real = 1.0 - threshold

    face_found = bool(base.get("faceFound"))
    face_score = (_p_fake(base.get("prediction", "real"), base.get("confidence", 50))
                  if face_found else None)

    votes = []
    if face_score is not None:
        votes.append({
            "model": "DeepShield face manipulation",
            "pFake": round(face_score, 4),
            "kind": "face-manipulation",
        })
    if origin_score is not None:
        origin_score = min(1.0, max(0.0, float(origin_score)))
        votes.append({
            "model": _origin.MODEL_NAME,
            "pFake": round(origin_score, 4),
            "kind": "ai-origin/full-frame",
            "note": "uncalibrated full-frame AI-generation score",
        })

    if votes:
        out["ensemble"] = votes

    # Existing face detector has strong fake evidence: never dilute it with a
    # detector trained for a different task.
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

    # Full-frame detector sees strong AI-generation evidence.  This is the
    # failure mode the face-only V3 path could not catch.
    if origin_score is not None and origin_score >= threshold:
        out["prediction"] = "deepfake"
        out["confidence"] = int(round(origin_score * 100))
        out["findingType"] = "ai_generated"
        out["insufficientEvidence"] = False
        out.pop("reason", None)
        out["disputed"] = bool(face_score is not None and face_score < 0.5)
        return out

    # No face: the origin detector can still provide a real verdict when its
    # evidence is strongly on the real side.  Otherwise keep the honest 50%
    # no-answer semantics rather than calling a landscape "real" by default.
    if not face_found:
        if origin_score is not None and origin_score <= strong_real:
            out["prediction"] = "real"
            out["confidence"] = int(round((1.0 - origin_score) * 100))
            out["findingType"] = "real_media"
            out["insufficientEvidence"] = False
            out.pop("reason", None)
        else:
            out["prediction"] = "real"
            out["confidence"] = 50
            out["findingType"] = "inconclusive"
            out["insufficientEvidence"] = True
            out["reason"] = (
                "No reliable face evidence and the full-frame AI-origin "
                "detector did not reach a strong decision."
            )
        return out

    # Face says real.  A mildly conflicting origin score is not enough to
    # accuse the media, but it should stop us displaying a high-confidence
    # real verdict.
    out["findingType"] = "real_media"
    if origin_score is None:
        return out
    if origin_score > 0.5:
        out["confidence"] = 50
        out["insufficientEvidence"] = True
        out["reason"] = "Face and full-frame detectors disagree; result is inconclusive."
        out["disputed"] = True
    else:
        origin_real_conf = int(round((1.0 - origin_score) * 100))
        out["confidence"] = min(int(base.get("confidence", 50)), origin_real_conf)
        out["disputed"] = False
    return out


def _video_face_result(path: str, frame_rate: float) -> dict:
    """Run V3 video scoring without letting no-face frames dilute evidence."""
    eng = _face._get_engine()
    records, meta = eng.predict_video(path, frame_rate)
    face_records = [r for r in records if r.get("facesFound", 0) > 0]
    any_face = bool(face_records)

    if any_face:
        agg = _face.aggregate_frames([r["pFake"] for r in face_records])
        score = float(agg["score"])
        prediction = "deepfake" if score > 0.5 else "real"
        confidence = int(round(max(score, 1.0 - score) * 100))
        hottest = sorted(face_records, key=lambda r: r["pFake"], reverse=True)
    else:
        # Shape-compatible neutral summary for the UI.  Crucially these 0.5
        # placeholders are not mixed with real face scores anymore.
        agg = _face.aggregate_frames([0.5])
        agg["suspicious"] = 0
        score = 0.5
        prediction = "real"
        confidence = 50
        hottest = []

    return {
        "prediction": prediction,
        "confidence": confidence,
        "framesAnalyzed": len(records),
        "faceFound": any_face,
        "facesFound": max((r.get("facesFound", 0) for r in records), default=0),
        "insufficientEvidence": not any_face,
        "reason": ("No faces were detected in the sampled frames; face evidence is inconclusive."
                   if not any_face else None),
        "video": {
            "framesAnalyzed": len(records),
            "faceFramesAnalyzed": len(face_records),
            "noFaceFrames": len(records) - len(face_records),
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
                    "timestamp": f"{int(r['time'])//60:02d}:{int(r['time'])%60:02d}",
                    "score": round(r["pFake"], 4),
                }
                for r in hottest[:CFG.VIDEO_TOP_TIMESTAMPS]
            ],
            "timeline": [
                {
                    "t": round(r["time"], 2),
                    "p": (round(r["pFake"], 4) if r.get("facesFound", 0) > 0 else None),
                    "facesFound": int(r.get("facesFound", 0)),
                }
                for r in records
            ],
            "temporal": {
                "facesFound": len(face_records),
                "framesSampled": len(records),
            },
            "fps": round(meta["fps"], 2),
            "sampledEveryNthFrame": meta["step"],
            "durationSeconds": round(meta["duration"], 2),
        },
    }


def analyze_file(path, file_type, frame_rate=CFG.DEFAULT_FRAME_RATE):
    if file_type == "video":
        base = _video_face_result(path, frame_rate)
        origin = _origin.score_video(path)
        origin_score = origin.get("score") if origin else None
        result = fuse_evidence(base, origin_score)
        if origin:
            result["video"]["originDetector"] = origin
        return result

    base = _face.analyze_file(path, file_type, frame_rate)
    origin_score = _origin.score_image(path)
    return fuse_evidence(base, origin_score)


def score_image(path) -> float:
    """Combined product score used by evaluation/benchmark tooling."""
    result = analyze_file(path, "image")
    if result.get("insufficientEvidence"):
        return 0.5
    return _p_fake(result.get("prediction", "real"), result.get("confidence", 50))
