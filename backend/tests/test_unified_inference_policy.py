"""Policy tests for face + full-frame synthetic-media fusion.

No model files or network access are required here; these pin the decision
rules so future refactors cannot silently turn weak evidence into an accusation.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import inference


def face_result(prediction="real", confidence=90, found=True):
    return {
        "prediction": prediction,
        "confidence": confidence,
        "faceFound": found,
        "facesFound": 1 if found else 0,
        "framesAnalyzed": 1,
        "insufficientEvidence": not found,
    }


def test_strong_ai_origin_can_flag_face_that_looks_real():
    out = inference.fuse_evidence(face_result("real", 96), 0.93, threshold=0.85)
    assert out["prediction"] == "deepfake"
    assert out["findingType"] == "ai_generated"
    assert out["confidence"] == 93


def test_face_manipulation_is_not_diluted_by_origin_detector():
    out = inference.fuse_evidence(face_result("deepfake", 91), 0.12, threshold=0.85)
    assert out["prediction"] == "deepfake"
    assert out["confidence"] == 91
    assert out["findingType"] == "face_manipulation"


def test_weak_conflict_does_not_accuse_real_media():
    out = inference.fuse_evidence(face_result("real", 97), 0.67, threshold=0.85)
    assert out["prediction"] == "real"
    assert out["confidence"] == 50
    assert out["insufficientEvidence"] is True
    assert out["disputed"] is True


def test_no_face_plus_strong_ai_origin_is_still_detectable():
    out = inference.fuse_evidence(face_result("real", 50, found=False), 0.94, threshold=0.85)
    assert out["prediction"] == "deepfake"
    assert out["findingType"] == "ai_generated"
    assert out["insufficientEvidence"] is False


def test_no_face_plus_weak_origin_remains_inconclusive():
    out = inference.fuse_evidence(face_result("real", 50, found=False), 0.52, threshold=0.85)
    assert out["prediction"] == "real"  # transport value kept for old clients
    assert out["confidence"] == 50
    assert out["findingType"] == "inconclusive"
    assert out["insufficientEvidence"] is True


def test_no_face_plus_strong_real_origin_can_return_real():
    out = inference.fuse_evidence(face_result("real", 50, found=False), 0.08, threshold=0.85)
    assert out["prediction"] == "real"
    assert out["confidence"] == 92
    assert out["findingType"] == "real_media"
    assert out["insufficientEvidence"] is False


def test_origin_detector_has_pinned_identity():
    import origin_detector
    assert len(origin_detector.MODEL_REVISION) == 40
    assert len(origin_detector.MODEL_SHA256) == 64
    assert origin_detector.MODEL_SOURCE == "onnx-community/ai-image-detect-distilled-ONNX"
