import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module


def test_analyze_preserves_explicit_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(app_module, "_housekeeping_started", True)
    monkeypatch.setattr(app_module.inference, "engine_available", lambda: True)
    monkeypatch.setattr(
        app_module,
        "_read_request",
        lambda live: ("landscape.jpg", "image", 123, 1.0, "unused.jpg", False),
    )
    monkeypatch.setattr(
        app_module,
        "_run_inference",
        lambda *args: {
            "prediction": "real",
            "confidence": 50,
            "framesAnalyzed": 1,
            "faceFound": False,
            "facesFound": 0,
            "insufficientEvidence": True,
            "reason": "No face detected; no reliable verdict.",
        },
    )
    monkeypatch.setattr(
        app_module,
        "model_identity",
        lambda *args, **kwargs: {"name": "MobileNetV3", "device": "CPU"},
    )
    monkeypatch.setattr(app_module.store, "record_analysis", lambda **kwargs: None)
    app_module.limiter._hits.clear()

    response = app_module.app.test_client().post(
        "/api/analyze", json={"scanId": "SCAN-NO-FACE"}
    )

    try:
        assert response.status_code == 200
        body = response.get_json()
        assert body["prediction"] == "real"  # compatibility value only
        assert body["confidence"] == 50
        assert body["faceFound"] is False
        assert body["facesFound"] == 0
        assert body["insufficientEvidence"] is True
        assert body["reason"] == "No face detected; no reliable verdict."
    finally:
        app_module.limiter._hits.clear()
