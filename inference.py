"""
============================================================
DeepShield AI — real inference engine (Phase 4)

Loads models/deepshield_mobilenetv3.pth (produced by
training/DeepShield_Training_Colab.ipynb) and scores images
and videos on CPU.

Design:
- Zero hard dependency at import time — torch/cv2 are imported
  lazily, so app.py works (echo mode) even before Phase 4 deps
  are installed or the checkpoint exists.
- engine_available() is the single switch app.py checks.
============================================================
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(ROOT, "models", "deepshield_mobilenetv3.pth")

_engine = None  # lazy singleton


# ---------------------------------------------------------- availability

def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


def engine_available() -> bool:
    """True only when both the ML stack and the trained checkpoint exist."""
    return os.path.exists(CKPT_PATH) and torch_available()


def engine_info() -> dict:
    """Metadata for /api/health (accuracy comes from the checkpoint)."""
    if not engine_available():
        return {}
    return _get_engine().info


# ---------------------------------------------------------- engine

class _Engine:
    """Wraps the fine-tuned MobileNetV3-Small for CPU inference."""

    def __init__(self):
        import torch
        from torchvision import models, transforms

        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)

        self.classes = ckpt["classes"]  # ['fake', 'real'] — index order matters
        size = ckpt.get("input_size", 224)
        norm = ckpt.get("normalize", {"mean": [0.485, 0.456, 0.406],
                                      "std":  [0.229, 0.224, 0.225]})

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))  # keep the i3 responsive
        self.model = model
        self.torch = torch

        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(norm["mean"], norm["std"]),
        ])

        self.info = {
            "engine": "live",
            "checkpoint": os.path.basename(CKPT_PATH),
            "val_accuracy": ckpt.get("val_accuracy"),
            "test_accuracy": ckpt.get("test_accuracy"),
            "trained_on": ckpt.get("trained_on"),
        }

    # ---- single image (PIL) → probability vector over classes
    def _probs(self, pil_image):
        x = self.tf(pil_image.convert("RGB")).unsqueeze(0)
        with self.torch.no_grad():
            return self.torch.softmax(self.model(x), dim=1)[0]

    def predict_image(self, image_path):
        """→ (prediction 'real'|'deepfake', confidence int, frames=1)"""
        from PIL import Image
        probs = self._probs(Image.open(image_path))
        return self._verdict(probs), 1

    def predict_video(self, video_path, frame_rate=1.0, max_frames=60):
        """Sample ~frame_rate frames/sec (CPU-friendly), average the
        probabilities, return the aggregate verdict."""
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps / max(0.25, frame_rate)))  # every Nth frame

        probs_sum, frames = None, 0
        idx = 0
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                p = self._probs(Image.fromarray(rgb))
                probs_sum = p if probs_sum is None else probs_sum + p
                frames += 1
            idx += 1
        cap.release()

        if not frames:
            raise ValueError("No readable frames in video")
        return self._verdict(probs_sum / frames), frames

    # ---- probs → (prediction, confidence)
    def _verdict(self, probs):
        top = int(probs.argmax())
        label = self.classes[top]                     # 'fake' or 'real'
        prediction = "deepfake" if label == "fake" else "real"
        confidence = int(round(float(probs[top]) * 100))
        return prediction, confidence


def _get_engine() -> _Engine:
    global _engine
    if _engine is None:
        _engine = _Engine()
    return _engine


# ---------------------------------------------------------- public API

def analyze_file(path, file_type, frame_rate=1.0):
    """Main entry for app.py.
    → dict {prediction, confidence, framesAnalyzed}"""
    eng = _get_engine()
    if file_type == "video":
        (prediction, confidence), frames = eng.predict_video(path, frame_rate)
    else:
        (prediction, confidence), frames = eng.predict_image(path)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "framesAnalyzed": frames,
    }
