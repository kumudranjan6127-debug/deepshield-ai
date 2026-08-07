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
YUNET_PATH = os.path.join(ROOT, "models", "face_detection_yunet.onnx")

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

        self._detector = None  # lazy YuNet face detector (OpenCV 5 DNN)

    # ---- face crop: align inference with the training domain ----
    # The dataset is tight face portraits; feeding whole photos
    # (background, clothes, scenery) biases the model toward "real".
    def _face_crop(self, pil_image):
        import cv2
        import numpy as np
        from PIL import Image

        if not os.path.exists(YUNET_PATH):
            return pil_image  # detector model missing → analyze full frame

        if self._detector is None:
            self._detector = cv2.FaceDetectorYN_create(
                YUNET_PATH, "", (320, 320), 0.6, 0.3, 5000)

        rgb = np.array(pil_image.convert("RGB"))
        h, w = rgb.shape[:2]

        # Detect on a downscaled copy for speed; map coords back
        scale = 1.0
        det_img = rgb
        if max(h, w) > 1024:
            scale = 1024.0 / max(h, w)
            det_img = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        bgr = cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return pil_image  # no face found → analyze the full frame

        # Largest detected face = the main subject
        best = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
        x, y, fw, fh = [v / scale for v in best[:4]]

        m = 0.35 * max(fw, fh)  # margin, matches portrait-style crops
        x0 = max(0, int(x - m));  y0 = max(0, int(y - m))
        x1 = min(w, int(x + fw + m));  y1 = min(h, int(y + fh + m))
        if x1 <= x0 or y1 <= y0:
            return pil_image
        return Image.fromarray(rgb[y0:y1, x0:x1])

    # ---- single image (PIL) → probability vector over classes
    def _probs(self, pil_image):
        x = self.tf(self._face_crop(pil_image).convert("RGB")).unsqueeze(0)
        with self.torch.no_grad():
            return self.torch.softmax(self.model(x), dim=1)[0]

    def predict_image(self, image_path):
        """→ (prediction 'real'|'deepfake', confidence int, frames=1)"""
        from PIL import Image
        # Context manager releases the file handle — without it Windows
        # blocks the post-analysis delete of the uploaded file.
        with Image.open(image_path) as im:
            probs = self._probs(im)
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


_engine_mtime = None


def _get_engine() -> _Engine:
    """Singleton, but reloads automatically if the checkpoint file is
    replaced (e.g. dropping in a newly trained model — no restart)."""
    global _engine, _engine_mtime
    mtime = os.path.getmtime(CKPT_PATH)
    if _engine is None or _engine_mtime != mtime:
        _engine = _Engine()
        _engine_mtime = mtime
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
