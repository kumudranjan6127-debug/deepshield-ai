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

# Repo layout: backend/inference.py → models/ lives at the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(BASE_DIR, "models", "deepshield_mobilenetv3.pth")
YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet.onnx")

# ---- Ensemble: pretrained HuggingFace verifiers (images only) ----
# Our MobileNetV3 is a fast GAN-face specialist; these add coverage for
# other pipelines/generators. Loaded lazily; any that fail to load are
# skipped, so the app degrades gracefully to MobileNetV3 alone.
HF_MODELS = [
    {"id": "prithivMLmods/Deep-Fake-Detector-v2-Model", "name": "ViT Deepfake v2"},
    {"id": "Ateeqq/ai-vs-human-image-detector",         "name": "SigLIP AI-image"},
]
import re as _re
_FAKE_LABEL = _re.compile(r"fake|deep|ai|artificial|synthetic|generat", _re.I)

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
    """True only when both the ML stack and the trained checkpoint exist.
    DS_ENGINE=echo forces the openly-labeled simulated mode (the UI then
    shows the yellow 'Simulated (demo)' badge — honest by design)."""
    if os.environ.get("DS_ENGINE", "").lower() == "echo":
        return False
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

        # Build whichever backbone the checkpoint was trained with
        arch = ckpt.get("arch", "mobilenet_v3_small")
        builders = {
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
        }
        if arch not in builders:
            raise ValueError(f"Unsupported arch in checkpoint: {arch}")
        model = builders[arch](weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, len(self.classes))
        self.arch = arch
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
            "arch": arch,
            "val_accuracy": ckpt.get("val_accuracy"),
            "test_accuracy": ckpt.get("test_accuracy"),
            "tpdn_accuracy": ckpt.get("tpdn_accuracy"),
            "trained_on": ckpt.get("trained_on"),
        }

        self._detector = None  # lazy YuNet face detector (OpenCV 5 DNN)

    # ---- face crop: align inference with the training domain ----
    # The dataset is tight face portraits; feeding whole photos
    # (background, clothes, scenery) biases the model toward "real".
    def _face_crop(self, pil_image):
        crop, _ = self._face_crop_ex(pil_image)
        return crop

    def _face_crop_ex(self, pil_image):
        import cv2
        import numpy as np
        from PIL import Image

        if not os.path.exists(YUNET_PATH):
            return pil_image, None  # detector missing → analyze full frame

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
            return pil_image, None  # no face found → analyze the full frame

        # Largest detected face = the main subject
        best = faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]
        x, y, fw, fh = [v / scale for v in best[:4]]

        m = 0.35 * max(fw, fh)  # margin, matches portrait-style crops
        x0 = max(0, int(x - m));  y0 = max(0, int(y - m))
        x1 = min(w, int(x + fw + m));  y1 = min(h, int(y + fh + m))
        if x1 <= x0 or y1 <= y0:
            return pil_image, None

        # YuNet landmarks (5 points) → crop coordinates, for the
        # explainability "focus region" text
        names = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
        landmarks = {}
        for i, name in enumerate(names):
            lx = best[4 + i * 2] / scale - x0
            ly = best[5 + i * 2] / scale - y0
            landmarks[name] = (float(lx), float(ly))

        return Image.fromarray(rgb[y0:y1, x0:x1]), landmarks

    # ---- probability vector over classes (input already face-cropped)
    # TTA: averages the prediction over the image + its mirror — a small
    # free accuracy/stability boost for ~2x compute (still <0.2s on CPU).
    def _probs_raw(self, pil_image):
        img = pil_image.convert("RGB")
        x = self.torch.stack([
            self.tf(img),
            self.tf(img.transpose(0)),  # PIL FLIP_LEFT_RIGHT
        ])
        with self.torch.no_grad():
            return self.torch.softmax(self.model(x), dim=1).mean(dim=0)

    # ---- crop + classify (used by the video path per-frame)
    def _probs(self, pil_image):
        return self._probs_raw(self._face_crop(pil_image))

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

    # ---- Explainability: Grad-CAM on the last conv block ----
    # Shows WHERE the model looked (its real internal attention) —
    # we never fabricate claims about WHAT is wrong.
    def explain(self, face_img, landmarks):
        import cv2
        import numpy as np
        import base64

        torch = self.torch
        x = self.tf(face_img.convert("RGB")).unsqueeze(0)

        captured = {}
        layer = self.model.features[-1]
        fh = layer.register_forward_hook(
            lambda m, i, o: captured.__setitem__("act", o))
        bh = layer.register_full_backward_hook(
            lambda m, gi, go: captured.__setitem__("grad", go[0]))
        try:
            self.model.zero_grad()
            out = self.model(x)  # gradients ON for this single pass
            cls = int(out.argmax(1))
            out[0, cls].backward()
            act = captured["act"][0]          # C×h×w
            grad = captured["grad"][0]
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * act).sum(0)).detach().numpy()
        finally:
            fh.remove()
            bh.remove()
            self.model.zero_grad()

        if cam.max() > 0:
            cam = cam / cam.max()

        # Hotspot in face-crop pixel coordinates
        iy, ix = np.unravel_index(int(np.argmax(cam)), cam.shape)
        fw, fh_ = face_img.size
        hot_x = (ix + 0.5) / cam.shape[1] * fw
        hot_y = (iy + 0.5) / cam.shape[0] * fh_

        # Focus-region text, grounded in YuNet landmarks (no fabrication)
        region = "the central face region"
        if landmarks:
            def dist(p):
                return ((p[0] - hot_x) ** 2 + (p[1] - hot_y) ** 2) ** 0.5
            nearest = min(landmarks.items(), key=lambda kv: dist(kv[1]))[0]
            region = {
                "right_eye": "the eye region", "left_eye": "the eye region",
                "nose": "the nose area",
                "mouth_right": "the mouth area", "mouth_left": "the mouth area",
            }[nearest]

        # Heatmap overlay image (JPEG data URL)
        base = cv2.resize(np.array(face_img.convert("RGB")), (224, 224))
        heat = cv2.resize((cam * 255).astype(np.uint8), (224, 224))
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.55,
                                  heat, 0.45, 0)
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        data_url = ("data:image/jpeg;base64," +
                    base64.b64encode(buf).decode()) if ok else None

        return {
            "heatmapDataUrl": data_url,
            "focusRegion": region,
            "note": f"Model attention concentrated around {region}.",
        }

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


# ---------------------------------------------------------- HF verifiers

class _HFEngine:
    """Generic wrapper around a HuggingFace image-classification model.
    Maps whatever labels the model uses onto P(fake) via regex."""

    def __init__(self, model_id, name):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.name = name
        self.torch = torch
        # local_files_only: inference must NEVER wait on the network —
        # a partially-downloaded model fails instantly and gets skipped.
        self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True)
        self.model.eval()

        # Which output index means "fake"?
        self.fake_idx = None
        for idx, label in self.model.config.id2label.items():
            if _FAKE_LABEL.search(str(label)):
                self.fake_idx = int(idx)
                break
        if self.fake_idx is None:
            raise ValueError(f"{model_id}: no fake-like label in {self.model.config.id2label}")

    def p_fake(self, pil_image):
        inputs = self.processor(images=pil_image.convert("RGB"), return_tensors="pt")
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**inputs).logits, dim=1)[0]
        return float(probs[self.fake_idx])


_hf_engines = None  # lazy: None = not tried yet, [] = tried, none loaded


def _get_hf_engines():
    global _hf_engines
    if _hf_engines is None:
        _hf_engines = []
        for cfg in HF_MODELS:
            try:
                _hf_engines.append(_HFEngine(cfg["id"], cfg["name"]))
            except Exception:
                pass  # offline / not downloaded / bad model → skip
    return _hf_engines


# ---------------------------------------------------------- public API

def analyze_file(path, file_type, frame_rate=1.0):
    """Main entry for app.py.
    → dict {prediction, confidence, framesAnalyzed, ensemble?}"""
    eng = _get_engine()

    # Video: our fast model only — running ViTs per-frame is too slow on CPU
    if file_type == "video":
        (prediction, confidence), frames = eng.predict_video(path, frame_rate)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "framesAnalyzed": frames,
            "ensemble": [{"model": "MobileNetV3 (ours)", "pFake": None,
                          "note": "video mode — single fast model"}],
        }

    # Image: ensemble — our model + every available HF verifier vote P(fake)
    from PIL import Image
    with Image.open(path) as im:
        face, landmarks = eng._face_crop_ex(im)
        probs = eng._probs_raw(face)
        fake_i = eng.classes.index("fake")
        votes = [{"model": "MobileNetV3 (ours)", "pFake": round(float(probs[fake_i]), 4)}]

        for hf in _get_hf_engines():
            try:
                votes.append({"model": hf.name, "pFake": round(hf.p_fake(face), 4)})
            except Exception:
                pass

        # Explainability (Grad-CAM heatmap + grounded focus text) —
        # best-effort: any failure just omits the section
        try:
            explain = eng.explain(face, landmarks)
        except Exception:
            explain = None

    # Weighted vote: our model leads (it's calibrated to our domain and
    # demo data); pretrained verifiers are advisory — testing showed they
    # carry their own dataset biases (one flagged an authentic official
    # portrait at 0.67 fake), so they must not be able to outvote alone.
    OWN_WEIGHT = 0.65
    if len(votes) > 1:
        w_hf = (1.0 - OWN_WEIGHT) / (len(votes) - 1)
        weights = [OWN_WEIGHT] + [w_hf] * (len(votes) - 1)
    else:
        weights = [1.0]
    for v, w in zip(votes, weights):
        v["weight"] = round(w, 3)

    p = sum(v["pFake"] * v["weight"] for v in votes)
    prediction = "deepfake" if p >= 0.5 else "real"
    confidence = int(round((p if p >= 0.5 else 1 - p) * 100))
    disputed = any((v["pFake"] >= 0.5) != (p >= 0.5) for v in votes)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "framesAnalyzed": 1,
        "ensemble": votes,
        "disputed": disputed,
        "explain": explain,
    }
