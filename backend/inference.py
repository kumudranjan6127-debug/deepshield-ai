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
ONNX_PATH = os.path.join(BASE_DIR, "models", "deepshield.onnx")
ONNX_META_PATH = ONNX_PATH + ".json"
YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet.onnx")

# ---- Ensemble: pretrained HuggingFace verifiers (images only) ----
# OPT-IN (DS_VERIFIERS=1). They mattered when our model was blind to
# StyleGAN2, but V3 covers those fakes itself and scores every image in
# our held set correctly alone — while the verifiers cost ~1 GB of disk
# and RAM and have flagged authentic photos (SigLIP scored a re-saved
# real portrait 1.00). Turn them on to cross-check a verdict.
HF_MODELS = [
    {"id": "prithivMLmods/Deep-Fake-Detector-v2-Model", "name": "ViT Deepfake v2"},
    {"id": "Ateeqq/ai-vs-human-image-detector",         "name": "SigLIP AI-image"},
]


def verifiers_enabled() -> bool:
    return os.environ.get("DS_VERIFIERS", "").lower() in ("1", "true", "on", "yes")
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


def onnx_available() -> bool:
    """The lean path: OpenCV runs the network, so PyTorch is not needed.
    Requires the exported pair written by scripts/export_onnx.py."""
    return os.path.exists(ONNX_PATH) and os.path.exists(ONNX_META_PATH)


def engine_available() -> bool:
    """True when a runnable model exists — ONNX (preferred) or a PyTorch
    checkpoint. DS_ENGINE=echo forces the openly-labeled simulated mode
    (the UI then shows the yellow 'Simulated (demo)' badge)."""
    if os.environ.get("DS_ENGINE", "").lower() == "echo":
        return False
    if onnx_available():
        return True
    return os.path.exists(CKPT_PATH) and torch_available()


def engine_info() -> dict:
    """Metadata for /api/health (accuracy comes from the checkpoint)."""
    if not engine_available():
        return {}
    return {**_get_engine().info, "verifiers": verifiers_enabled()}


# ---------------------------------------------------------- engine

class _Engine:
    """Runs the trained classifier on CPU.

    Two interchangeable backends, both fed by the same preprocessing:
      'onnx'  — OpenCV's DNN module runs the exported graph. Default,
                and the reason the deployment needs no PyTorch at all.
      'torch' — the original .pth path, used when no ONNX export exists.
    Verified to agree to ~1e-7 on every probability, so which one is
    active never changes a verdict.
    """

    def __init__(self):
        import numpy as np
        self.np = np

        if onnx_available():
            self._init_onnx()
        else:
            self._init_torch()

        self.info = {
            "engine": "live",
            "backend": self.backend,
            "checkpoint": self.checkpoint_name,
            "arch": self.arch,
            "val_accuracy": self.meta.get("val_accuracy"),
            "test_accuracy": self.meta.get("test_accuracy"),
            "tpdn_accuracy": self.meta.get("tpdn_accuracy"),
            "dfdc_accuracy": self.meta.get("dfdc_accuracy"),
            "trained_on": self.meta.get("trained_on"),
        }

        self._detector = None  # lazy YuNet face detector (OpenCV 5 DNN)

    # ---- backend: ONNX through OpenCV (no torch anywhere) ----
    def _init_onnx(self):
        import cv2
        import json

        with open(ONNX_META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.backend = "onnx"
        self.checkpoint_name = os.path.basename(ONNX_PATH)
        self.arch = self.meta.get("arch", "mobilenet_v3_large")
        self.classes = self.meta["classes"]      # ['fake', 'real'] — order matters
        self.size = self.meta.get("input_size", 224)
        self.norm = self.meta["normalize"]
        self.net = cv2.dnn.readNetFromONNX(ONNX_PATH)
        self.torch = None

    # ---- backend: the original PyTorch checkpoint ----
    def _init_torch(self):
        import torch
        from torchvision import models

        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        self.meta = ckpt
        self.backend = "torch"
        self.checkpoint_name = os.path.basename(CKPT_PATH)
        self.classes = ckpt["classes"]
        self.size = ckpt.get("input_size", 224)
        self.norm = ckpt.get("normalize", {"mean": [0.485, 0.456, 0.406],
                                           "std":  [0.229, 0.224, 0.225]})

        arch = ckpt.get("arch", "mobilenet_v3_small")
        builders = {
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
        }
        if arch not in builders:
            raise ValueError(f"Unsupported arch in checkpoint: {arch}")
        self.arch = arch

        model = builders[arch](weights=None)
        model.classifier[3] = torch.nn.Linear(
            model.classifier[3].in_features, len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))  # keep the i3 responsive
        self.model = model
        self.torch = torch

    # ---- shared preprocessing: PIL resize + normalise, in numpy ----
    # Matches torchvision's Resize→ToTensor→Normalize exactly (torchvision
    # resizes PIL images with PIL itself), so both backends see the same
    # array and the swap stays invisible in the results.
    def _to_input(self, pil_image):
        from PIL import Image
        np = self.np
        img = pil_image.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0          # HWC, 0-1
        arr = (arr - np.array(self.norm["mean"], dtype=np.float32)) \
            / np.array(self.norm["std"], dtype=np.float32)
        return arr.transpose(2, 0, 1)                            # CHW

    # ---- shared forward: (N,3,H,W) float32 → (N,classes) probabilities ----
    # OpenCV's DNN engine dies on large batches (36 crashed the process,
    # 16 is fine), so requests are chunked rather than trusted whole.
    MAX_BATCH = 8

    def _forward(self, batch):
        np = self.np
        parts = []
        for i in range(0, len(batch), self.MAX_BATCH):
            chunk = np.ascontiguousarray(batch[i:i + self.MAX_BATCH], dtype=np.float32)
            if self.backend == "onnx":
                self.net.setInput(chunk)
                parts.append(self.net.forward())
            else:
                with self.torch.no_grad():
                    parts.append(self.model(self.torch.from_numpy(chunk)).numpy())
        logits = np.concatenate(parts, axis=0)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

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

        # Cap the working resolution. Cropping straight out of a very large
        # photo hands the model a downsampling path it never saw in training
        # (dataset faces are ~256px): a 2687px press portrait scored 0.94
        # fake, the same photo at 1024px scored 0.02. Normalising the scale
        # first removes that artefact.
        MAX_SIDE = 1024
        if max(rgb.shape[:2]) > MAX_SIDE:
            s = MAX_SIDE / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), int(rgb.shape[0] * s)),
                             interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]

        # Detection runs on the (already capped) image
        scale = 1.0
        det_img = rgb

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
        img = self._normalize_compression(pil_image.convert("RGB"))
        batch = self.np.stack([
            self._to_input(img),
            self._to_input(img.transpose(0)),   # PIL FLIP_LEFT_RIGHT
        ])
        return self._forward(batch).mean(axis=0)

    @staticmethod
    def _normalize_compression(img, quality=88):
        """Put every input in the same compression domain the model trained
        on. Training saw JPEG-recompressed faces (q30-95); a pristine
        camera original carries high-frequency detail it never learned as
        'normal' and scored 0.95 fake, while the same photo re-saved as
        JPEG scored 0.02. One round-trip removes that mismatch.
        Applied to our model only — the verifiers do their own thing, and
        SigLIP in particular reacts badly to re-encoding."""
        import io
        from PIL import Image
        try:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        except Exception:
            return img

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

    # ---- Explainability: occlusion sensitivity ----
    # Blank out one patch at a time and watch the verdict move: the
    # regions whose removal changes the score the most are the ones the
    # model was relying on. Forward passes only, so it behaves identically
    # on both backends (Grad-CAM would need gradients, which the ONNX
    # runtime cannot provide) — and it is easier to justify: we measure
    # the model's dependence rather than interpret its internals.
    def explain(self, face_img, landmarks, grid=6):
        import cv2
        import base64
        np = self.np

        img = self._normalize_compression(face_img.convert("RGB"))
        base_input = self._to_input(img)
        cls = int(self._forward(base_input[None]).argmax())
        base_score = float(self._forward(base_input[None])[0, cls])

        # One batch: the image with each patch greyed out in turn
        S = self.size
        patch = S // grid
        variants, cells = [], []
        for gy in range(grid):
            for gx in range(grid):
                v = base_input.copy()
                y0, x0 = gy * patch, gx * patch
                v[:, y0:y0 + patch, x0:x0 + patch] = 0.0   # 0 = dataset mean
                variants.append(v)
                cells.append((gy, gx))

        scores = self._forward(np.stack(variants))[:, cls]
        cam = np.zeros((grid, grid), dtype=np.float32)
        for (gy, gx), s in zip(cells, scores):
            cam[gy, gx] = max(0.0, base_score - float(s))  # drop = reliance

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
        heat = cv2.resize((cam * 255).astype(np.uint8), (224, 224),
                          interpolation=cv2.INTER_CUBIC)  # smooth the coarse grid
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.55,
                                  heat, 0.45, 0)
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        data_url = ("data:image/jpeg;base64," +
                    base64.b64encode(buf).decode()) if ok else None

        return {
            "heatmapDataUrl": data_url,
            "focusRegion": region,
            "method": "occlusion sensitivity",
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


def _active_model_path() -> str:
    """Whichever file the engine will actually load."""
    return ONNX_PATH if onnx_available() else CKPT_PATH


def _get_engine() -> _Engine:
    """Singleton, but reloads automatically if the model file is replaced
    (dropping in a newly trained model needs no restart)."""
    global _engine, _engine_mtime
    stamp = (_active_model_path(), os.path.getmtime(_active_model_path()))
    if _engine is None or _engine_mtime != stamp:
        _engine = _Engine()
        _engine_mtime = stamp
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
    if not verifiers_enabled():
        return []
    if _hf_engines is None:
        _hf_engines = []
        for cfg in HF_MODELS:
            try:
                _hf_engines.append(_HFEngine(cfg["id"], cfg["name"]))
            except Exception:
                pass  # offline / not downloaded / transformers missing → skip
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

    # ---- Combining the votes ----
    # Our own model leads. Noisy-OR made sense while it was blind to
    # StyleGAN2, but that blindness is gone (V3 trains on those fakes) and
    # trusting any confident vote backfires: SigLIP scores a re-saved
    # authentic photo at 1.00 — it recognises processing, not manipulation.
    # A false "fake" on someone's real photo is the worst outcome we can
    # produce, so verifiers now advise rather than decide.
    OWN_WEIGHT = 0.75
    if len(votes) > 1:
        w_hf = (1.0 - OWN_WEIGHT) / (len(votes) - 1)
        weights = [OWN_WEIGHT] + [w_hf] * (len(votes) - 1)
    else:
        weights = [1.0]
    for v, w in zip(votes, weights):
        v["weight"] = round(w, 3)

    p = sum(v["pFake"] * v["weight"] for v in votes)

    # Verifiers can still overrule, but only together and only when both
    # are near-certain — one biased verifier is never enough.
    verifiers = [v["pFake"] for v in votes[1:]]
    if verifiers and all(x >= 0.85 for x in verifiers):
        p = max(p, sum(verifiers) / len(verifiers))

    prediction = "deepfake" if p >= 0.5 else "real"
    confidence = int(round((p if p >= 0.5 else 1 - p) * 100))
    disputed = any((v["pFake"] >= 0.5) != (p >= 0.5) for v in votes)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "framesAnalyzed": 1,
        "ensemble": votes,
        "disputed": disputed,
        "combiner": "own-led + verifier consensus",
        "explain": explain,
    }
