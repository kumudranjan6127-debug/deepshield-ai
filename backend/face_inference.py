"""DeepShield inference engine.

The serving path is deliberately conservative:
- only a loaded checkpoint/ONNX model can produce a live verdict;
- face-trained models score detected face crops, not arbitrary backgrounds;
- every detected face is considered, so a manipulated face cannot be outvoted by
  another person in a group photo;
- no-face media is reported as insufficient evidence (50% confidence), never as
  a high-confidence real/fake verdict;
- video frames use the most suspicious detected face per frame and robust
  median/mean/top-k aggregation.
"""
import io
import logging
import os
import statistics
from typing import Iterable

from config import CFG

log = logging.getLogger("deepshield")
BASE_DIR = CFG.BASE_DIR
CKPT_PATH = CFG.CKPT_PATH
ONNX_PATH = CFG.ONNX_PATH
ONNX_META_PATH = CFG.ONNX_META_PATH
YUNET_PATH = CFG.YUNET_PATH

ARCH_NAMES = {"mobilenet_v3_small": "MobileNetV3-Small", "mobilenet_v3_large": "MobileNetV3-Large"}
ARCH_PARAMS = {"mobilenet_v3_small": "2.5M", "mobilenet_v3_large": "5.4M"}
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
    return "low_evidence"


def certainty_bands() -> list:
    bands = []
    ordered = sorted(CFG.CERTAINTY_BANDS, key=lambda x: x[0], reverse=True)
    for i, (lower, key, label) in enumerate(ordered):
        upper = ordered[i - 1][0] - 1 if i else 100
        bands.append({"from": lower, "to": upper, "key": key, "label": label})
    return bands


def aggregate_frames(p_fakes: Iterable[float], weights=None, topk_fraction=None,
                     suspicious_at=None) -> dict:
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
    parts = {"median": statistics.median(ps), "mean": sum(ps) / len(ps), "top_k": sum(top) / len(top)}
    score = sum(parts[k] * clean[k] for k in allowed)
    threshold = CFG.VIDEO_SUSPICIOUS_AT if suspicious_at is None else min(1.0, max(0.0, float(suspicious_at)))
    return {
        "score": score, "components": {k: round(v, 4) for k, v in parts.items()},
        "weights": {k: round(v, 4) for k, v in clean.items()}, "k": k, "frames": len(ps),
        "suspicious": sum(p >= threshold for p in ps), "suspiciousAt": threshold,
        "peak": max(ps), "lowest": min(ps), "variance": statistics.pvariance(ps) if len(ps) > 1 else 0.0,
    }


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


def onnx_available() -> bool:
    return os.path.exists(ONNX_PATH) and os.path.exists(ONNX_META_PATH)


def engine_available() -> bool:
    if CFG.FORCE_ECHO:
        return False
    return onnx_available() or (os.path.exists(CKPT_PATH) and torch_available())


def version_from(meta: dict) -> str:
    value = str(meta.get("version") or meta.get("trained_on") or "")
    token = value.split(":", 1)[0].split()[0] if value else ""
    return token if token.lower().startswith("v") else "unversioned"


def engine_info() -> dict:
    if not engine_available():
        return {}
    return {**_get_engine().info, "verifiers": bool(CFG.VERIFIERS)}


class _Engine:
    MAX_BATCH = CFG.MAX_FORWARD_BATCH

    def __init__(self):
        import numpy as np
        self.np = np
        if onnx_available():
            self._init_onnx()
        else:
            self._init_torch()
        self.info = {
            "engine": "live", "model_name": self.meta.get("model_name", "DeepShield"),
            "architecture": ARCH_NAMES.get(self.arch, self.arch),
            "version": self.meta.get("version") or version_from(self.meta),
            "runtime": "ONNX" if self.backend == "onnx" else "PyTorch",
            "input_size": self.size, "classes": list(self.classes),
            "backend": self.backend, "checkpoint": self.checkpoint_name,
            "arch": self.arch, "params": ARCH_PARAMS.get(self.arch),
            "val_accuracy": self.meta.get("val_accuracy"),
            "test_accuracy": self.meta.get("test_accuracy"),
            "tpdn_accuracy": self.meta.get("tpdn_accuracy"),
            "dfdc_accuracy": self.meta.get("dfdc_accuracy"),
            "trained_on": self.meta.get("trained_on"),
            "calibrated": bool(self.meta.get("calibrated", False)),
        }
        self._detector = None

    def _init_onnx(self):
        import cv2, json
        with open(ONNX_META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.backend = "onnx"
        self.checkpoint_name = os.path.basename(ONNX_PATH)
        self.arch = self.meta.get("arch", "mobilenet_v3_large")
        self.classes = list(self.meta["classes"])
        self.size = int(self.meta.get("input_size", 224))
        self.norm = self.meta.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
        if "fake" not in self.classes or "real" not in self.classes:
            raise ValueError("model classes must contain 'fake' and 'real'")
        self.net = cv2.dnn.readNetFromONNX(ONNX_PATH)
        self.torch = None

    def _init_torch(self):
        import torch
        from torchvision import models
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        self.meta = ckpt
        self.backend = "torch"
        self.checkpoint_name = os.path.basename(CKPT_PATH)
        self.classes = list(ckpt["classes"])
        self.size = int(ckpt.get("input_size", 224))
        self.norm = ckpt.get("normalize", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
        arch = ckpt.get("arch", "mobilenet_v3_small")
        builders = {"mobilenet_v3_small": models.mobilenet_v3_small, "mobilenet_v3_large": models.mobilenet_v3_large}
        if arch not in builders:
            raise ValueError(f"Unsupported arch in checkpoint: {arch}")
        self.arch = arch
        model = builders[arch](weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        self.model, self.torch = model, torch

    def _to_input(self, image):
        from PIL import Image
        arr = self.np.asarray(image.convert("RGB").resize((self.size, self.size), Image.BILINEAR), dtype=self.np.float32) / 255.0
        mean = self.np.asarray(self.norm["mean"], dtype=self.np.float32)
        std = self.np.asarray(self.norm["std"], dtype=self.np.float32)
        return ((arr - mean) / std).transpose(2, 0, 1)

    def _forward(self, batch):
        parts = []
        for i in range(0, len(batch), self.MAX_BATCH):
            chunk = self.np.ascontiguousarray(batch[i:i + self.MAX_BATCH], dtype=self.np.float32)
            if self.backend == "onnx":
                self.net.setInput(chunk)
                parts.append(self.net.forward())
            else:
                with self.torch.no_grad():
                    parts.append(self.model(self.torch.from_numpy(chunk)).cpu().numpy())
        logits = self.np.concatenate(parts, axis=0)
        logits -= logits.max(axis=1, keepdims=True)
        exp = self.np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _normalize_compression(self, image):
        from PIL import Image
        try:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, "JPEG", quality=CFG.JPEG_NORMALISE_QUALITY)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        except Exception:
            return image.convert("RGB")

    def _probs_raw(self, image):
        from PIL import ImageOps
        image = self._normalize_compression(image)
        batch = self.np.stack([self._to_input(image), self._to_input(ImageOps.mirror(image))])
        return self._forward(batch).mean(axis=0)

    def _detect_faces(self, image, limit=None):
        import cv2
        import numpy as np
        from PIL import Image
        image = image.convert("RGB")
        rgb = np.asarray(image)
        if max(rgb.shape[:2]) > CFG.MAX_IMAGE_SIDE:
            scale = CFG.MAX_IMAGE_SIDE / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        miss = {"crop": Image.fromarray(rgb), "landmarks": None, "box": None, "origin": (0, 0), "frame": (w, h), "found": False}
        if not os.path.exists(YUNET_PATH):
            return [miss]
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN_create(YUNET_PATH, "", (320, 320), 0.6, 0.3, 5000)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return [miss]
        names = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
        out = []
        for row in faces[np.argsort(-(faces[:, 2] * faces[:, 3]))[:(CFG.MAX_FACES if limit is None else limit)]]:
            x, y, fw, fh = [float(v) for v in row[:4]]
            margin = 0.35 * max(fw, fh)
            x0, y0 = max(0, int(x - margin)), max(0, int(y - margin))
            x1, y1 = min(w, int(x + fw + margin)), min(h, int(y + fh + margin))
            if x1 <= x0 or y1 <= y0:
                continue
            landmarks = {}
            for i, name in enumerate(names):
                landmarks[name] = (float(row[4 + i * 2] - x0), float(row[5 + i * 2] - y0))
            out.append({"crop": Image.fromarray(rgb[y0:y1, x0:x1]), "landmarks": landmarks,
                        "box": (x, y, fw, fh), "origin": (x0, y0), "frame": (w, h), "found": True})
        return out or [miss]

    def _verdict(self, probs):
        fake_i = self.classes.index("fake")
        p = float(probs[fake_i])
        if p >= 0.5:
            return "deepfake", int(round(p * 100))
        return "real", int(round((1 - p) * 100))

    def predict_image(self, path):
        from PIL import Image
        with Image.open(path) as image:
            detected = self._detect_faces(image)
            found = [d for d in detected if d["found"]]
            if not found:
                return {"prediction": "real", "confidence": 50, "faceFound": False, "facesFound": 0,
                        "insufficientEvidence": True, "reason": "No face detected; the face-trained model cannot provide a reliable verdict."}
            scored = [(self._probs_raw(d["crop"]), d) for d in found]
            probs, selected = max(scored, key=lambda x: float(x[0][self.classes.index("fake")]))
            prediction, confidence = self._verdict(probs)
            return {"prediction": prediction, "confidence": confidence, "faceFound": True,
                    "facesFound": len(found), "framesAnalyzed": 1, "selectedFace": selected}

    def predict_video(self, path, frame_rate=CFG.DEFAULT_FRAME_RATE, max_frames=CFG.MAX_VIDEO_FRAMES):
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
                    if not cap.grab(): break
                    idx += 1; continue
                ok, frame = cap.read()
                if not ok: break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = [d for d in self._detect_faces(Image.fromarray(rgb)) if d["found"]]
                if not faces:
                    p = 0.5
                    face_count = 0
                else:
                    scores = [float(self._probs_raw(d["crop"])[fake_i]) for d in faces]
                    p = max(scores)
                    face_count = len(faces)
                records.append({"index": idx, "time": idx / fps, "pFake": p, "facesFound": face_count})
                idx += 1
        finally:
            cap.release()
        if not records: raise ValueError("No readable frames in video")
        return records, {"fps": fps, "step": step, "duration": idx / fps}


def _active_model_path():
    return ONNX_PATH if onnx_available() else CKPT_PATH


def _get_engine():
    global _engine, _engine_mtime
    path = _active_model_path()
    stamp = (path, os.path.getmtime(path))
    if _engine is None or _engine_mtime != stamp:
        _engine, _engine_mtime = _Engine(), stamp
    return _engine


def analyze_file(path, file_type, frame_rate=CFG.DEFAULT_FRAME_RATE):
    eng = _get_engine()
    if file_type == "image":
        result = eng.predict_image(path)
        return result
    records, meta = eng.predict_video(path, frame_rate)
    agg = aggregate_frames([r["pFake"] for r in records])
    score = float(agg["score"])
    # A video containing only no-face frames is inconclusive, not real.
    any_face = any(r["facesFound"] > 0 for r in records)
    if not any_face:
        score = 0.5
    prediction = "deepfake" if score >= 0.5 and score > 0.5 else "real"
    confidence = 50 if not any_face else int(round(max(score, 1 - score) * 100))
    hottest = sorted(records, key=lambda r: r["pFake"], reverse=True)
    return {
        "prediction": prediction, "confidence": confidence, "framesAnalyzed": len(records),
        "faceFound": any_face, "facesFound": max((r["facesFound"] for r in records), default=0),
        "insufficientEvidence": not any_face,
        "reason": "No faces were detected in the sampled frames; result is inconclusive." if not any_face else None,
        "video": {
            "framesAnalyzed": len(records), "suspiciousFrames": agg["suspicious"],
            "suspiciousAt": agg["suspiciousAt"], "peakFakeScore": round(agg["peak"], 4),
            "medianFakeScore": round(agg["components"]["median"], 4), "meanFakeScore": round(agg["components"]["mean"], 4),
            "topKFakeScore": round(agg["components"]["top_k"], 4), "lowestFakeScore": round(agg["lowest"], 4),
            "scoreVariance": round(agg["variance"], 6), "combinedScore": round(score, 4), "k": agg["k"],
            "weights": agg["weights"],
            "topTimestamps": [{"time": round(r["time"], 2), "timestamp": f"{int(r['time'])//60:02d}:{int(r['time'])%60:02d}", "score": round(r["pFake"], 4)} for r in hottest[:CFG.VIDEO_TOP_TIMESTAMPS]],
            "timeline": [{"t": round(r["time"], 2), "p": round(r["pFake"], 4)} for r in records],
            "temporal": {"facesFound": sum(r["facesFound"] > 0 for r in records), "framesSampled": len(records)},
            "fps": round(meta["fps"], 2), "sampledEveryNthFrame": meta["step"], "durationSeconds": round(meta["duration"], 2),
        },
        "ensemble": [{"model": "MobileNetV3 (ours)", "pFake": round(score, 4), "weight": 1.0}],
    }


def score_image(path) -> float:
    eng = _get_engine()
    from PIL import Image
    with Image.open(path) as image:
        found = [d for d in eng._detect_faces(image) if d["found"]]
        if not found:
            return 0.5
        fake_i = eng.classes.index("fake")
        return max(float(eng._probs_raw(d["crop"])[fake_i]) for d in found)
