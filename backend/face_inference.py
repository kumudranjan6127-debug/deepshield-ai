"""DeepShield V3 face-manipulation inference engine.

This module owns the face-trained detector only.  ``inference.py`` layers the
full-frame AI-origin detector on top of it.  Keeping the two responsibilities
separate lets us preserve the V3 model's behaviour while adding synthetic-media
coverage without pretending the models were trained for the same task.

Serving rules:
- score every detected face and let the most suspicious face decide;
- no-face media is inconclusive for this face-trained model;
- production preprocessing is resolution cap -> YuNet crop -> q88 JPEG ->
  224px ImageNet normalization;
- image explanations use occlusion sensitivity;
- video uses deterministic sampling and median/mean/top-k aggregation.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import re
import statistics
from typing import Iterable

from config import CFG

log = logging.getLogger("deepshield")
BASE_DIR = CFG.BASE_DIR
CKPT_PATH = CFG.CKPT_PATH
ONNX_PATH = CFG.ONNX_PATH
ONNX_META_PATH = CFG.ONNX_META_PATH
YUNET_PATH = CFG.YUNET_PATH

ARCH_NAMES = {
    "mobilenet_v3_small": "MobileNetV3-Small",
    "mobilenet_v3_large": "MobileNetV3-Large",
}
ARCH_PARAMS = {
    "mobilenet_v3_small": "2.5M",
    "mobilenet_v3_large": "5.4M",
}

HF_MODELS = [
    {"id": "prithivMLmods/Deep-Fake-Detector-v2-Model", "name": "ViT Deepfake v2"},
    {"id": "Ateeqq/ai-vs-human-image-detector", "name": "SigLIP AI-image"},
]
_FAKE_LABEL = re.compile(r"fake|deep|ai|artificial|synthetic|generat", re.I)

_engine = None
_engine_mtime = None
_hf_engines = None


def version_from(meta: dict) -> str:
    value = str(meta.get("version") or meta.get("trained_on") or "")
    token = value.split(":", 1)[0].split()[0] if value else ""
    return token if token.lower().startswith("v") else "unversioned"


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
    """Return contiguous half-open bands; the top band includes 100."""
    ordered = list(CFG.CERTAINTY_BANDS)
    return [
        {
            "from": lower,
            "to": (100 if i == 0 else ordered[i - 1][0]),
            "key": key,
            "label": label,
        }
        for i, (lower, key, label) in enumerate(ordered)
    ]


def aggregate_frames(p_fakes: Iterable[float], weights=None, topk_fraction=None,
                     suspicious_at=None) -> dict:
    ps = [min(1.0, max(0.0, float(p))) for p in p_fakes]
    if not ps:
        raise ValueError("no frame scores to aggregate")

    allowed = ("median", "mean", "top_k")
    raw = dict(weights or CFG.VIDEO_WEIGHTS)
    clean = {name: max(0.0, float(raw.get(name, 0.0))) for name in allowed}
    total = sum(clean.values())
    if total <= 0:
        clean = {"median": 1.0, "mean": 0.0, "top_k": 0.0}
        total = 1.0
    clean = {name: value / total for name, value in clean.items()}

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
    threshold = (CFG.VIDEO_SUSPICIOUS_AT if suspicious_at is None
                 else min(1.0, max(0.0, float(suspicious_at))))
    return {
        "score": score,
        "components": {name: round(value, 4) for name, value in parts.items()},
        "weights": {name: round(value, 4) for name, value in clean.items()},
        "k": k,
        "frames": len(ps),
        "suspicious": sum(p >= threshold for p in ps),
        "suspiciousAt": threshold,
        "peak": max(ps),
        "lowest": min(ps),
        "variance": statistics.pvariance(ps) if len(ps) > 1 else 0.0,
    }


def timestamp(seconds) -> str:
    total = max(0, int(round(float(seconds))))
    return f"{total // 60:02d}:{total % 60:02d}"


def temporal_signals(records) -> dict:
    """Descriptive face-motion signals.  They never vote on the verdict."""
    import numpy as np

    faces = [r for r in records if r.get("box")]
    out = {"facesFound": len(faces), "framesSampled": len(records)}
    if len(faces) < 2:
        return {
            **out,
            "facePositionJitter": None,
            "faceSizeJitter": None,
            "landmarkJitter": None,
            "appearanceContinuity": None,
        }

    cx = np.array([(r["box"][0] + r["box"][2] / 2) / r["frame"][0] for r in faces])
    cy = np.array([(r["box"][1] + r["box"][3] / 2) / r["frame"][1] for r in faces])
    out["facePositionJitter"] = round(float((cx.std() + cy.std()) / 2), 4)

    sizes = np.array([np.sqrt(max(r["box"][2] * r["box"][3], 1e-6)) for r in faces])
    out["faceSizeJitter"] = round(float(sizes.std() / sizes.mean()), 4) if sizes.mean() else None

    steps = []
    for a, b in zip(faces, faces[1:]):
        if not (a.get("landmarks") and b.get("landmarks")):
            continue
        width = max(b["box"][2], 1e-6)
        shared = set(a["landmarks"]) & set(b["landmarks"])
        if not shared:
            continue
        moved = []
        for name in shared:
            ax, ay = a["landmarks"][name]
            bx, by = b["landmarks"][name]
            ax += a["origin"][0]; ay += a["origin"][1]
            bx += b["origin"][0]; by += b["origin"][1]
            moved.append(np.hypot(bx - ax, by - ay) / width)
        if moved:
            steps.append(float(np.mean(moved)))
    out["landmarkJitter"] = round(float(np.mean(steps)), 4) if steps else None

    similarities = []
    for a, b in zip(faces, faces[1:]):
        ta, tb = a.get("thumb"), b.get("thumb")
        if ta is None or tb is None:
            continue
        va, vb = np.asarray(ta, float).ravel(), np.asarray(tb, float).ravel()
        if va.std() < 1e-6 or vb.std() < 1e-6:
            continue
        similarities.append(float(np.corrcoef(va, vb)[0, 1]))
    out["appearanceContinuity"] = (
        round(float(np.mean(similarities)), 4) if similarities else None
    )
    return out


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


def verifiers_enabled() -> bool:
    return bool(CFG.VERIFIERS)


def engine_info() -> dict:
    if not engine_available():
        return {}
    return {**_get_engine().info, "verifiers": verifiers_enabled()}


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
            "engine": "live",
            "model_name": self.meta.get("model_name", "DeepShield"),
            "architecture": ARCH_NAMES.get(self.arch, self.arch),
            "version": self.meta.get("version") or version_from(self.meta),
            "runtime": "ONNX" if self.backend == "onnx" else "PyTorch",
            "input_size": self.size,
            "classes": list(self.classes),
            "backend": self.backend,
            "checkpoint": self.checkpoint_name,
            "arch": self.arch,
            "params": ARCH_PARAMS.get(self.arch),
            "val_accuracy": self.meta.get("val_accuracy"),
            "test_accuracy": self.meta.get("test_accuracy"),
            "tpdn_accuracy": self.meta.get("tpdn_accuracy"),
            "dfdc_accuracy": self.meta.get("dfdc_accuracy"),
            "trained_on": self.meta.get("trained_on"),
            "calibrated": bool(self.meta.get("calibrated", False)),
        }
        self._detector = None

    def _init_onnx(self):
        import cv2
        import json
        with open(ONNX_META_PATH, encoding="utf-8") as f:
            self.meta = json.load(f)
        self.backend = "onnx"
        self.checkpoint_name = os.path.basename(ONNX_PATH)
        self.arch = self.meta.get("arch", "mobilenet_v3_large")
        self.classes = list(self.meta["classes"])
        self.size = int(self.meta.get("input_size", 224))
        self.norm = self.meta.get(
            "normalize",
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        )
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
        self.norm = ckpt.get(
            "normalize",
            {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        )
        arch = ckpt.get("arch", "mobilenet_v3_small")
        builders = {
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
        }
        if arch not in builders:
            raise ValueError(f"Unsupported arch in checkpoint: {arch}")
        self.arch = arch
        model = builders[arch](weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, len(self.classes))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        self.model = model
        self.torch = torch

    def _to_input(self, image):
        from PIL import Image
        arr = self.np.asarray(
            image.convert("RGB").resize((self.size, self.size), Image.BILINEAR),
            dtype=self.np.float32,
        ) / 255.0
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
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = self.np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def _normalize_compression(self, image, quality=None):
        from PIL import Image
        quality = CFG.JPEG_NORMALISE_QUALITY if quality is None else int(quality)
        try:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, "JPEG", quality=quality)
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
            rgb = cv2.resize(
                rgb,
                (max(1, int(rgb.shape[1] * scale)), max(1, int(rgb.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        h, w = rgb.shape[:2]
        miss = {
            "crop": Image.fromarray(rgb),
            "landmarks": None,
            "box": None,
            "origin": (0, 0),
            "frame": (w, h),
            "found": False,
        }
        if not os.path.exists(YUNET_PATH):
            return [miss]
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN_create(
                YUNET_PATH, "", (320, 320), 0.6, 0.3, 5000
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr)
        if faces is None or len(faces) == 0:
            return [miss]

        cap = CFG.MAX_FACES if limit is None else max(1, int(limit))
        order = np.argsort(-(faces[:, 2] * faces[:, 3]))[:cap]
        names = ["right_eye", "left_eye", "nose", "mouth_right", "mouth_left"]
        out = []
        for row in faces[order]:
            x, y, fw, fh = [float(v) for v in row[:4]]
            margin = 0.35 * max(fw, fh)
            x0, y0 = max(0, int(x - margin)), max(0, int(y - margin))
            x1, y1 = min(w, int(x + fw + margin)), min(h, int(y + fh + margin))
            if x1 <= x0 or y1 <= y0:
                continue
            landmarks = {
                name: (float(row[4 + i * 2] - x0), float(row[5 + i * 2] - y0))
                for i, name in enumerate(names)
            }
            out.append({
                "crop": Image.fromarray(rgb[y0:y1, x0:x1]),
                "landmarks": landmarks,
                "box": (x, y, fw, fh),
                "origin": (x0, y0),
                "frame": (w, h),
                "found": True,
            })
        return out or [miss]

    def _detect_face(self, image):
        """Compatibility helper: return the largest detected face."""
        return self._detect_faces(image, limit=1)[0]

    def _face_crop_ex(self, image):
        found = self._detect_face(image)
        return found["crop"], found["landmarks"]

    def _face_crop(self, image):
        return self._face_crop_ex(image)[0]

    def _verdict(self, probs):
        fake_i = self.classes.index("fake")
        p = float(probs[fake_i])
        return (("deepfake", int(round(p * 100))) if p >= 0.5
                else ("real", int(round((1 - p) * 100))))

    def predict_image(self, path):
        from PIL import Image
        with Image.open(path) as image:
            detections = self._detect_faces(image)
            found = [d for d in detections if d["found"]]
            if not found:
                return {
                    "prediction": "real",
                    "confidence": 50,
                    "pFake": 0.5,
                    "faceFound": False,
                    "facesFound": 0,
                    "framesAnalyzed": 1,
                    "insufficientEvidence": True,
                    "reason": "No face detected; the face-trained model cannot provide a reliable verdict.",
                    "selectedFace": None,
                }
            fake_i = self.classes.index("fake")
            scored = [(self._probs_raw(d["crop"]), d) for d in found]
            probs, selected = max(scored, key=lambda item: float(item[0][fake_i]))
            p_fake = float(probs[fake_i])
            prediction, confidence = self._verdict(probs)
            return {
                "prediction": prediction,
                "confidence": confidence,
                "pFake": p_fake,
                "faceFound": True,
                "facesFound": len(found),
                "framesAnalyzed": 1,
                "insufficientEvidence": False,
                "selectedFace": selected,
            }

    def predict_video(self, path, frame_rate=None, max_frames=None):
        import cv2
        from PIL import Image

        frame_rate = CFG.DEFAULT_FRAME_RATE if frame_rate is None else float(frame_rate)
        max_frames = CFG.MAX_VIDEO_FRAMES if max_frames is None else int(max_frames)
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps == fps and fps > 0 else 25.0
        step = max(1, round(fps / max(0.25, frame_rate)))
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
                detections = [d for d in self._detect_faces(Image.fromarray(rgb)) if d["found"]]
                if detections:
                    scored = [(float(self._probs_raw(d["crop"])[fake_i]), d) for d in detections]
                    p_fake, selected = max(scored, key=lambda item: item[0])
                    grey = cv2.cvtColor(
                        self.np.asarray(selected["crop"].convert("RGB")), cv2.COLOR_RGB2GRAY
                    )
                    thumb = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA)
                    record = {
                        "index": idx,
                        "time": idx / fps,
                        "pFake": p_fake,
                        "facesFound": len(detections),
                        "box": selected["box"],
                        "origin": selected["origin"],
                        "frame": selected["frame"],
                        "landmarks": selected["landmarks"],
                        "thumb": thumb,
                    }
                else:
                    record = {
                        "index": idx,
                        "time": idx / fps,
                        "pFake": 0.5,
                        "facesFound": 0,
                        "box": None,
                        "origin": (0, 0),
                        "frame": (rgb.shape[1], rgb.shape[0]),
                        "landmarks": None,
                        "thumb": None,
                    }
                records.append(record)
                idx += 1
        finally:
            cap.release()
        if not records:
            raise ValueError("No readable frames in video")
        return records, {"fps": fps, "step": step, "duration": idx / fps}

    def explain(self, face_img, landmarks, grid=None):
        import cv2
        import numpy as np

        grid = CFG.OCCLUSION_GRID if grid is None else max(2, int(grid))
        img = self._normalize_compression(face_img.convert("RGB"))
        base_input = self._to_input(img)
        base_probs = self._forward(base_input[None])[0]
        cls = int(base_probs.argmax())
        base_score = float(base_probs[cls])

        size = self.size
        patch = max(1, size // grid)
        variants, cells = [], []
        for gy in range(grid):
            for gx in range(grid):
                variant = base_input.copy()
                y0, x0 = gy * patch, gx * patch
                y1 = size if gy == grid - 1 else min(size, y0 + patch)
                x1 = size if gx == grid - 1 else min(size, x0 + patch)
                variant[:, y0:y1, x0:x1] = 0.0
                variants.append(variant)
                cells.append((gy, gx))

        scores = self._forward(np.stack(variants))[:, cls]
        cam = np.zeros((grid, grid), dtype=np.float32)
        for (gy, gx), score in zip(cells, scores):
            cam[gy, gx] = max(0.0, base_score - float(score))
        if cam.max() > 0:
            cam /= cam.max()

        face_w, face_h = face_img.size
        iy, ix = np.unravel_index(int(np.argmax(cam)), cam.shape)
        hot_x = (ix + 0.5) / grid * face_w
        hot_y = (iy + 0.5) / grid * face_h
        region_map = {
            "right_eye": "the eye region",
            "left_eye": "the eye region",
            "nose": "the nose area",
            "mouth_right": "the mouth area",
            "mouth_left": "the mouth area",
        }

        def region_at(px, py):
            if not landmarks:
                return None
            nearest = min(
                landmarks.items(),
                key=lambda item: (item[1][0] - px) ** 2 + (item[1][1] - py) ** 2,
            )[0]
            return region_map.get(nearest)

        focus = region_at(hot_x, hot_y) or "the face overall"
        by_region = {}
        for (gy, gx), value in zip(cells, cam.reshape(-1)):
            if value <= 0:
                continue
            px = (gx + 0.5) / grid * face_w
            py = (gy + 0.5) / grid * face_h
            name = region_at(px, py) or "the face overall"
            by_region[name] = max(by_region.get(name, 0.0), float(value))
        ranked = sorted(by_region.items(), key=lambda item: item[1], reverse=True)
        if ranked:
            top_weight = ranked[0][1] or 1.0
            regions = [
                {"name": name, "weight": round(weight / top_weight, 3)}
                for name, weight in ranked
                if weight >= 0.25 * top_weight
            ][:3]
            # The focus must describe the strongest measured region.
            focus = regions[0]["name"]
        else:
            regions = [{"name": focus, "weight": 1.0}]

        base = cv2.resize(np.asarray(face_img.convert("RGB")), (224, 224))
        heat = cv2.resize(
            (cam * 255).astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC
        )
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            cv2.cvtColor(base, cv2.COLOR_RGB2BGR), 0.55, heat, 0.45, 0
        )
        ok, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        data_url = (
            "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None
        )
        return {
            "heatmapDataUrl": data_url,
            "focusRegion": focus,
            "regions": regions,
            "method": "occlusion sensitivity",
            "note": f"Prediction was most sensitive to {focus}.",
        }


class _HFEngine:
    def __init__(self, model_id, name):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.name = name
        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True)
        self.model.eval()
        self.fake_idx = None
        for idx, label in self.model.config.id2label.items():
            if _FAKE_LABEL.search(str(label)):
                self.fake_idx = int(idx)
                break
        if self.fake_idx is None:
            raise ValueError(f"{model_id}: no fake-like label")

    def p_fake(self, image):
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**inputs).logits, dim=1)[0]
        return float(probs[self.fake_idx])


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
                pass
    return _hf_engines


def _active_model_path():
    return ONNX_PATH if onnx_available() else CKPT_PATH


def _get_engine():
    global _engine, _engine_mtime
    path = _active_model_path()
    stamp = (path, os.path.getmtime(path))
    if _engine is None or _engine_mtime != stamp:
        _engine, _engine_mtime = _Engine(), stamp
    return _engine


def _combine_votes(votes):
    own_weight = CFG.OWN_WEIGHT
    if len(votes) > 1:
        other = (1.0 - own_weight) / (len(votes) - 1)
        weights = [own_weight] + [other] * (len(votes) - 1)
    else:
        weights = [1.0]
    for vote, weight in zip(votes, weights):
        vote["weight"] = round(weight, 3)
    p_fake = sum(v["pFake"] * v["weight"] for v in votes)
    verifier_scores = [v["pFake"] for v in votes[1:]]
    if verifier_scores and all(x >= CFG.VERIFIER_OVERRULE_AT for x in verifier_scores):
        p_fake = max(p_fake, sum(verifier_scores) / len(verifier_scores))
    return min(1.0, max(0.0, p_fake))


def analyze_file(path, file_type, frame_rate=None):
    eng = _get_engine()
    if file_type == "video":
        records, meta = eng.predict_video(path, frame_rate)
        face_records = [r for r in records if r.get("facesFound", 0) > 0]
        if face_records:
            agg = aggregate_frames([r["pFake"] for r in face_records])
            score = float(agg["score"])
            prediction = "deepfake" if score > 0.5 else "real"
            confidence = int(round(max(score, 1 - score) * 100))
            hottest = sorted(face_records, key=lambda r: r["pFake"], reverse=True)
        else:
            agg = aggregate_frames([0.5])
            score = 0.5
            prediction, confidence, hottest = "real", 50, []
        return {
            "prediction": prediction,
            "confidence": confidence,
            "framesAnalyzed": len(records),
            "faceFound": bool(face_records),
            "facesFound": max((r.get("facesFound", 0) for r in records), default=0),
            "insufficientEvidence": not bool(face_records),
            "reason": (None if face_records else
                       "No faces were detected in the sampled frames; result is inconclusive."),
            "video": {
                "framesAnalyzed": len(records),
                "faceFramesAnalyzed": len(face_records),
                "noFaceFrames": len(records) - len(face_records),
                "suspiciousFrames": agg["suspicious"] if face_records else 0,
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
                    {"time": round(r["time"], 2), "timestamp": timestamp(r["time"]),
                     "score": round(r["pFake"], 4)}
                    for r in hottest[:CFG.VIDEO_TOP_TIMESTAMPS]
                ],
                "timeline": [
                    {"t": round(r["time"], 2),
                     "p": round(r["pFake"], 4) if r.get("facesFound", 0) else None,
                     "facesFound": int(r.get("facesFound", 0))}
                    for r in records
                ],
                "temporal": temporal_signals(records),
                "fps": round(meta["fps"], 2),
                "sampledEveryNthFrame": meta["step"],
                "durationSeconds": round(meta["duration"], 2),
            },
            "ensemble": ([{"model": "MobileNetV3 (ours)", "pFake": round(score, 4),
                           "weight": 1.0}] if face_records else []),
            "explain": None,
        }

    result = eng.predict_image(path)
    selected = result.pop("selectedFace", None)
    if not result["faceFound"] or selected is None:
        result["ensemble"] = []
        result["disputed"] = False
        result["combiner"] = "face detector unavailable"
        result["explain"] = None
        result.pop("pFake", None)
        return result

    own_p = float(result.pop("pFake"))
    votes = [{"model": "MobileNetV3 (ours)", "pFake": round(own_p, 4)}]
    for hf in _get_hf_engines():
        try:
            votes.append({"model": hf.name, "pFake": round(hf.p_fake(selected["crop"]), 4)})
        except Exception:
            pass
    p_fake = _combine_votes(votes)
    result["prediction"] = "deepfake" if p_fake >= 0.5 else "real"
    result["confidence"] = int(round(max(p_fake, 1.0 - p_fake) * 100))
    result["ensemble"] = votes
    result["disputed"] = any((v["pFake"] >= 0.5) != (p_fake >= 0.5) for v in votes)
    result["combiner"] = "own-led + verifier consensus"
    try:
        result["explain"] = eng.explain(selected["crop"], selected["landmarks"])
    except Exception:
        log.exception("explainability failed")
        result["explain"] = None
    return result


def score_image(path) -> float:
    """Precise V3 face score without explanation/HF overhead."""
    from PIL import Image
    eng = _get_engine()
    fake_i = eng.classes.index("fake")
    with Image.open(path) as image:
        faces = [d for d in eng._detect_faces(image) if d["found"]]
        if not faces:
            return 0.5
        return max(float(eng._probs_raw(d["crop"])[fake_i]) for d in faces)
